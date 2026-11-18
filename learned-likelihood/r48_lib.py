"""R48: a learned per-PMT likelihood (FreeDOM-style likelihood-ratio classifier) for single e- tracks.

A network D(g(theta, PMT, obs)) is trained to tell a PMT's observation (hit flag, charge, time) paired with
the TRUE track from the same observation paired with a track drawn from the prior (another event's true
track). Its logit estimates log p(obs | theta) - log p(obs); summed over the PMTs of an event it is the
event log-likelihood up to a theta-independent constant, and is minimised like Coral's own.

theta = (t0, x, y, z, dx, dy, dz, p) on Coral's clock and frame (hit times on the trigger clock).
"""
import numpy as np
import torch

C0 = 29.9792458          # cm/ns
N_GROUP = 1.38           # Coral's refraction index for photon time of flight
TAN_C, SIN_C = np.tan(np.radians(41.4)), np.sin(np.radians(41.4))
NFEAT = 14


class Net(torch.nn.Module):
    def __init__(self, width=256, depth=3):
        super().__init__()
        layers, n = [], NFEAT
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]
            n = width
        layers.append(torch.nn.Linear(n, 1))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def features(theta, pmt_pos, pmt_dir, hit, lq, t, wall):
    """theta (N, 8) per row; pmt_pos / pmt_dir (N, 3); hit, lq, t (N,); wall = (radius, half_height).
    Returns (N, NFEAT) float32 tensor."""
    t0, v, d, p = theta[:, 0], theta[:, 1:4], theta[:, 4:7], theta[:, 7]
    w = pmt_pos - v
    R = torch.linalg.norm(w, dim=1).clamp(min=1.0)
    along = (w*d).sum(1)
    perp = torch.sqrt((R*R - along*along).clamp(min=1e-6))
    coseta = -(w*pmt_dir).sum(1)/R
    s = (along - perp/TAN_C).clamp(min=0.0)
    path = torch.linalg.norm(pmt_pos - (v + s[:, None]*d), dim=1)
    tres_c = t - (t0 + s/C0 + N_GROUP*path/C0)
    tres_v = t - (t0 + N_GROUP*R/C0)
    radius, half = wall
    rv = torch.hypot(v[:, 0], v[:, 1])
    dwall = torch.minimum(radius - rv, half - v[:, 2].abs())
    a = d[:, 0]**2 + d[:, 1]**2
    b = 2*(v[:, 0]*d[:, 0] + v[:, 1]*d[:, 1])
    c = v[:, 0]**2 + v[:, 1]**2 - radius**2
    disc = (b*b - 4*a*c).clamp(min=0.0)
    t_r = torch.where(a > 1e-9, (-b + torch.sqrt(disc))/(2*a).clamp(min=1e-9), torch.full_like(a, 1e5))
    t_z = torch.where(d[:, 2] > 0, (half - v[:, 2])/d[:, 2].clamp(min=1e-9),
                      torch.where(d[:, 2] < 0, (-half - v[:, 2])/d[:, 2].clamp(max=-1e-9), torch.full_like(a, 1e5)))
    towall = torch.minimum(t_r, t_z).clamp(max=1e4)
    h = hit.float()
    return torch.stack([R/1e3, along/R, coseta, along/1e3, perp/1e3, torch.log(p) - 6.0, towall/1e3, dwall/1e3,
                        pmt_pos[:, 2]/3e3, torch.hypot(pmt_pos[:, 0], pmt_pos[:, 1])/3e3, h, lq*h,
                        h*tres_v.clamp(-50, 200)/20, h*tres_c.clamp(-50, 200)/20], dim=1).float()


def theta_from_x(x):
    """Coral's 7-vector (t, x, y, z, theta, phi, p) -> theta (8,)."""
    st = np.sin(x[4])
    return np.array([x[0], x[1], x[2], x[3], st*np.cos(x[5]), st*np.sin(x[5]), np.cos(x[4]), x[6]])


# ---- v2: per-PMT density model trained by maximum likelihood ----------------------------------------------
NGEO, K = 10, 3
LOG2PI = float(np.log(2*np.pi))


def geometry(theta, pmt_pos, pmt_dir, wall):
    """(N, NGEO) track-relative PMT geometry and the Cherenkov emission time t_pred (N,) for theta rows."""
    t0, v, d, p = theta[:, 0], theta[:, 1:4], theta[:, 4:7], theta[:, 7]
    w = pmt_pos - v
    R = torch.linalg.norm(w, dim=1).clamp(min=1.0)
    along = (w*d).sum(1)
    perp = torch.sqrt((R*R - along*along).clamp(min=1e-6))
    coseta = -(w*pmt_dir).sum(1)/R
    s = (along - perp/TAN_C).clamp(min=0.0)
    path = torch.linalg.norm(pmt_pos - (v + s[:, None]*d), dim=1)
    t_pred = t0 + s/C0 + N_GROUP*path/C0
    radius, half = wall
    rv = torch.hypot(v[:, 0], v[:, 1])
    dwall = torch.minimum(radius - rv, half - v[:, 2].abs())
    a = d[:, 0]**2 + d[:, 1]**2
    b = 2*(v[:, 0]*d[:, 0] + v[:, 1]*d[:, 1])
    c = v[:, 0]**2 + v[:, 1]**2 - radius**2
    disc = (b*b - 4*a*c).clamp(min=0.0)
    t_r = torch.where(a > 1e-9, (-b + torch.sqrt(disc))/(2*a).clamp(min=1e-9), torch.full_like(a, 1e5))
    t_z = torch.where(d[:, 2] > 0, (half - v[:, 2])/d[:, 2].clamp(min=1e-9),
                      torch.where(d[:, 2] < 0, (-half - v[:, 2])/d[:, 2].clamp(max=-1e-9), torch.full_like(a, 1e5)))
    towall = torch.minimum(t_r, t_z).clamp(max=1e4)
    g = torch.stack([R/1e3, along/R, coseta, along/1e3, perp/1e3, torch.log(p) - 6.0, towall/1e3, dwall/1e3,
                     pmt_pos[:, 2]/3e3, torch.hypot(pmt_pos[:, 0], pmt_pos[:, 1])/3e3], dim=1).float()
    return g, t_pred


class DensityNet(torch.nn.Module):
    """Per-PMT p(hit | theta), p(log q | hit, theta), p(t - t_pred | hit, theta) (K-Gaussian mixtures)."""
    def __init__(self, width=256, depth=3, n_in=NGEO):
        super().__init__()
        layers, n = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]
            n = width
        self.trunk = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(n, 1 + 6*K)

    def forward(self, g):
        o = self.head(self.trunk(g))
        return o[:, 0], o[:, 1:1 + 3*K], o[:, 1 + 3*K:]


def _mdn_logp(par, x):
    logit_w, mu, log_s = par[:, :K], par[:, K:2*K], par[:, 2*K:].clamp(-4, 4)
    z = (x[:, None] - mu)/torch.exp(log_s)
    comp = -0.5*z*z - log_s - 0.5*LOG2PI
    return torch.logsumexp(torch.log_softmax(logit_w, 1) + comp, 1)


def pmt_logp(net, g, t_pred, hit, lq, t):
    """log p(observation | theta) per PMT; time residual in units of 10 ns (clipped)."""
    h_logit, qpar, tpar = net(g)
    tres = ((t - t_pred).clamp(-100, 400)/10.0).float()
    lp_hit = -torch.nn.functional.softplus(-h_logit)
    lp_unhit = -torch.nn.functional.softplus(h_logit)
    hitf = hit.float()
    return torch.where(hitf > 0.5, lp_hit + _mdn_logp(qpar, lq.float()) + _mdn_logp(tpar, tres), lp_unhit)


# ---- v4: emission-point geometry, time density conditioned on the observed charge, K = 5, ensembles ------
NGEO4, K4 = 14, 5


def geometry4(theta, pmt_pos, pmt_dir, wall):
    """v2 geometry + emission distance, emission->PMT path, cos(track, emission->PMT), incidence from emission."""
    g, t_pred = geometry(theta, pmt_pos, pmt_dir, wall)
    v, d = theta[:, 1:4], theta[:, 4:7]
    w = pmt_pos - v
    R = torch.linalg.norm(w, dim=1).clamp(min=1.0)
    along = (w*d).sum(1)
    perp = torch.sqrt((R*R - along*along).clamp(min=1e-6))
    s = (along - perp/TAN_C).clamp(min=0.0)
    e = pmt_pos - (v + s[:, None]*d)
    path = torch.linalg.norm(e, dim=1).clamp(min=1.0)
    cos_em = (e*d).sum(1)/path
    inc_em = -(e*pmt_dir).sum(1)/path
    return torch.cat([g, torch.stack([s/1e3, path/1e3, cos_em, inc_em], dim=1).float()], dim=1), t_pred


def _mdn_logp_k(par, x, k):
    logit_w, mu, log_s = par[:, :k], par[:, k:2*k], par[:, 2*k:].clamp(-4, 4)
    z = (x[:, None] - mu)/torch.exp(log_s)
    comp = -0.5*z*z - log_s - 0.5*LOG2PI
    return torch.logsumexp(torch.log_softmax(logit_w, 1) + comp, 1)


class DensityNet4(torch.nn.Module):
    """p(hit), p(log q | hit), p(t - t_pred | hit, log q): the time mixture sees the observed charge."""
    def __init__(self, width=256, depth=3, n_in=NGEO4, k=K4):
        super().__init__()
        layers, n = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]
            n = width
        self.k = k
        self.trunk = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(n, 1 + 3*k)
        self.thead = torch.nn.Sequential(torch.nn.Linear(n + 1, 128), torch.nn.SiLU(), torch.nn.Linear(128, 3*k))

    def logp(self, g, t_pred, hit, lq, t):
        z = self.trunk(g)
        o = self.head(z)
        tpar = self.thead(torch.cat([z, lq.float()[:, None]], dim=1))
        tres = ((t - t_pred).clamp(-100, 400)/10.0).float()
        lp_hit = -torch.nn.functional.softplus(-o[:, 0])
        lp_unhit = -torch.nn.functional.softplus(o[:, 0])
        return torch.where(hit.float() > 0.5, lp_hit + _mdn_logp_k(o[:, 1:], lq.float(), self.k)
                           + _mdn_logp_k(tpar, tres, self.k), lp_unhit)


class Ensemble:
    """Mean per-PMT log p over checkpoints (v2 DensityNet or v4 DensityNet4, read from the checkpoint)."""
    def __init__(self, paths, device="cpu"):
        self.nets, self.v4 = [], None
        for p in paths:
            ck = torch.load(p, weights_only=False, map_location=device)
            v4 = ck.get("arch") == "v4"
            net = DensityNet4() if v4 else DensityNet()
            net.load_state_dict(ck["state"]); net.to(device).eval()
            for q in net.parameters():
                q.requires_grad_(False)
            assert self.v4 in (None, v4), "one architecture per ensemble"
            self.v4, self.wall = v4, ck["wall"]
            self.nets.append(net)

    def logp(self, theta, pmt_pos, pmt_dir, hit, lq, t):
        if self.v4:
            g, tp = geometry4(theta, pmt_pos, pmt_dir, self.wall)
            return sum(n.logp(g, tp, hit, lq, t) for n in self.nets)/len(self.nets)
        g, tp = geometry(theta, pmt_pos, pmt_dir, self.wall)
        return sum(pmt_logp(n, g, tp, hit, lq, t) for n in self.nets)/len(self.nets)
