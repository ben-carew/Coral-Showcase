"""The forward model the bound is built on: mean charge and arrival time per PMT.

    mu_j(x) = N_gamma(p) * J_j(vertex) * h(delta_alpha_j | p, R_j)  +  mu_dark

h is the measured light profile (see `build.py`): photoelectrons per emitted photon
per unit acceptance, as a function of the PMT's angle from the ring edge
delta_alpha = alpha - theta_edge(p), the vertex-PMT distance R and the momentum. It
is interpolated trilinearly in (ln p, R, delta_alpha), so mu is continuous in every
parameter and its derivatives are well defined everywhere except on the (measure-zero)
cell boundaries.

The arrival time is Coral's convention (`likelihood._compute_time_residuals`):
emission at the track midpoint, plus the time the particle takes to get there. The
timing table in `build.py` was measured against exactly this convention, so the
derivatives of T and the information per hit are consistent with each other.
"""

from __future__ import annotations

import json

import numpy as np

C_LIGHT = 29.9792458      # cm/ns
MU_DARK = 2.0e-3          # 4.2 kHz dark rate x ~450 ns fit buffer, pe per PMT


class Detector:
    """PMT positions and inward normals, and the tank dimensions."""

    def __init__(self, pos, ori, *, radius, half_height, pmt_radius):
        self.pos = np.asarray(pos, float)
        self.ori = np.asarray(ori, float)
        self.n = len(self.pos)
        self.radius = float(radius)
        self.half_height = float(half_height)
        self.pmt_radius = float(pmt_radius)

    @classmethod
    def from_wcsim_h5(cls, path):
        import h5py
        with h5py.File(path) as f:
            g = f["wcsimGeoT/PMT"][:]
            geo = f["wcsimGeoT/Geometry"][:]
        pos = np.stack([g["Position_x0"], g["Position_x1"], g["Position_x2"]], 1)
        ori = np.stack([g["Orientation_x0"], g["Orientation_x1"], g["Orientation_x2"]], 1)
        return cls(pos, ori, radius=float(geo["WCCylRadius"][0]),
                   half_height=float(np.abs(pos[:, 2]).max()),
                   pmt_radius=float(geo["WCPMTRadius"][0]))

    def to_arrays(self):
        return {"det_pos": self.pos, "det_ori": self.ori,
                "det_dims": np.array([self.radius, self.half_height, self.pmt_radius])}

    @classmethod
    def from_arrays(cls, a):
        r, h, pr = a["det_dims"]
        return cls(a["det_pos"], a["det_ori"], radius=r, half_height=h, pmt_radius=pr)

    def dwall(self, v):
        return min(self.radius - np.hypot(v[0], v[1]), self.half_height - abs(v[2]))


class LightModel:
    """mu_j(vertex, direction, p) and T_j for one particle type."""

    def __init__(self, det, *, h, den, logp_ctr, r_ctr, da_edges, nphot_pars,
                 maxd_pars, mass, n_phase, n_group, atten, eps_pol4,
                 mu_dark=MU_DARK):
        self.det = det
        self.h = _fill_empty_r(np.asarray(h, float), np.asarray(den, float))
        self.logp_ctr = np.asarray(logp_ctr, float)
        self.r_ctr = np.asarray(r_ctr, float)
        self.da_ctr = 0.5 * (da_edges[1:] + da_edges[:-1])
        self.nphot_pars = np.asarray(nphot_pars, float)
        self.maxd_pars = np.asarray(maxd_pars, float)
        self.mass = float(mass)
        self.n_phase, self.n_group, self.atten = float(n_phase), float(n_group), float(atten)
        self.eps_pol4 = np.asarray(eps_pol4, float)
        self.mu_dark = float(mu_dark)

    def n_photons(self, p):
        return np.exp(np.polyval(self.nphot_pars[::-1], np.log(p)))

    def track_length(self, p):
        return np.exp(np.polyval(self.maxd_pars[::-1], np.log(p)))

    def theta_edge(self, p):
        beta = p / np.sqrt(p * p + self.mass ** 2)
        return np.degrees(np.arccos(np.clip(1.0 / (self.n_phase * beta), -1.0, 1.0)))

    def _profile(self, p, R, da):
        """h interpolated in (ln p, R, delta_alpha)."""
        lp = np.log(p)
        k = np.clip(np.searchsorted(self.logp_ctr, lp) - 1, 0, len(self.logp_ctr) - 2)
        f = np.clip((lp - self.logp_ctr[k]) / (self.logp_ctr[k + 1] - self.logp_ctr[k]), 0, 1)
        return ((1 - f) * _bilinear(self.h[k], self.r_ctr[k], self.da_ctr, R, da)
                + f * _bilinear(self.h[k + 1], self.r_ctr[k + 1], self.da_ctr, R, da))

    def charge(self, vertex, direction, p):
        """Mean photoelectrons per PMT (all light, plus dark noise)."""
        r = self.det.pos - vertex
        R = np.sqrt(np.einsum("ij,ij->i", r, r))
        a = self.det.pmt_radius
        omega = 2.0 * np.pi * (1.0 - R / np.sqrt(R * R + a * a))
        eta = np.clip(-np.einsum("ij,ij->i", r, self.det.ori) / R, 0.0, 1.0)
        J = omega * np.polyval(self.eps_pol4[::-1], eta) * np.exp(-R / self.atten)
        alpha = np.degrees(np.arccos(np.clip((r @ direction) / R, -1.0, 1.0)))
        h = self._profile(p, R, alpha - self.theta_edge(p))
        return self.n_photons(p) * J * np.maximum(h, 0.0) + self.mu_dark

    __call__ = charge

    def time(self, t, vertex, direction, p):
        """Coral's predicted arrival time per PMT, ns."""
        half = 0.5 * self.track_length(p)
        mid = vertex + half * direction
        dist = np.sqrt(((self.det.pos - mid) ** 2).sum(axis=1))
        return t + half / C_LIGHT + dist * self.n_group / C_LIGHT

    def delta_alpha(self, vertex, direction, p):
        r = self.det.pos - vertex
        R = np.sqrt(np.einsum("ij,ij->i", r, r))
        alpha = np.degrees(np.arccos(np.clip((r @ direction) / R, -1.0, 1.0)))
        return alpha - self.theta_edge(p)


class TimingTable:
    """Measured Fisher information per digitised hit, I_t(delta_alpha, mu), in ns^-2."""

    def __init__(self, info, da_edges, mu_edges):
        self.info = np.asarray(info, float)
        self.da_edges = np.asarray(da_edges, float)
        self.mu_edges = np.asarray(mu_edges, float)

    def __call__(self, da, mu):
        i = np.clip(np.searchsorted(self.da_edges, da, side="right") - 1, 0, len(self.da_edges) - 2)
        j = np.clip(np.searchsorted(self.mu_edges, mu, side="right") - 1, 0, len(self.mu_edges) - 2)
        return self.info[i, j]


class CRBModel:
    """Everything the Fisher computation needs, loaded from one .npz."""

    def __init__(self, det, light, timing, provenance, floors=None):
        self.det = det
        self.light = light
        self.timing = timing
        self.provenance = provenance
        self.floors = floors or {}          # particle -> floor.Floors (physics floors), if measured

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls.from_arrays({k: a[k] for k in a.files})

    @classmethod
    def from_arrays(cls, a, particles=None, allow_missing_timing=False):
        det = Detector.from_arrays(a)
        n_phase, n_group, atten = a["const"]
        light, timing = {}, {}
        for particle in particles or ("e-", "mu-"):
            tag = particle.replace("-", "")
            if f"{tag}_light_h" not in a:
                continue
            light[particle] = LightModel(
                det, h=a[f"{tag}_light_h"], den=a[f"{tag}_light_den"],
                logp_ctr=a[f"{tag}_light_logp_ctr"], r_ctr=a[f"{tag}_light_r_ctr"],
                da_edges=a["da_edges"], nphot_pars=a[f"{tag}_nphot_pars"],
                maxd_pars=a[f"{tag}_maxd_pars"], mass=float(a[f"{tag}_mass"]),
                n_phase=n_phase, n_group=n_group, atten=atten, eps_pol4=a["eps_pol4"])
            if f"{tag}_timing_info" in a:
                timing[particle] = TimingTable(a[f"{tag}_timing_info"],
                                               a["t_da_edges"], a["t_mu_edges"])
            elif not allow_missing_timing:
                raise KeyError(f"model has no timing table for {particle}")
        prov = json.loads(str(a["provenance"])) if "provenance" in a else {}
        from .floor import Floors
        floors = {pt: f for pt in ("e-", "mu-") if (f := Floors.from_arrays(a, pt.replace("-", ""))) is not None}
        return cls(det, light, timing, prov, floors)


def _bilinear(tab, xc, yc, x, y):
    """Bilinear interpolation of tab[ix, iy] on (monotone) centres xc, yc."""
    ix = np.clip(np.searchsorted(xc, x) - 1, 0, len(xc) - 2)
    iy = np.clip(np.searchsorted(yc, y) - 1, 0, len(yc) - 2)
    fx = np.clip((x - xc[ix]) / (xc[ix + 1] - xc[ix]), 0.0, 1.0)
    fy = np.clip((y - yc[iy]) / (yc[iy + 1] - yc[iy]), 0.0, 1.0)
    return ((1 - fx) * (1 - fy) * tab[ix, iy] + fx * (1 - fy) * tab[ix + 1, iy]
            + (1 - fx) * fy * tab[ix, iy + 1] + fx * fy * tab[ix + 1, iy + 1])


def _fill_empty_r(h, den):
    """Cells no event ever exposed borrow the nearest exposed distance bin."""
    h = h.copy()
    for ip in range(h.shape[0]):
        for ia in range(h.shape[2]):
            seen = den[ip, :, ia] > 0
            if seen.all() or not seen.any():
                continue
            idx = np.where(seen)[0]
            for ir in np.where(~seen)[0]:
                h[ip, ir, ia] = h[ip, idx[np.argmin(np.abs(idx - ir))], ia]
    return h
