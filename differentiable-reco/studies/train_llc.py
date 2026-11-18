"""R39: train the learned light correction (LLC) of one hypothesis by maximum likelihood at
the true track.

Coral's own prefit and sub-eventing run on calibration events; the fit is skipped and each
primary sub-event's hits, out-of-window mask, true track and photon yield are recorded. Every
PMT of the event then enters the differentiable likelihood (likelihood_diff, JAX) at the TRUE
vertex, direction and momentum, with the direct photons multiplied by exp(MLP(features)); the
charge terms (hit probability, charge pdf, unhit) are minimised w.r.t. the MLP's weights by
full-batch Adam. The time term is left out (R28: re-fitting the time pdf at the true track did
not move the momentum core). A fifth of the events is held out to pick the epoch.

usage: train_llc.py <config> <particle> <out.npz> <n_events> <input.root> [input.root ...]
"""
import os, re, sys, tempfile, time
import numpy as np

cfg_path, particle, out_path, n_events = sys.argv[1:5]
inputs = sys.argv[5:]
EPOCHS = int(os.environ.get("LLC_EPOCHS", 150))
HIDDEN = int(os.environ.get("LLC_HIDDEN", 16))
LR = float(os.environ.get("LLC_LR", 3e-3))
L2 = float(os.environ.get("LLC_L2", 1e-3))
# LLC_USE: which inputs the MLP may see (1/0 per likelihood_diff.LLC_FEATURES); the others'
# first-layer weights are zero and stay zero
USE = np.array([float(v) for v in os.environ.get("LLC_USE", "1,1,1,1,1,1").split(",")])
# LLC_TIME=1: train on the full likelihood (time term included); each event's vertex time is
# profiled at the true geometry with the uncorrected model (the truth carries t = 0, not Coral's frame)
TIME = os.environ.get("LLC_TIME", "0") == "1"
PDG = {"e-": 11, "mu-": 13}

# a copy of the config that reads the MC truth (the true track) and fits only this hypothesis
text = open(cfg_path).read()
if re.search(r"^save_truth\s*=", text, re.M):
    text = re.sub(r"^save_truth\s*=.*$", "save_truth = True  ; True | False", text, flags=re.M)
else:
    text = text.replace("[in_out]\n", "[in_out]\nsave_truth = True  ; True | False\n", 1)
tmp_cfg = os.path.join(tempfile.mkdtemp(), "train_llc.ini")
open(tmp_cfg, "w").write(text)

import fitqun.globals as gbl
import fitqun.fitqun as fq

recs = []


def capture(manager, sub_event_hits, base_seed, rivals, mask_tubeids, unhit_orders, bounds,
            event, i_subevent, x_true, pdg_true, dump=True):
    if i_subevent == 0 and x_true is not None and abs(int(pdg_true)) == PDG[particle]:
        recs.append(dict(hits=sub_event_hits.copy(), mask=mask_tubeids, x=np.array(x_true, dtype=float),
                         event=event, qe=gbl.qe * manager.qe_corrections.get(particle, 1.0),
                         t_seed=float(base_seed["t"])))
    return {p: (False, np.inf, np.full(7, np.nan)) for p in manager.particles}, 0.0


fq.fit_single_ring = capture
sys.argv = ["fitqun", tmp_cfg, "-i", *inputs, "-o", tempfile.mkdtemp(), "-n", n_events]
t0 = time.time()
fq.main()
print(f"capture: {len(recs)} {particle} events in {time.time() - t0:.0f} s", flush=True)

import jax
import jax.numpy as jnp
import fitqun_tuning.corrections as corrections
from fitqun_tuning.cherenkov_profile import cprofs
from fitqun_tuning.time_pdf import tpdfs
from fitqun_fit.likelihood import select_unhit_subset
from fitqun_fit import likelihood_diff as L

B = L.backend("jax")
print("devices:", jax.devices(), flush=True)
cprof, tpdf = cprofs[particle], tpdfs[particle]


def features(x, pmts):
    """The LLC features at track x for a PMT table (numpy, as likelihood_diff._light)."""
    pos = pmts[["x", "y", "z"]].to_numpy(float)
    ori = pmts[["ox", "oy", "oz"]].to_numpy(float)
    st = np.sin(x[4])
    d = np.array([st * np.cos(x[5]), st * np.sin(x[5]), np.cos(x[4])])
    r0 = x[1:4] - pos
    m = np.linalg.norm(r0, axis=1)
    th0 = -(r0 @ d) / m
    eta0 = np.einsum("ij,ij->i", r0, ori) / m
    logp = np.log(max(x[6] - getattr(cprof, "poffset", 0.0), 1.0))
    R = corrections.wall_radius if corrections.wall_radius is not None else gbl.radius
    Hh = corrections.wall_half_length if corrections.wall_half_length is not None else gbl.length / 2
    dwall = corrections.dwall(x, R, Hh)
    tw = corrections._towall(x[1], x[2], x[3], d[0], d[1], d[2], R, Hh)
    n = len(m)
    return np.stack([th0, np.log10(m / 1000.0), eta0, np.full(n, logp), np.full(n, dwall / 1000.0),
                     np.full(n, tw / 1000.0)])


# feature normalisation from the first events' PMTs
sample = [features(r["x"], select_unhit_subset(r["hits"], 1.0, r["mask"])[0]) for r in recs[:50]]
F = np.concatenate(sample, axis=1)
mu, sd = F.mean(axis=1), F.std(axis=1) + 1e-6
rng = np.random.default_rng(39)
nf = len(L.LLC_FEATURES)
theta0 = dict(W1=rng.normal(0, 1 / np.sqrt(nf), (nf, HIDDEN)) * USE[:, None], b1=np.zeros(HIDDEN),
              W2=rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, HIDDEN)), b2=np.zeros(HIDDEN),
              w3=np.zeros(HIDDEN), b3=np.zeros(()), mu=mu, sd=sd)

# every event's full-PMT likelihood at its true track, charge terms only; a true momentum
# outside the profile's tuning range (e.g. a muon below its tuned threshold) is left out
p_lo, p_hi = cprof.momentum(cprof.logpbins[0]), cprof.momentum(cprof.logpbins[-1])
n_all = len(recs)
recs = [r for r in recs if p_lo <= r["x"][6] <= p_hi]
print(f"{len(recs)} / {n_all} events with the true momentum inside [{p_lo:.0f}, {p_hi:.0f}] MeV/c", flush=True)
items = []
for r in recs:
    pmts_all = select_unhit_subset(r["hits"], 1.0, r["mask"])[0]
    nll = L.nll_diff("jax", pmts_all, r["hits"], cprof, tpdf, 1.0, qe=r["qe"],
                     geom_curves=corrections.geom_light.get(particle), light_correction=theta0)
    if TIME:
        from scipy.optimize import minimize_scalar
        x = np.array(r["x"], dtype=float)

        def f_t(t):
            x[0] = t
            return nll.parts.value(x)
        res = minimize_scalar(f_t, bounds=(r["t_seed"] - 30.0, r["t_seed"] + 30.0), method="bounded",
                              options=dict(xatol=0.01))
        r["x"] = np.array(r["x"], dtype=float)
        r["x"][0] = res.x
        parts = list(nll.parts.parts)
    else:
        parts = [(k, P, M, S, c._replace(use_tpdf=False)) for k, P, M, S, c in nll.parts.parts]
    items.append((B.asarray(r["x"]), parts))
# and any event whose likelihood at the truth is not finite
def _finite(it):
    X, parts = it
    return np.isfinite(sum(float(val_f0(X, P, M, S, c, k)) for k, P, M, S, c in parts))


val_f0 = jax.jit(lambda X, P, M, S, cfg, kind: L._FUNCS[kind](B, X, P, M, S, cfg), static_argnames=("cfg", "kind"))
n_items = len(items)
items = [it for it in items if _finite(it)]
print(f"{len(items)} / {n_items} events with a finite likelihood at the truth", flush=True)
n_val = max(1, len(items) // 5)
val, train = items[:n_val], items[n_val:]
if TIME:
    dt = np.array([r["x"][0] - r["t_seed"] for r in recs])
    print(f"time profiled at the true geometry: t - prefit seed median {np.median(dt):+.2f} ns, spread (16-84%) "
          f"{np.percentile(dt, 16):+.2f} .. {np.percentile(dt, 84):+.2f} ns", flush=True)
print(f"train {len(train)} / validation {len(val)} events, {sum(len(p[1].get('mask')) for _, ps in items for p in ps)} padded PMT terms", flush=True)

FIT = ("W1", "b1", "W2", "b2", "w3", "b3")


def part_loss(th, X, P, M, S, cfg, kind):
    llc = dict(th, mu=M["llc"]["mu"], sd=M["llc"]["sd"])
    return L._FUNCS[kind](B, X, P, dict(M, llc=llc), S, cfg)


vg = jax.jit(jax.value_and_grad(part_loss), static_argnames=("cfg", "kind"))
val_f = jax.jit(part_loss, static_argnames=("cfg", "kind"))


def total(th, its, grad=True):
    v, g = 0.0, None
    for X, parts in its:
        for k, P, M, S, c in parts:
            if grad:
                a, b = vg(th, X, P, M, S, c, k)
                v = v + a
                g = b if g is None else jax.tree_util.tree_map(jnp.add, g, b)
            else:
                v = v + val_f(th, X, P, M, S, c, k)
    return v, g


th = {k: jnp.asarray(theta0[k]) for k in FIT}
m_ = jax.tree_util.tree_map(jnp.zeros_like, th)
v_ = jax.tree_util.tree_map(jnp.zeros_like, th)
v0_train = float(total(th, train, False)[0]) / len(train)
v0_val = float(total(th, val, False)[0]) / len(val)
print(f"start: nLL/event train {v0_train:.3f} validation {v0_val:.3f}", flush=True)
best = (v0_val, {k: np.asarray(v) for k, v in th.items()}, 0)
b1, b2, eps = 0.9, 0.999, 1e-8
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    v, g = total(th, train)
    g = jax.tree_util.tree_map(lambda a, w: a / len(train) + L2 * w, g, th)
    m_ = jax.tree_util.tree_map(lambda m, a: b1 * m + (1 - b1) * a, m_, g)
    v_ = jax.tree_util.tree_map(lambda s, a: b2 * s + (1 - b2) * a * a, v_, g)
    lr = LR * min(1.0, ep / 10)
    th = jax.tree_util.tree_map(lambda w, m, s: w - lr * (m / (1 - b1 ** ep)) / (jnp.sqrt(s / (1 - b2 ** ep)) + eps),
                                th, m_, v_)
    th = dict(th, W1=th["W1"] * jnp.asarray(USE)[:, None])      # unused inputs stay off
    if ep % 5 == 0 or ep == EPOCHS:
        vv = float(total(th, val, False)[0]) / len(val)
        if vv < best[0]:
            best = (vv, {k: np.asarray(x) for k, x in th.items()}, ep)
        print(f"epoch {ep}: train {float(v) / len(train):.3f} (d {float(v) / len(train) - v0_train:+.3f}) "
              f"validation {vv:.3f} (d {vv - v0_val:+.3f}) best epoch {best[2]}  [{time.time() - t0:.0f} s]", flush=True)

vbest, thb, ep_best = best
np.savez(out_path, **thb, mu=mu, sd=sd, features=np.array(L.LLC_FEATURES), use=USE, time_term=TIME, particle=particle,
         n_train=len(train), n_val=len(val), val_gain=v0_val - vbest, epoch=ep_best)
print(f"saved {out_path}: validation nLL/event {v0_val:.3f} -> {vbest:.3f} ({v0_val - vbest:+.3f}) at epoch {ep_best}")
