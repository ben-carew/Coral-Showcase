"""R39 v4: train the learned light correction THROUGH THE FIT.

For each calibration event the correction theta changes where the fit lands: X*(theta) =
argmin_X L(X; theta). Each epoch every event's minimum is tracked by damped Newton steps (exact
gradient and Hessian, from the previous epoch's point; the first epoch starts from the
production fit), and theta follows the gradient of a Huber loss on the fitted residuals:

    l(X*) = h((p*/p_true - 1)/0.035) + h(along/25 cm) + h(trans1/8 cm) + h(trans2/8 cm) + sum_k h((d* - d_true)_k/0.04)

(the vertex after the production vertex-bias shift, held constant per epoch). By the implicit
function theorem dX*/dtheta = -H^-1 d(grad_X L)/dtheta, so dl/dtheta = -d/dtheta [v . grad_X L(X*; theta)]
with v = H^-1 dl/dX* fixed: one mixed-derivative pass per event. The epoch is chosen on held-out
calibration events. Fits of the true hypothesis only, sub-event 0, converged (< 20 deg).

usage: train_llc_fit.py <config> <particle> <out.npz> <n_events> <fit_stages glob> <input.root> [input.root ...]
"""
import glob, os, re, sys, tempfile, time
import numpy as np
import pandas as pd

cfg_path, particle, out_path, n_events, stages_glob = sys.argv[1:6]
inputs = sys.argv[6:]
EPOCHS = int(os.environ.get("LLC_EPOCHS", 80))
HIDDEN = int(os.environ.get("LLC_HIDDEN", 16))
LR = float(os.environ.get("LLC_LR", 1e-3))
L2 = float(os.environ.get("LLC_L2", 1e-3))
NEWTON = int(os.environ.get("LLC_NEWTON", 2))
# LLC_KIND=ltw: learn per-PMT weights of the charge and time terms (likelihood_diff LTW_FEATURES)
# instead of the light correction
LTW = os.environ.get("LLC_KIND", "llc") == "ltw"
# LLC_PLOSS=binned: the momentum term reads p*/p_true divided by (1 + its median - 1) in the event's true-p sextile
# (training events, refreshed each epoch, no gradient): the correction cannot gain by flattening the momentum bias,
# nor (a ratio, not a difference) by shrinking the momentum scale
PBINNED = os.environ.get("LLC_PLOSS", "raw") == "binned"
USE = np.array([float(v) for v in os.environ.get("LLC_USE", "1,1,1,1" if LTW else "1,1,1,1,1,1").split(",")])
PDG = {"e-": 11, "mu-": 13}

text = open(cfg_path).read()
if re.search(r"^save_truth\s*=", text, re.M):
    text = re.sub(r"^save_truth\s*=.*$", "save_truth = True  ; True | False", text, flags=re.M)
else:
    text = text.replace("[in_out]\n", "[in_out]\nsave_truth = True  ; True | False\n", 1)
tmp_cfg = os.path.join(tempfile.mkdtemp(), "train_llc_fit.ini")
open(tmp_cfg, "w").write(text)

import fitqun.globals as gbl
import fitqun.fitqun as fq

recs = []


def capture(manager, sub_event_hits, base_seed, rivals, mask_tubeids, unhit_orders, bounds,
            event, i_subevent, x_true, pdg_true, dump=True):
    if i_subevent == 0 and x_true is not None and abs(int(pdg_true)) == PDG[particle]:
        recs.append(dict(hits=sub_event_hits.copy(), mask=mask_tubeids, x=np.array(x_true, dtype=float),
                         event=int(event), file=int(manager.file_index()),
                         qe=gbl.qe * manager.qe_corrections.get(particle, 1.0)))
    return {p: (False, np.inf, np.full(7, np.nan)) for p in manager.particles}, 0.0


fq.fit_single_ring = capture
sys.argv = ["fitqun", tmp_cfg, "-i", *inputs, "-o", tempfile.mkdtemp(), "-n", n_events]
t0 = time.time()
fq.main()
print(f"capture: {len(recs)} {particle} events in {time.time() - t0:.0f} s", flush=True)

# the production (calibration-run) fits: winning fit's last step, raw (before the vertex-bias shift).
# A job's events are numbered within its input file: key by (file, event), the file index read
# from the job's output ROOT name as FQManager.file_index does
frames = []
for f in sorted(glob.glob(stages_glob)):
    out = glob.glob(os.path.join(os.path.dirname(f), "*_rwcs.root"))
    m = re.search(r"-(\d+)_rwcs", os.path.basename(out[0])) if out else None
    if m is None:
        continue
    d = pd.read_csv(f)
    d["file"] = int(m.group(1))
    frames.append(d)
fs = pd.concat(frames, ignore_index=True)
fs = fs[(fs.particle == particle) & (fs.sub_event == 0)]
K = ["file", "event", "sub_event", "particle"]
last = fs.loc[fs.groupby(K + ["fit"]).stage.idxmax()]
win = last.loc[last.groupby(K).nll.idxmin()].set_index(["file", "event"])
print(f"calibration fits: {len(win)} {particle} events", flush=True)

import jax
import jax.numpy as jnp
import fitqun_tuning.corrections as corrections
from fitqun_tuning.cherenkov_profile import cprofs
from fitqun_tuning.time_pdf import tpdfs
from fitqun_fit.likelihood import select_unhit_subset
from fitqun_fit import likelihood_diff as L

B = L.backend("jax")
cprof, tpdf = cprofs[particle], tpdfs[particle]
print("devices:", jax.devices(), "smooth model:", gbl.cprof_r0_interp, gbl.angular_interp, flush=True)


def unit(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def features(x, pmts):
    pos = pmts[["x", "y", "z"]].to_numpy(float)
    ori = pmts[["ox", "oy", "oz"]].to_numpy(float)
    d = unit(x[4], x[5])
    r0 = x[1:4] - pos
    m = np.linalg.norm(r0, axis=1)
    R = corrections.wall_radius if corrections.wall_radius is not None else gbl.radius
    Hh = corrections.wall_half_length if corrections.wall_half_length is not None else gbl.length / 2
    n = len(m)
    return np.stack([-(r0 @ d) / m, np.log10(m / 1000.0), np.einsum("ij,ij->i", r0, ori) / m,
                     np.full(n, np.log(max(x[6] - getattr(cprof, "poffset", 0.0), 1.0))),
                     np.full(n, corrections.dwall(x, R, Hh) / 1000.0),
                     np.full(n, corrections._towall(x[1], x[2], x[3], *d, R, Hh) / 1000.0)])


# events: true track, production fit converged
items = []
for r in recs:
    if (r["file"], r["event"]) not in win.index:
        continue
    w = win.loc[(r["file"], r["event"])]
    xs = w[["t", "x", "y", "z", "theta", "phi", "p"]].to_numpy(float)
    u = unit(r["x"][4], r["x"][5])
    if np.degrees(np.arccos(np.clip(unit(xs[4], xs[5]) @ u, -1, 1))) > 20:
        continue
    r["xstar"] = xs
    items.append(r)
print(f"{len(items)} converged events with truth and a calibration fit", flush=True)
sample = [features(r["x"], select_unhit_subset(r["hits"], 1.0, r["mask"])[0]) for r in items[:50]]
F = np.concatenate(sample, axis=1)
mu, sd = F.mean(axis=1), F.std(axis=1) + 1e-6
rng = np.random.default_rng(39)
FEATURES = L.LTW_FEATURES if LTW else L.LLC_FEATURES
NAME = "ltw" if LTW else "llc"
if LTW:
    # the geometry features as the light correction's; log10 mu centred at 0.1 p.e., unit width
    mu, sd = np.append(mu[:3], -1.0), np.append(sd[:3], 1.0)
nf, nout = len(FEATURES), (2 if LTW else 1)
theta0 = dict(W1=rng.normal(0, 1 / np.sqrt(nf), (nf, HIDDEN)) * USE[:, None], b1=np.zeros(HIDDEN),
              W2=rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, HIDDEN)), b2=np.zeros(HIDDEN),
              w3=np.zeros((HIDDEN, 2)) if LTW else np.zeros(HIDDEN), b3=np.zeros(2) if LTW else np.zeros(()),
              mu=mu, sd=sd, kind="term_weights" if LTW else "light")
W3I = (0, 1) if LTW else (0,)          # the gradient check's weight (LTW: the time output)
FIT = ("W1", "b1", "W2", "b2", "w3", "b3")

ev = []
for r in items:
    pmts_all = select_unhit_subset(r["hits"], 1.0, r["mask"])[0]
    nll = L.nll_diff("jax", pmts_all, r["hits"], cprof, tpdf, 1.0, qe=r["qe"],
                     geom_curves=corrections.geom_light.get(particle), light_correction=theta0)
    parts = nll.parts.parts
    structure = tuple((k, c) for k, P, M, S, c in parts)
    datas = tuple((P, M, S) for k, P, M, S, c in parts)
    u = unit(r["x"][4], r["x"][5])
    e1 = np.cross(u, [0.0, 0.0, 1.0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0, 0])
    e2 = np.cross(u, e1)
    ev.append(dict(structure=structure, datas=datas, X=np.array(r["xstar"]), xt=r["x"], u=u, e1=e1, e2=e2,
                   event=r["event"]))
n_val = max(1, len(ev) // 5)
val, train = ev[:n_val], ev[n_val:]
print(f"train {len(train)} / validation {len(val)} events", flush=True)


def total(X, th, datas, structure):
    tot = 0.0
    for (kind, cfg), (P, M, S) in zip(structure, datas):
        net = dict(th, mu=M[NAME]["mu"], sd=M[NAME]["sd"])
        tot = tot + L._FUNCS[kind](B, X, P, dict(M, **{NAME: net}), S, cfg)
    return tot


f_val = jax.jit(total, static_argnames="structure")
f_vg = jax.jit(jax.value_and_grad(total, argnums=0), static_argnames="structure")
f_h = jax.jit(jax.hessian(total, argnums=0), static_argnames="structure")
f_mix = jax.jit(jax.grad(lambda th, X, v, datas, structure: jnp.dot(v, jax.grad(total, argnums=0)(X, th, datas, structure)),
                         argnums=0), static_argnames="structure")
SC = np.array([1.0, 10.0, 10.0, 10.0, 0.01, 0.01, 0.0])         # Newton scaling (p: 1% of p)


def huber(z, d=3.0):
    a = jnp.abs(z)
    return jnp.where(a <= d, 0.5 * z * z, d * (a - 0.5 * d))


def loss_fn(X, xt, u, e1, e2, shift, pb):
    v = X[1:4] + shift - xt[1:4]
    d = jnp.stack([jnp.sin(X[4]) * jnp.cos(X[5]), jnp.sin(X[4]) * jnp.sin(X[5]), jnp.cos(X[4])])
    terms = [huber((X[6] / xt[6] / (1.0 + pb) - 1.0) / 0.035), huber(jnp.dot(v, u) / 25.0), huber(jnp.dot(v, e1) / 8.0),
             huber(jnp.dot(v, e2) / 8.0)] + [huber((d[k] - u[k]) / 0.04) for k in range(3)]
    return sum(terms)


f_loss = jax.jit(jax.value_and_grad(loss_fn))


def shift_of(X):
    return corrections.correct_vertex_bias(particle, X)[1:4] - X[1:4]


def refit(e, th):
    """NEWTON damped steps from e['X'] with the current theta; returns (H at the final point, ok)."""
    X = e["X"].copy()
    a = B.asarray
    for _ in range(NEWTON):
        v, g = f_vg(a(X), th, e["datas"], e["structure"])
        H = np.asarray(f_h(a(X), th, e["datas"], e["structure"]))
        v, g = float(v), np.asarray(g)
        sc = SC.copy(); sc[6] = 0.01 * X[6]
        Hs, gs = sc[:, None] * H * sc[None, :], sc * g
        lam0 = 1e-3 * max(np.mean(np.abs(np.diag(Hs))), 1e-6)
        moved = False
        for lam in (lam0, 10 * lam0, 100 * lam0, 1000 * lam0):
            try:
                du = -np.linalg.solve(Hs + lam * np.eye(7), gs)
            except np.linalg.LinAlgError:
                continue
            Xn = X + sc * du
            if float(f_val(a(Xn), th, e["datas"], e["structure"])) < v:
                X, moved = Xn, True
                break
        if not moved:
            break
    H = np.asarray(f_h(a(X), th, e["datas"], e["structure"]))
    e["X"] = X
    if not np.all(np.isfinite(H)):
        return H, False
    try:
        return H, bool(np.all(np.linalg.eigvalsh(0.5 * (H + H.T)) > 0))
    except np.linalg.LinAlgError:
        return H, False


def epoch(events, th, grad=True):
    tot, n, g_acc, n_bad = 0.0, 0, None, 0
    a = B.asarray
    for e in events:
        H, ok = refit(e, th)
        if not np.all(np.isfinite(e["X"])):
            continue
        shift = shift_of(e["X"])
        l, dl = f_loss(a(e["X"]), a(e["xt"]), a(e["u"]), a(e["e1"]), a(e["e2"]), a(shift), a(e.get("pb", 0.0)))
        tot += float(l); n += 1
        if not grad:
            continue
        if not ok:
            n_bad += 1
            continue
        v = np.linalg.solve(0.5 * (H + H.T), np.asarray(dl))
        gth = f_mix(th, a(e["X"]), a(v), e["datas"], e["structure"])
        gth = jax.tree_util.tree_map(lambda q: -q, gth)
        g_acc = gth if g_acc is None else jax.tree_util.tree_map(jnp.add, g_acc, gth)
    return tot / max(n, 1), g_acc, n_bad


def set_bias(events):
    """pb of every event: the training events' median p*/p_true - 1 in its true-p sextile (LLC_PLOSS=binned)."""
    if not PBINNED:
        return
    pt = np.array([e["xt"][6] for e in train])
    edges = np.percentile(pt, np.linspace(0, 100, 7)); edges[0], edges[-1] = -np.inf, np.inf
    dp = np.array([e["X"][6] / e["xt"][6] - 1 for e in train])
    med = [np.median(dp[(pt >= lo) & (pt < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
    for e in events:
        e["pb"] = float(med[np.searchsorted(edges, e["xt"][6], side="right") - 1])


def residuals(events):
    """momentum residual (scale-removed iqr68 and core) and transverse vertex p68 at the tracked minima."""
    dp = np.array([e["X"][6] / e["xt"][6] - 1 for e in events])
    tr = np.array([np.linalg.norm(np.cross(e["X"][1:4] + shift_of(e["X"]) - e["xt"][1:4], e["u"])) for e in events])
    lo, hi = np.percentile(dp, [16, 84])
    return 100 * (hi - lo) / 2 / (1 + np.median(dp)), np.percentile(tr, 68)


th = {k: jnp.asarray(theta0[k]) for k in FIT}
# converge every event's minimum of the full-PMT likelihood before training (up to 12 Newton steps)
t1 = time.time()
for e in ev:
    for _ in range(6):
        refit(e, th)
print(f"initial minima converged in {time.time() - t1:.0f} s", flush=True)
if os.environ.get("LLC_CHECK") == "1":
    # implicit gradient vs finite differences of the re-minimised loss, a few events, one weight
    import copy
    a_ = B.asarray
    for e0 in train[:4]:
        e = copy.deepcopy({k: v for k, v in e0.items() if k != "datas"}); e["datas"] = e0["datas"]
        for _ in range(4):
            refit(e, th)
        H, ok = refit(e, th)
        l, dl = f_loss(a_(e["X"]), a_(e["xt"]), a_(e["u"]), a_(e["e1"]), a_(e["e2"]), a_(shift_of(e["X"])), a_(e.get("pb", 0.0)))
        v = np.linalg.solve(0.5 * (H + H.T), np.asarray(dl))
        gi = -float(np.asarray(f_mix(th, a_(e["X"]), a_(v), e["datas"], e["structure"])["w3"])[W3I])
        fd = []
        for sgn in (+1, -1):
            th2 = dict(th, w3=th["w3"].at[W3I].add(sgn * 1e-2))
            e2 = dict(e, X=e["X"].copy())
            for _ in range(10):
                refit(e2, th2)
            l2, _ = f_loss(a_(e2["X"]), a_(e2["xt"]), a_(e2["u"]), a_(e2["e1"]), a_(e2["e2"]), a_(shift_of(e["X"])), a_(e.get("pb", 0.0)))
            fd.append(float(l2))
        print(f"gradient check event {e['event']}: implicit {gi:+.5e}  finite difference {(fd[0] - fd[1]) / 2e-2:+.5e}  (PD {ok})", flush=True)
m_ = jax.tree_util.tree_map(jnp.zeros_like, th)
s_ = jax.tree_util.tree_map(jnp.zeros_like, th)
set_bias(ev)
l0_val, _, _ = epoch(val, th, grad=False)
r0 = residuals(val)
print(f"start: validation loss {l0_val:.4f}; p iqr68 {r0[0]:.3f}% trans p68 {r0[1]:.2f} cm", flush=True)
best = (l0_val, {k: np.asarray(v) for k, v in th.items()}, 0)
b1, b2, eps = 0.9, 0.999, 1e-8
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    set_bias(ev)
    lt, g, nb = epoch(train, th)
    g = jax.tree_util.tree_map(lambda a_, w: a_ / len(train) + L2 * w, g, th)
    m_ = jax.tree_util.tree_map(lambda m, a_: b1 * m + (1 - b1) * a_, m_, g)
    s_ = jax.tree_util.tree_map(lambda s, a_: b2 * s + (1 - b2) * a_ * a_, s_, g)
    lr = LR * min(1.0, ep / 5)
    th = jax.tree_util.tree_map(lambda w, m, s: w - lr * (m / (1 - b1 ** ep)) / (jnp.sqrt(s / (1 - b2 ** ep)) + eps), th, m_, s_)
    th = dict(th, W1=th["W1"] * jnp.asarray(USE)[:, None])
    if ep % 2 == 0 or ep == EPOCHS:
        set_bias(ev)
        lv, _, _ = epoch(val, th, grad=False)
        rv = residuals(val)
        if lv < best[0]:
            best = (lv, {k: np.asarray(x) for k, x in th.items()}, ep)
        print(f"epoch {ep}: train loss {lt:.4f} (non-PD {nb}) validation {lv:.4f} (start {l0_val:.4f}); "
              f"val p iqr68 {rv[0]:.3f}% trans p68 {rv[1]:.2f} cm; best epoch {best[2]}  [{time.time() - t0:.0f} s]", flush=True)

lb, thb, epb = best
np.savez(out_path, **thb, mu=mu, sd=sd, features=np.array(FEATURES), use=USE, p_loss="binned" if PBINNED else "raw",
         kind="term_weights" if LTW else "light", particle=particle,
         trained="through the fit", n_train=len(train), n_val=len(val), val_loss0=l0_val, val_loss=lb, epoch=epb)
print(f"saved {out_path}: validation loss {l0_val:.4f} -> {lb:.4f} at epoch {epb}")
