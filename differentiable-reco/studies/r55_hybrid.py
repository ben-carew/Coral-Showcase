"""R55: offline minimiser study on the real multi-ring / pi0 fit problems.

A Coral run (pi0 or multiring preset, JAX) records every pi0 two-ring fit and every multi-ring
simultaneous fit (likelihood, seed, bounds, free entries) while it runs with its configured
minimiser; then every recorded fit with >= 8 free entries is re-solved by the tight simplex,
L-BFGS and the hybrid (loose simplex stopped at an nll spread F, then L-BFGS) for each F, all
through the production fit functions. Per variant: final nll vs the tight simplex, evaluations,
wall time (the first 3 fits of each variant, where JAX compiles, are left out of the time).

usage: r55_hybrid.py <config> <pi0|mr> <n_events> <out.csv> <input.root> [F,F,...]
"""
import os, re, sys, tempfile, time
import numpy as np
import pandas as pd

cfg_path, kind, n_events, out = sys.argv[1:5]
infile = sys.argv[5]
FS = [float(v) for v in (sys.argv[6] if len(sys.argv) > 6 else "0.3,1,3,10").split(",")]

text = open(cfg_path).read()
smooth = "" if os.environ.get("R55_FLAT") == "1" else "cprof_r0_interp = True\nangular_interp = True\n"
text = text.replace("[likelihood]\n", "[likelihood]\nbackend = jax\ndevice_minimiser = True\n"
                    + smooth + "multi_ring_minimiser = simplex\n", 1)
if os.environ.get("R55_ORDER_WIDTH"):
    text = text.replace("[multiring]\n", f"[multiring]\ntime_order_width = {os.environ['R55_ORDER_WIDTH']}\n", 1)
tmp_cfg = os.path.join(tempfile.mkdtemp(), "r55.ini")
open(tmp_cfg, "w").write(text)

import fitqun.globals as gbl
import fitqun_fit.pi0_fit as pf
import fitqun_fit.multi_ring_fit as mrf

problems = []
_two, _simp = pf.two_ring_fit, mrf._simplex_fit


def rec_two(nll, seed, bounds12, mass_constraint, free_idx, simplex_scale=1.0):
    r = _two(nll, seed, bounds12, mass_constraint, free_idx, simplex_scale)
    if len([i for i in free_idx if not (mass_constraint and i == pf.IP2)]) >= 8:
        problems.append(("pi0", (nll, np.array(seed), bounds12, mass_constraint, list(free_idx), simplex_scale)))
    return r


def rec_simp(nll, x_seed, bounds, free_idx):
    r = _simp(nll, x_seed, bounds, free_idx)
    if len(free_idx) >= 8:
        problems.append(("mr", (nll, np.array(x_seed), bounds, list(free_idx))))
    return r


pf.two_ring_fit, mrf._simplex_fit = rec_two, rec_simp
import fitqun.fitqun as fq
sys.argv = ["fitqun", tmp_cfg, "-i", infile, "-o", tempfile.mkdtemp(), "-n", n_events]
t0 = time.time()
fq.main()
pf.two_ring_fit, mrf._simplex_fit = _two, _simp
gbl.multi_ring_polish = os.environ.get("R55_POLISH", "lbfgs")
gbl.multi_ring_precondition = os.environ.get("R55_PRECOND", "none")
print(f"run: {len(problems)} fits with >= 8 free entries in {time.time() - t0:.0f} s", flush=True)

# evaluations: wrap the likelihood calls the fit functions make
variants = [("simplex", None)] + ([("lbfgs", None)] if os.environ.get("R55_LBFGS", "1") == "1" else []) \
    + [("hybrid", F) for F in FS]
XS = {}
# R55_WARM=1: one untimed pass of the first hybrid first, so no variant's time holds JAX compilation
if os.environ.get("R55_WARM") == "1":
    gbl.multi_ring_minimiser, gbl.multi_ring_coarse_fatol = "hybrid", FS[0]
    for typ, args in problems:
        (_two(*args) if typ == "pi0" else _simp(*args))
    print("warm-up pass done", flush=True)
rows = []
for name, F in variants:
    gbl.multi_ring_minimiser = name
    if F is not None:
        gbl.multi_ring_coarse_fatol = F
    tag = name if F is None else f"hybrid{F:g}"
    for k, (typ, args) in enumerate(problems):
        t1 = time.time()
        if typ == "pi0":
            nll, seed, b, mc, free, sc = args
            ok, fun, x = _two(nll, seed, b, mc, free, sc)
            nfree = len([i for i in free if not (mc and i == pf.IP2)])
        else:
            nll, xs, b, free = args
            ok, fun, x = _simp(nll, xs, b, free)
            nfree = len(free)
        rows.append(dict(variant=tag, k=k, typ=typ, nfree=nfree, ok=ok, nll=fun, sec=time.time() - t1))
        XS[(tag, k)] = np.array(x)
    d = pd.DataFrame([r for r in rows if r["variant"] == tag])
    print(f"{tag:12s} done: {d.sec.iloc[3:].sum():.1f} s steady", flush=True)

D = pd.DataFrame(rows)
D.to_csv(out, index=False)
ref = D[D.variant == "simplex"].set_index("k")
print(f"\n{'variant':12s} {'steady s':>9s} {'speed-up':>8s} {'worse>1':>8s} {'better>1':>9s} {'worse>10':>9s} {'median gap':>10s}")
for tag, d in D.groupby("variant", sort=False):
    d = d.set_index("k")
    gap = d.nll - ref.nll
    st = d.sec.iloc[3:].sum()
    print(f"{tag:12s} {st:9.1f} {ref.sec.iloc[3:].sum() / st:8.2f} {100 * np.mean(gap > 1):7.1f}% {100 * np.mean(gap < -1):8.1f}% "
          f"{100 * np.mean(gap > 10):8.1f}% {gap.median():+10.2f}")

# R55_DIAG=1: for every fit where the first hybrid ends > 1 above the tight simplex, the nll on the straight
# segment from the hybrid point xh to the simplex point xs (11 points) and the gradient's prediction g(xh).(xs - xh)
# of its initial slope: a barrier on the segment = separate basins; a monotone fall with g.d ~ 0 = a gradient
# that does not describe the function (a staircase)
if os.environ.get("R55_DIAG") == "1":
    tag = f"hybrid{FS[0]:g}"
    print(f"\nsegment diagnostic, {tag} vs simplex (fits worse by > 1):")
    for k, (typ, args) in enumerate(problems):
        nll = args[0]
        xh, xs = XS[(tag, k)], XS[("simplex", k)]
        if typ == "pi0":
            f = lambda x: float(nll(*x)); vg = lambda x: nll.value_and_grad(*x)
        else:
            f = lambda x: float(nll(x)); vg = nll.value_and_grad
        fh, fs = f(xh), f(xs)
        if fh - fs <= 1:
            continue
        d = xs - xh
        prof = [f(xh + a * d) for a in np.linspace(0, 1, 11)]
        g = np.asarray(vg(xh)[1])
        fd = (f(xh + 1e-3 * d) - fh) / 1e-3
        if typ == "mr" and os.environ.get("R55_KINK") == "1":
            # which term creases? scan the free entry with the largest |gradient| (simplex-step units) over
            # +-0.05 steps around xh: hit part with time, hit part without time, unhit part
            import jax
            from fitqun_fit import likelihood_diff as LD
            B = LD.backend("jax")
            free = args[3]
            st = mrf._step_sizes(args[1], free)
            gu = np.asarray(nll.value_and_grad(xh)[1])[free] * st
            order = np.argsort(-np.abs(gu))[:2]
            terms = []
            for kd, P, Ms, S, rc in nll.device_parts:
                terms.append((kd, jax.jit(lambda X, P=P, Ms=Ms, S=S, rc=rc, kd=kd: LD._FUNCS[kd](B, X, P, Ms, S, rc))))
                if kd == "nring_hit":
                    rc0 = rc._replace(include_time=False)
                    terms.append(("hit_notime", jax.jit(lambda X, P=P, Ms=Ms, S=S, rc0=rc0: LD._FUNCS["nring_hit"](B, X, P, Ms, S, rc0))))
            us = np.linspace(-0.05, 0.05, 11)
            for j in order:
                line = f"    kink scan entry {free[j]} (g_u {gu[j]:+.1f}):"
                for name, fn in terms:
                    vals = []
                    for u in us:
                        xx = xh.copy(); xx[free[j]] += u * st[j]
                        vals.append(float(fn(B.asarray(xx))))
                    vals = np.array(vals) - vals[5]
                    line += f"  {name}: " + " ".join(f"{v:+.2f}" for v in vals)
                print(line, flush=True)
            continue
        if typ == "mr":
            # re-polish from xh with scipy's own report: why does L-BFGS stop on a descent slope?
            from scipy.optimize import minimize as _min
            free = args[3]
            st = mrf._step_sizes(args[1], free)
            fb = [args[2][i] for i in free]
            x0 = xh[free]

            def fu(u, x0=x0, st=st, free=free, xh=xh):
                xx = xh.copy(); xx[free] = x0 + st * u
                val, gg = nll.value_and_grad(xx)
                return float(val), np.asarray(gg)[free] * st
            ub = [(None if lo is None else (lo - x0[i]) / st[i], None if hi is None else (hi - x0[i]) / st[i])
                  for i, (lo, hi) in enumerate(fb)]
            v0, g0 = fu(np.zeros(len(free)))
            r = _min(fu, np.zeros(len(free)), jac=True, method="L-BFGS-B", bounds=ub,
                     options=dict(maxiter=500, gtol=1e-2, ftol=0.0))
            du = (xs[free] - x0) / st
            print(f"    re-polish: {r.message} nit {r.nit} nfev {r.nfev} df {r.fun - v0:+.3f}; |g_u|inf at xh {np.max(np.abs(g0)):.3f}; "
                  f"d in steps {np.round(du, 2).tolist()}; g_u {np.round(g0, 3).tolist()}; at bound "
                  f"{[i for i, (lo, hi) in enumerate(ub) if (lo is not None and abs(lo) < 1e-9) or (hi is not None and abs(hi) < 1e-9)]}", flush=True)
        print(f"  fit {k:3d} {typ} {args[3] if typ == 'mr' else ''}: f(xh) - f(xs) {fh - fs:+.2f}  segment max above f(xh) "
              f"{max(prof) - fh:+.2f}  g.d {g @ d:+.3f}  fd slope {fd:+.3f}  profile " + " ".join(f"{v - fh:+.1f}" for v in prof), flush=True)
