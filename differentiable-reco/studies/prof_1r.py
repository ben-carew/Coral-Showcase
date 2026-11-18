"""R39 port: where the JAX 1R likelihood spends its time, and whether its gradient converges
to central differences as the step shrinks (JAX's own value and numba's).

usage: prof_1r.py <config> <input.root> [torch=1]"""
import sys, tempfile, time
import numpy as np

cfg_path, infile = sys.argv[1:3]
do_torch = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"
import fitqun_fit.likelihood as lk
from fitqun import full_fit_vars_names
recs = []
_orig = lk.neg_loglikelihood_wrapper


def capture(pmts, hits, particle, unhit_term_weight=1.0, step=None, qe=None):
    nll = _orig(pmts, hits, particle, unhit_term_weight, step, qe)
    rec = dict(args=(pmts, hits, particle, unhit_term_weight, step, qe), X=None)
    recs.append(rec)

    def wrapped(*a, **kw):
        rec["X"] = list(a) if a else [kw[k] for k in full_fit_vars_names]
        return nll(*a, **kw)
    for n in ("pmt_mu", "term_sums"):
        if hasattr(nll, n):
            setattr(wrapped, n, getattr(nll, n))
    return wrapped


lk.neg_loglikelihood_wrapper = capture
import fitqun.fitqun as fq
import fitqun.globals as gbl
sys.argv = ["fitqun", cfg_path, "-i", infile, "-o", tempfile.mkdtemp(), "-n", "1"]
fq.main()
lk.neg_loglikelihood_wrapper = _orig
rec = next(r for r in recs if (r["args"][4] or {}).get("select") == "charge_threshold" and len(r["args"][1]) and r["X"])
pmts, hits, particle, w, step, qe = rec["args"]
X0 = np.array(rec["X"])
print("record:", particle, len(pmts), "PMTs", len(hits), "hits", flush=True)

import jax
from fitqun_fit import likelihood_diff as L
gbl.likelihood_backend = "jax"
f_jax = _orig(pmts, hits, particle, w, step, qe)
f_nb = (gbl.__setattr__("likelihood_backend", "numba"), _orig(pmts, hits, particle, w, step, qe))[1]
B = L.backend("jax")
parts = f_jax.parts.parts                       # [(kind, PMT set, M, S, cfg)]


def timeit(fn, n=60):
    fn(); ts = []
    for _ in range(n):
        t1 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t1)
    return 1e3 * np.median(ts)


Xj = f_jax.parts.X(X0)
print("jax devices:", jax.devices(), flush=True)
print("timings (ms):")
print("  numba value         ", round(timeit(lambda: f_nb(*X0)), 3))
print("  jax value (wrapper) ", round(timeit(lambda: f_jax(*X0)), 3))
for kind, P, M, S, cfg in parts:
    f = L._compiled(B, kind)
    print(f"  jax {kind:6s} part      ", round(timeit(lambda: f(Xj, P, M, S, cfg).block_until_ready()), 3))
    if kind == "hit":
        print("  jax hit, no time    ", round(timeit(lambda: f(Xj, P, M, S, cfg._replace(use_tpdf=False)).block_until_ready()), 3))
        K = dict(cfg.k)
        red = jax.jit(lambda X: L._reduced(B, M, K, (X[1], X[2], X[3]),
                      (jax.numpy.sin(X[4]) * jax.numpy.cos(X[5]), jax.numpy.sin(X[4]) * jax.numpy.sin(X[5]), jax.numpy.cos(X[4])),
                      L._scalars(B, X, M, K, cfg)[1]))
        print("  jax contraction only", round(timeit(lambda: red(Xj).block_until_ready()), 3))
print("  jax value+grad      ", round(timeit(lambda: f_jax.value_and_grad(X0)), 3))
print("  jax hessian         ", round(timeit(lambda: f_jax.hessian(X0), 10), 3))

g = f_jax.value_and_grad(X0)[1]
scale = np.array([1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3, 1.0])
print("gradient check: exact jax grad vs central differences at step h*scale")
for k in range(7):
    line = f"  {full_fit_vars_names[k]:5s} exact {g[k]: .6e} |"
    for h in (1e-1, 1e-2, 1e-3, 1e-4):
        e = np.zeros(7); e[k] = h * scale[k]
        fj = (f_jax(*(X0 + e)) - f_jax(*(X0 - e))) / (2 * e[k])
        fn = (f_nb(*(X0 + e)) - f_nb(*(X0 - e))) / (2 * e[k])
        line += f" h={h:g}: jax {fj: .5e} numba {fn: .5e} |"
    print(line, flush=True)

if do_torch:
    import os
    for mode in ("1", "0"):
        os.environ["CORAL_TORCH_COMPILE"] = mode
        L._COMPILED.clear()
        try:
            gbl.likelihood_backend = "torch"
            f_t = _orig(pmts, hits, particle, w, step, qe)
            t1 = time.time(); v = f_t(*X0); first = time.time() - t1
            t1 = time.time(); f_t.value_and_grad(X0); first_g = time.time() - t1
            print(f"torch compile={mode}: value {v:.10f} (numba {f_nb(*X0):.10f}) first {first:.1f} s, first grad {first_g:.1f} s,"
                  f" value {timeit(lambda: f_t(*X0), 30):.3f} ms, value+grad {timeit(lambda: f_t.value_and_grad(X0), 30):.3f} ms;"
                  f" grad diff vs jax {np.max(np.abs(f_t.value_and_grad(X0)[1] - g) / (np.abs(g) + 1e-9)):.1e}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"torch compile={mode} FAILED: {repr(e)[:400]}", flush=True)
