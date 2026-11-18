"""R39: the on-device Nelder-Mead (likelihood_diff.device_nelder_mead) against scipy's on the
same captured fit steps (JAX likelihood, same seed and initial simplex): final nLL, x,
iterations, evaluations and wall time. usage: nm_test.py <config (backend = jax)> <input.root> <n_events>"""
import sys, tempfile, time
import numpy as np
cfg_path, infile, n = sys.argv[1:4]
import fitqun.globals as gbl
import fitqun_fit.minimization as mn
calls = []
_orig = mn.update_variables


def capture(full_nll, names, seed, full_bounds, method, options, simplex_scale=1.0, trace=None, scale=False, jac=None):
    if method == "Nelder-Mead" and len(calls) < 24:
        calls.append(dict(nll=full_nll, names=list(names), seed=dict(seed), bounds=dict(full_bounds),
                          options=dict(options), scale=simplex_scale))
    return _orig(full_nll, names, seed, full_bounds, method, options, simplex_scale, trace, scale, jac)


mn.update_variables = capture
import fitqun.fitqun as fq
sys.argv = ["fitqun", cfg_path, "-i", infile, "-o", tempfile.mkdtemp(), "-n", n]
fq.main()
mn.update_variables = _orig
print(f"captured {len(calls)} simplex steps", flush=True)
rows = []
for i, c in enumerate(calls):
    out = {}
    for dev in (False, True):
        gbl.device_minimiser = dev
        _orig(c["nll"], c["names"], c["seed"], c["bounds"], "Nelder-Mead", c["options"], c["scale"])  # warm (compile)
        t1 = time.perf_counter()
        fv, (ok, f, x), (nit, nfev, _) = _orig(c["nll"], c["names"], c["seed"], c["bounds"], "Nelder-Mead",
                                                c["options"], c["scale"])
        out[dev] = (time.perf_counter() - t1, f, x, nit, nfev, ok)
    (ts, fs, xs, its, ns, oks), (td, fd, xd, itd, nd, okd) = out[False], out[True]
    print(f"step {i:2d} {len(c['names'])} vars: scipy f {fs:.4f} nit {its} nfev {ns} {1e3*ts:.0f} ms | device f {fd:.4f} "
          f"nit {itd} nfev {nd} {1e3*td:.0f} ms | df {fd - fs:+.2e} max|dx| {np.max(np.abs(xd - xs)):.2e} "
          f"speed-up {ts / td:.1f}x", flush=True)
    rows.append((ts, td, fd - fs, its == itd))
ts, td, df, same = map(np.array, zip(*rows))
print(f"total scipy {ts.sum():.2f} s, device {td.sum():.2f} s ({ts.sum() / td.sum():.1f}x); |df| median {np.median(np.abs(df)):.1e} "
      f"max {np.max(np.abs(df)):.1e}; identical iteration counts {same.mean():.0%}")
