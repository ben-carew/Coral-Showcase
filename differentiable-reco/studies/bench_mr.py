"""R39 port: the N-ring likelihood (pi0 / multi-ring) in numba vs the differentiable backends,
on likelihoods captured from a real Coral run (as bench_1r.py).

usage: bench_mr.py <config> <input.root> <n_events> <out.json> [backends=numba,jax] [per_kind=10]
"""
import json, sys, tempfile, time
import numpy as np

cfg_path, infile, n_events, out = sys.argv[1:5]
backends = (sys.argv[5] if len(sys.argv) > 5 else "numba,jax").split(",")
per_kind = int(sys.argv[6]) if len(sys.argv) > 6 else 10

import fitqun.globals as gbl
import fitqun_fit.likelihood_multiring as lm
from fitqun_fit.likelihood_diff import make_nring_nll

records = []
_orig = lm.MultiRingLikelihood.make_nll


def capture(self, ring_ptypes, include_time=True, include_scattering=True, tpdf_total=None):
    nll = _orig(self, ring_ptypes, include_time, include_scattering, tpdf_total)
    rec = dict(mrl=self, types=tuple(ring_ptypes), time=include_time, scat=include_scattering,
               tpdf_total=self._tpdf_total if tpdf_total is None else tpdf_total, x=None, n=0)
    records.append(rec)

    def wrapped(x):
        rec["x"] = np.array(x, dtype=float)
        rec["n"] += 1
        return nll(x)
    return wrapped


lm.MultiRingLikelihood.make_nll = capture
import fitqun.fitqun as fq
sys.argv = ["fitqun", cfg_path, "-i", infile, "-o", tempfile.mkdtemp(), "-n", n_events]
t0 = time.time()
fq.main()
lm.MultiRingLikelihood.make_nll = _orig
print(f"capture: {len(records)} N-ring likelihoods in {time.time()-t0:.0f} s", flush=True)

pool = {}
for r in records:
    if r["n"] >= 3:
        pool.setdefault((len(r["types"]), r["time"], r["scat"]), []).append(r)
sample = [r for k in sorted(pool) for r in pool[k][:per_kind]]
print("kinds:", {str(k): len(v) for k, v in pool.items()}, "-> sample", len(sample), flush=True)


def build(b, r):
    if b == "numba":
        gbl.likelihood_backend = "numba"
        return _orig(r["mrl"], list(r["types"]), r["time"], r["scat"], r["tpdf_total"])
    return make_nring_nll(b, r["mrl"], r["types"], r["time"], r["scat"], r["tpdf_total"])


rng = np.random.default_rng(39)
rows = []
for ir, r in enumerate(sample):
    x0 = r["x"]
    nr = len(r["types"])
    step = np.r_[0.02, 0.05, 0.05, 0.05, np.tile([1e-4, 1e-4, 0.05, 0.05], nr)]
    jit = np.r_[0.3, 3.0, 3.0, 3.0, np.tile([0.005, 0.005, 0.0, 1.0], nr)]
    pts = [x0] + [x0 + jit * rng.normal(size=len(x0)) for _ in range(4)]
    row = dict(i=ir, nring=nr, types="".join(t[0] for t in r["types"]), time=r["time"], scat=r["scat"],
               n_pmts=len(r["mrl"]._sel), n_hits=int(r["mrl"]._sel.sum()))
    f = {}
    for b in backends:
        try:
            t1 = time.time(); fb = build(b, r); row[f"{b}_build_s"] = time.time() - t1
            t1 = time.time(); fb(x0); row[f"{b}_first_s"] = time.time() - t1
            row[f"{b}_vals"] = [fb(x) for x in pts]
            if hasattr(fb, "value_and_grad"):
                t1 = time.time(); fb.value_and_grad(x0); row[f"{b}_first_grad_s"] = time.time() - t1
                row[f"{b}_grad"] = fb.value_and_grad(x0)[1].tolist()
            f[b] = fb
        except Exception as e:
            import traceback; traceback.print_exc()
            row[f"{b}_error"] = repr(e)[:300]
    if "numba" in f:
        g = []
        for k in range(len(x0)):
            e = np.zeros(len(x0)); e[k] = step[k]
            g.append((f["numba"](x0 + e) - f["numba"](x0 - e)) / (2 * step[k]))
        row["fd_grad"] = g
    K = 30
    xs = [x0 + 1e-3 * jit * rng.normal(size=len(x0)) for _ in range(K)]
    for b, fb in f.items():
        ts = []
        for x in xs:
            t1 = time.perf_counter(); fb(x); ts.append(time.perf_counter() - t1)
        row[f"{b}_f_ms"] = 1e3 * float(np.median(ts))
        if hasattr(fb, "value_and_grad"):
            ts = []
            for x in xs:
                t1 = time.perf_counter(); fb.value_and_grad(x); ts.append(time.perf_counter() - t1)
            row[f"{b}_fg_ms"] = 1e3 * float(np.median(ts))
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                      if not isinstance(v, list)}), flush=True)
json.dump(rows, open(out, "w"))
print("wrote", out)
