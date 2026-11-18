"""R39 port: the 1R likelihood in numba vs JAX vs PyTorch on likelihoods captured from a real Coral run.

Phase 1 runs Coral (numba) on N events and records every likelihood it builds (inputs and
the last point evaluated). Phase 2 rebuilds a sample of them in each backend: value at the
recorded point and 4 perturbations, gradient vs central differences of the numba value,
first-call (compile) time, and the median time of f and of f+gradient.

usage: bench_1r.py <config> <input.root> <n_events> <out.json> [backends=numba,jax,torch] [per_kind=12]
"""
import json, sys, tempfile, time
import numpy as np

cfg_path, infile, n_events, out = sys.argv[1:5]
backends = (sys.argv[5] if len(sys.argv) > 5 else "numba,jax,torch").split(",")
per_kind = int(sys.argv[6]) if len(sys.argv) > 6 else 12

import fitqun_fit.likelihood as lk
from fitqun_fit.likelihood_cpu import nll_fast
import fitqun_tuning.corrections as corrections
from fitqun_tuning.cherenkov_profile import cprofs
from fitqun_tuning.time_pdf import tpdfs
from fitqun import full_fit_vars_names

records = []
_orig = lk.neg_loglikelihood_wrapper


def capture(pmts, hits, particle, unhit_term_weight=1.0, step=None, qe=None):
    nll = _orig(pmts, hits, particle, unhit_term_weight, step, qe)
    rec = dict(pmts=pmts, hits=hits, particle=particle, w=float(unhit_term_weight),
               step=dict(step or {}), qe=qe, X=None, n=0)
    records.append(rec)

    def wrapped(*args, **kw):
        rec["X"] = list(args) if args else [kw[k] for k in full_fit_vars_names]
        rec["n"] += 1
        return nll(*args, **kw)
    for a in ("term_sums", "pmt_mu", "time_term", "hit_times"):
        if hasattr(nll, a):
            setattr(wrapped, a, getattr(nll, a))
    return wrapped


lk.neg_loglikelihood_wrapper = capture
import fitqun.fitqun as fq
sys.argv = ["fitqun", cfg_path, "-i", infile, "-o", tempfile.mkdtemp(), "-n", n_events]
t0 = time.time()
fq.main()
print(f"capture: {len(records)} likelihoods from {n_events} events in {time.time()-t0:.0f} s", flush=True)
lk.neg_loglikelihood_wrapper = _orig


def kind(r):
    s = r["step"]
    if len(r["hits"]) == 0:
        return "dim"
    if s.get("dispersion", 0) > 0:
        return "step0"
    if s.get("select") == "charge_threshold":
        return "step2_core"
    if s.get("fit_vars_names") == ["t", "p"]:
        return "step1"
    return "other"


pool = {}
for r in records:
    if r["n"] >= 3 and r["X"] is not None:
        pool.setdefault(kind(r), []).append(r)
sample = [r for k in ("step2_core", "step0", "dim", "step1") for r in pool.get(k, [])[:per_kind]]
print("kinds:", {k: len(v) for k, v in pool.items()}, "-> sample", len(sample), flush=True)


def build(b, r):
    s, prt = r["step"], r["particle"]
    kw = dict(dispersion=s.get("dispersion", 0.0), charge_weight=s.get("charge_weight", False),
              unhit_calib=s.get("unhit_calib", False), qe=r["qe"], geom_curves=corrections.geom_light.get(prt))
    if b == "numba":
        return nll_fast(r["pmts"], r["hits"], cprofs[prt], tpdfs[prt], r["w"], **kw)
    from fitqun_fit.likelihood_diff import nll_diff
    return nll_diff(b, r["pmts"], r["hits"], cprofs[prt], tpdfs[prt], r["w"], **kw)


rng = np.random.default_rng(39)
H = np.array([0.02, 0.05, 0.05, 0.05, 1e-4, 1e-4, 0.05])      # FD steps (ns, cm, rad, MeV)
J = np.array([0.3, 3.0, 3.0, 3.0, 0.005, 0.005, 0.0])         # perturbations
rows = []
for ir, r in enumerate(sample):
    X0 = np.array(r["X"], dtype=float)
    pts = [X0] + [X0 + J * rng.normal(size=7) * np.r_[1, 1, 1, 1, 1, 1, 0] + np.r_[0, 0, 0, 0, 0, 0, 0.01 * X0[6] * rng.normal()]
                  for _ in range(4)]
    row = dict(i=ir, kind=kind(r), particle=r["particle"], n_pmts=len(r["pmts"]), n_hits=len(r["hits"]))
    f = {}
    for b in backends:
        try:
            t1 = time.time(); fb = build(b, r); row[f"{b}_build_s"] = time.time() - t1
            t1 = time.time(); v0 = fb(*X0); row[f"{b}_first_s"] = time.time() - t1
            row[f"{b}_vals"] = [fb(*x) for x in pts]
            if hasattr(fb, "value_and_grad"):
                t1 = time.time(); fb.value_and_grad(X0); row[f"{b}_first_grad_s"] = time.time() - t1
                row[f"{b}_grad"] = fb.value_and_grad(X0)[1].tolist()
            f[b] = fb
        except Exception as e:                      # report and carry on with the other backends
            import traceback; traceback.print_exc()
            row[f"{b}_error"] = repr(e)[:300]
    if "numba" in f:   # central differences of the numba value
        g = []
        for k in range(7):
            e = np.zeros(7); e[k] = H[k]
            g.append((f["numba"](*(X0 + e)) - f["numba"](*(X0 - e))) / (2 * H[k]))
        row["fd_grad"] = g
    # timing, interleaved: K calls each at jittered points
    K = 40
    xs = [X0 + 1e-3 * J * rng.normal(size=7) for _ in range(K)]
    for b, fb in f.items():
        ts = []
        for x in xs:
            t1 = time.perf_counter(); fb(*x); ts.append(time.perf_counter() - t1)
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
