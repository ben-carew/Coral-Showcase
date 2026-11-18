"""Summarise bench_1r / bench_mr JSON: agreement with numba, gradients vs central differences,
and per-kind median timings. usage: bench_summary.py <bench.json> [...]"""
import json, sys
import numpy as np

for path in sys.argv[1:]:
    rows = json.load(open(path))
    bks = sorted({k[:-5] for r in rows for k in r if k.endswith("_vals")})
    key = "kind" if "kind" in rows[0] else "nring"
    print(f"== {path}: {len(rows)} likelihoods, backends {bks}")
    errs = {b: sum(1 for r in rows if f"{b}_error" in r) for b in bks + ["torch", "jax"]}
    print("   errors:", {b: n for b, n in errs.items() if n})
    for b in bks:
        if b == "numba":
            continue
        rel = [np.max(np.abs(np.array(r[f"{b}_vals"]) - r["numba_vals"]) / np.abs(r["numba_vals"]))
               for r in rows if f"{b}_vals" in r and "numba_vals" in r]
        gr = []
        for r in rows:
            if f"{b}_grad" in r and "fd_grad" in r:
                g, fd = np.array(r[f"{b}_grad"]), np.array(r["fd_grad"])
                gr.append(np.max(np.abs(g - fd) / (np.abs(fd) + 1e-3 * np.abs(fd).max() + 1e-9)))
        print(f"   {b:6s} value rel diff vs numba: max {max(rel):.1e} median {np.median(rel):.1e}"
              + (f" | grad vs central diff: median rel {np.median(gr):.1e}, max {max(gr):.1e}" if gr else ""))
    kinds = sorted({str(r[key]) for r in rows})
    print(f"   {'kind':12s} {'n':>3s} " + " ".join(f"{b+' f':>9s} {b+' fg':>9s} {b+' 1st':>8s}" for b in bks) + "   (ms, ms, s)")
    for k in kinds:
        rk = [r for r in rows if str(r[key]) == k]
        cells = []
        for b in bks:
            f = [r[f"{b}_f_ms"] for r in rk if f"{b}_f_ms" in r]
            fg = [r[f"{b}_fg_ms"] for r in rk if f"{b}_fg_ms" in r]
            first = [r[f"{b}_first_s"] for r in rk if f"{b}_first_s" in r]
            cells.append(f"{np.median(f) if f else np.nan:9.2f} {np.median(fg) if fg else np.nan:9.2f} {np.max(first) if first else np.nan:8.2f}")
        print(f"   {k:12s} {len(rk):3d} " + " ".join(cells))
