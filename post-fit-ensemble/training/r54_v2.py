#!/usr/bin/env python
"""R54 v2: the post-fit estimator, applied only where the FITTED towall >= T.

T = the smallest of GATES (m) with no (true p, true towall) bin worse beyond CI in 5-fold cross-validation
(whole files held out) on the training files 0-299; models then trained on all of 0-299, frozen, and applied
once to the FRESH files 700-999 (r54_fresh_{e,mu}) with the same pre-registered PASS as v1 (roadmap R54).

    python r54_v2.py --hyp e-     # or mu-
"""
import argparse, importlib.util, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v1", ROOT / "Analysis/Scripts/r54_heldout.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
GATES = [0, 1, 2, 3, 4, 6]
RNG = np.random.default_rng(542)


def model():
    return HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.04, min_samples_leaf=50,
                                         loss="absolute_error")


def evaluate(Y, C, ptrue, twt, hyp, boot):
    """(overall rows, per-bin rows, list of bins worse beyond CI)."""
    def compare(i):
        m0, m1 = v1.metrics(Y, i), v1.metrics(C, i)
        bs = {k: [] for k in m0}
        for _ in range(boot):
            j = RNG.choice(i, len(i))
            b0, b1 = v1.metrics(Y, j), v1.metrics(C, j)
            for k in m0:
                bs[k].append(b1[k]/b0[k] if k != "abs_bias" else b1[k] - b0[k])
        return {k: (m0[k], m1[k], *np.percentile(bs[k], [2.5, 97.5])) for k in m0}
    allr = compare(np.arange(len(Y)))
    pb = [0, 200, 400, 600, 800, 1200, 3000] if hyp == "e-" else [0, 400, 600, 800, 1000, 1500, 3000]
    tb = [0, 300, 600, 1500, 1e9]
    rows, worse = [], []
    for plo, phi in zip(pb[:-1], pb[1:]):
        for tlo, thi in zip(tb[:-1], tb[1:]):
            i = np.flatnonzero((ptrue >= plo) & (ptrue < phi) & (twt >= tlo) & (twt < thi))
            if len(i) < 40:
                continue
            r = compare(i)
            lab = f"p {plo}-{phi}, towall {tlo/100:.0f}-{min(thi/100, 99):.0f} m"
            rows.append((lab, len(i), r))
            worse += [f"{lab} {k} x{r[k][1]/r[k][0]:.2f} [{r[k][2]:.2f},{r[k][3]:.2f}]"
                      for k in ("p_core", "vtx68", "ang68") if r[k][2] > 1.0]
            if r["abs_bias"][2] > 0:
                worse.append(f"{lab} |bias| {r['abs_bias'][0]:.2f} -> {r['abs_bias'][1]:.2f}%")
    return allr, rows, worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default="e-")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--prefix", default="r54", help="runs <prefix>_train_* / <prefix>_fresh_*, models in Output/Sextant/<prefix>")
    a = ap.parse_args()
    smp, sample = ("e", "electron") if a.hyp == "e-" else ("mu", "muon")
    truth = v1.Truth.from_file(ROOT / f"Output/Truth/truth_{sample}_v1.12.29.root")
    Xa, Ya, pa, ta = v1.table(f"{a.prefix}_train_{smp}", a.hyp, sample, truth)
    groups = Xa.index.to_numpy()//100
    P = pd.DataFrame(0.0, index=Ya.index, columns=v1.TARGETS)
    for c in v1.TARGETS:
        for trn, te in GroupKFold(5).split(Xa, Ya[c], groups):
            P.iloc[te, P.columns.get_loc(c)] = model().fit(Xa.iloc[trn], Ya[c].iloc[trn]).predict(Xa.iloc[te])
    chosen = None
    for T in GATES:
        g = (Xa.towall.to_numpy() >= 100*T)[:, None]
        allr, _, worse = evaluate(Ya, Ya - P*g, pa, ta, a.hyp, a.boot)
        print(f"  CV gate {T} m: p core x{allr['p_core'][1]/allr['p_core'][0]:.3f}, vertex x{allr['vtx68'][1]/allr['vtx68'][0]:.3f}, "
              f"bins worse {len(worse)}" + (f": {worse}" if worse else ""))
        if not worse and chosen is None:
            chosen = T
    if chosen is None:
        print("no gate passes in CV: stop (pre-registered)")
        return
    print(f"chosen gate: fitted towall >= {chosen} m")
    models = {c: model().fit(Xa, Ya[c]) for c in v1.TARGETS}
    out = ROOT / "Output/Sextant" / a.prefix; out.mkdir(parents=True, exist_ok=True)
    pickle.dump({"features": list(Xa.columns), "models": models, "gate_towall_cm": 100*chosen},
                open(out / f"post_fit_estimator_v2_{smp}.pkl", "wb"))
    Xb, Yb, pb_, tb_ = v1.table(f"{a.prefix}_fresh_{smp}", a.hyp, sample, truth)
    assert (Xb.index//100).min() >= 700
    g = (Xb.towall.to_numpy() >= 100*chosen)[:, None]
    Cb = Yb - pd.DataFrame({c: models[c].predict(Xb) for c in v1.TARGETS}, index=Yb.index)*g
    allr, rows, worse = evaluate(Yb, Cb, pb_, tb_, a.hyp, 400)
    print(f"{a.hyp} FRESH files 700-999: {len(Xb)} events, {g.mean():.0%} corrected")
    print("  all: " + "  ".join(f"{k} {v[0]:.2f} -> {v[1]:.2f} [{v[2]:.3f},{v[3]:.3f}]" for k, v in allr.items()))
    for lab, n, r in rows:
        print(f"    {lab:30s} n={n:4d}  p_core x{r['p_core'][1]/r['p_core'][0]:.2f}  vtx68 x{r['vtx68'][1]/r['vtx68'][0]:.2f}"
              f"  ang68 x{r['ang68'][1]/r['ang68'][0]:.2f}  |bias| {r['abs_bias'][0]:.2f} -> {r['abs_bias'][1]:.2f}%")
    print("  worse beyond CI:", worse if worse else "none")
    ratio_hi = allr["p_core"][3]
    print(f"  p core ratio {allr['p_core'][1]/allr['p_core'][0]:.3f} [{allr['p_core'][2]:.3f},{ratio_hi:.3f}] -> "
          + ("PASS" if (ratio_hi <= 0.90 or a.hyp != "e-") and not worse else "FAIL"))


if __name__ == "__main__":
    main()
