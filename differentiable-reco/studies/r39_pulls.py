"""R39: per-event uncertainties from the exact Hessian of the NLL at the fitted point
(dump_hessian, JAX backend): pulls against truth and whether the predicted error ranks events.

For every fit of the true hypothesis (sub-event 0) the covariance is C = H^-1. Pulls:
momentum (p_fit - p_true)/sigma_p, the vertex along the true track and in two directions
transverse to it, theta and phi. A calibrated error gives pull width 1 (iqr68 and the
Gaussian core); the Spearman correlation of sigma with |residual| says whether the
per-event error carries information about the event's actual error.

usage: r39_pulls.py <hessian.csv> <truth.root> [particle=e-]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Software/Sextant/src"))
from sextant.io import truth as truth_io                      # noqa: E402
from sextant.metrics.stats import gauss_sigma                 # noqa: E402

path, truth_path = sys.argv[1:3]
particle = sys.argv[3] if len(sys.argv) > 3 else "e-"
h = pd.read_csv(path)
h = h[(h.particle == particle) & (h.sub_event == 0)].drop_duplicates("event").set_index("event")
rings, _ = truth_io.read(truth_path)
tr = rings[rings.is_primary_lepton].drop_duplicates("event").set_index("event")
ev = h.index.intersection(tr.index)
h, t = h.loc[ev], tr.loc[ev]
H = h[[f"h{i}{j}" for i in range(7) for j in range(7)]].to_numpy().reshape(-1, 7, 7)
X = h[["t", "x", "y", "z", "theta", "phi", "p"]].to_numpy()
u = t[["dx", "dy", "dz"]].to_numpy()
u /= np.linalg.norm(u, axis=1, keepdims=True)


def iqr68(a):
    lo, hi = np.percentile(a, [16, 84])
    return (hi - lo) / 2


rows = []
for k in range(len(ev)):
    Hk = 0.5 * (H[k] + H[k].T)
    w = np.linalg.eigvalsh(Hk)
    if not np.all(np.isfinite(Hk)) or w.min() <= 0:
        rows.append(None)
        continue
    C = np.linalg.inv(Hk)
    d = X[k, 1:4] - t[["x", "y", "z"]].to_numpy()[k]
    uk = u[k]
    e1 = np.cross(uk, [0, 0, 1.0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0, 0])
    e2 = np.cross(uk, e1)
    Cv = C[1:4, 1:4]
    th_t = np.arccos(np.clip(uk[2], -1, 1))
    ph_t = np.arctan2(uk[1], uk[0]) % (2 * np.pi)
    dphi = (X[k, 5] - ph_t + np.pi) % (2 * np.pi) - np.pi
    rows.append(dict(
        dp=X[k, 6] - t.p.to_numpy()[k], sp=np.sqrt(C[6, 6]),
        dl=d @ uk, sl=np.sqrt(uk @ Cv @ uk),
        dt1=d @ e1, st1=np.sqrt(e1 @ Cv @ e1), dt2=d @ e2, st2=np.sqrt(e2 @ Cv @ e2),
        dth=X[k, 4] - th_t, sth=np.sqrt(C[4, 4]), dph=dphi, sph=np.sqrt(C[5, 5]),
        ang=np.degrees(np.arccos(np.clip(np.dot([np.sin(X[k, 4]) * np.cos(X[k, 5]),
                                                 np.sin(X[k, 4]) * np.sin(X[k, 5]), np.cos(X[k, 4])], uk), -1, 1)))))
ok = [r for r in rows if r is not None]
R = pd.DataFrame(ok)
n_bad = sum(r is None for r in rows)
R = R[R.ang < 20]                                   # converged fits (Sextant's convention)
print(f"{particle}: {len(ev)} fits with truth, {n_bad} non-positive-definite Hessians, {len(R)} converged (< 20 deg)")
print(f"{'quantity':22s} {'median sigma':>13s} {'residual iqr68':>15s} {'pull iqr68':>11s} {'pull core':>10s} {'Spearman(sigma,|res|)':>22s}")
from scipy.stats import spearmanr
for name, dcol, scol, unit, f in (("momentum", "dp", "sp", "MeV/c", 1.0), ("vertex along track", "dl", "sl", "cm", 1.0),
                                  ("vertex transverse 1", "dt1", "st1", "cm", 1.0), ("vertex transverse 2", "dt2", "st2", "cm", 1.0),
                                  ("theta", "dth", "sth", "mrad", 1e3), ("phi", "dph", "sph", "mrad", 1e3)):
    pull = R[dcol] / R[scol]
    rho = spearmanr(R[scol], R[dcol].abs()).correlation
    print(f"{name:22s} {f * R[scol].median():10.3f} {unit:5s} {f * iqr68(R[dcol]):10.3f} {unit:5s} {iqr68(pull):9.2f} "
          f"{gauss_sigma(pull.to_numpy()):10.2f} {rho:16.2f}")
# events ranked by the predicted momentum error: does the worst-predicted quartile have the worst actual error?
q = pd.qcut(R.sp / R.dp.abs().median(), 4, labels=False, duplicates="drop") if len(R) > 20 else None
if q is not None:
    print("momentum residual iqr68 by quartile of predicted sigma_p:",
          [round(iqr68(R.dp[q == k]), 1) for k in range(int(q.max()) + 1)], "MeV/c")
