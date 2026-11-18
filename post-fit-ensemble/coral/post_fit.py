"""
    Learned post-fit estimator: corrects a single-ring fit's momentum, vertex and direction from what
    the other hypothesis' fit of the same sub-event says (momentum ratio, vertex shift, nLL difference),
    the fitted geometry and the hit count per unit momentum. Gradient-boosted trees trained on simulated
    guns (files 0-299) and stored as arrays; applied to the WRITTEN values (after the vertex-bias
    correction and the frame rotation), only where the fitted distance to the wall along the track is
    at least the estimator's gate.
"""
import numpy as np

from fitqun_tuning.corrections import resolve, _towall

estimators = {}


def load(particle, path):
    with np.load(resolve(path)) as z:
        estimators[particle] = {k: z[k] for k in z.files}
    return estimators[particle]


def _predict(est, target, x):
    f, th, nl = est[f"{target}_feature"], est[f"{target}_threshold"], est[f"{target}_nan_left"]
    le, ri, leaf, val = est[f"{target}_left"], est[f"{target}_right"], est[f"{target}_leaf"], est[f"{target}_value"]
    s = float(est[f"{target}_baseline"])
    for n in est[f"{target}_roots"]:
        while not leaf[n]:
            v = x[f[n]]
            n = (le[n] if nl[n] else ri[n]) if np.isnan(v) else (le[n] if v <= th[n] else ri[n])
        s += val[n]
    return s


def _frame(d):
    ref = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _unit(x):
    st = np.sin(x[4])
    return np.array([st*np.cos(x[5]), st*np.sin(x[5]), np.cos(x[4])])


def correct(particle, own, other, nhits):
    """own, other = (nll, x) of this hypothesis and the other one, x = (t, x, y, z, theta, phi, p) as
    written; returns the corrected x (unchanged without an estimator or outside its gate)."""
    est = estimators.get(particle)
    x = np.array(own[1], dtype=float)
    if est is None or not (np.all(np.isfinite(x)) and np.all(np.isfinite(other[1]))):
        return x
    R, H = float(est["wall_radius_cm"]), float(est["wall_half_height_cm"])
    d, y = _unit(x), np.asarray(other[1], dtype=float)
    dy = _unit(y)
    towall = _towall(x[1], x[2], x[3], *d, R, H)
    if not towall >= float(est["gate_towall_cm"]):
        return x
    e1, e2 = _frame(d)
    r = np.hypot(x[1], x[2])
    dvh = y[1:4] - x[1:4]
    feats = {"dlogp": np.log(y[6]/x[6]), "dwall": min(R - r, H - abs(x[3])), "towall": towall,
             "r": r, "z": x[3], "cz": d[2], "crad": (x[1]*d[0] + x[2]*d[1])/max(r, 1.0),
             "dnll": other[0] - own[0], "nll": own[0],
             "dv_along": dvh @ d, "dv_u": dvh @ e1, "dv_v": dvh @ e2, "dt": y[0] - x[0],
             "dd_u": np.degrees(dy @ e1), "dd_v": np.degrees(dy @ e2), "e1z": e1[2], "e2z": e2[2],
             "e1rad": (e1[0]*x[1] + e1[1]*x[2])/max(r, 1.0), "e2rad": (e2[0]*x[1] + e2[1]*x[2])/max(r, 1.0),
             "q_per_p": np.nan, "hits_per_p": np.log(nhits/x[6])}
    f = np.array([feats[k] for k in est["features"]], dtype=float)
    pred = {t: _predict(est, t, f) for t in est["targets"]}
    out = x.copy()
    out[6] = x[6]*np.exp(-pred["logp"])
    out[1:4] = x[1:4] - (pred["along"]*d + pred["u"]*e1 + pred["v"]*e2)
    nd = d + np.radians(pred["a1"])*e1 + np.radians(pred["a2"])*e2
    nd /= np.linalg.norm(nd)
    out[4], out[5] = np.arccos(np.clip(nd[2], -1., 1.)), np.arctan2(nd[1], nd[0]) % (2.*np.pi)
    return out
