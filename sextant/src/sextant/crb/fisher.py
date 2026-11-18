"""Fisher information and the Cramer-Rao bound for one event.

Parameters, in order: (t, x, y, z, tilt1, tilt2, p). The two tilts are small
rotations of the direction about two axes perpendicular to it -- a chart with no
polar singularity, unlike (theta, phi) -- and the opening angle Sextant reports is
their quadrature sum.

    I_ab = SUM_j w(mu_j)            dmu_j/dx_a  dmu_j/dx_b        charge + occupancy
         + SUM_j P_hit(mu_j) I_t,j  dT_j/dx_a   dT_j/dx_b         hit times

w(mu) is the exact information in one PMT's hit/charge (`response.py`); I_t,j is the
measured information per digitised hit for a shift of its arrival time (`build.py`).

Derivatives are central finite differences of the model. The model is piecewise
trilinear, so a step well inside one cell returns that cell's slope EXACTLY -- no
truncation error to tune -- and the steps below are ~50x smaller than the finest
cell (0.5 deg in angle).

Nuisance: an electromagnetic shower's total light fluctuates from shower to shower by
more than photon counting allows. That was measured directly (fixed vertex, direction
and energy; 400 showers per energy) at 2.24 / 1.59 / 1.55 % for 300 / 600 / 1000 MeV.
It is entered as an overall light scale xi with a Gaussian prior of that width and
marginalised, which is exactly what a fitter faces: it cannot tell a bright shower
from a more energetic one except through the ring's shape.

Physics floors (`floor.py`, when the model carries them): the event-level fluctuations of the
light's axis, emission centroid and emitting line, which shift the light pattern like a change of
direction or vertex would, are ADDED to the covariance (an independent error on top); the muon's
track-length straggling enters as the light-scale prior, like the shower yield. `floor=False` gives
the pure Fisher bound.
"""

from __future__ import annotations

import numpy as np

from .response import ChargeResponse, sigma_t_ski

PARAMS = ("t", "x", "y", "z", "tilt1", "tilt2", "p")
STEPS = np.array([0.02, 0.5, 0.5, 0.5, 2e-4, 2e-4, 2e-3])   # last is fractional

#: Measured shower light-yield fluctuation, electrons: (E [MeV], sigma [fraction]).
SHOWER_FLUCTUATION = {"e-": (np.array([300.0, 600.0, 1000.0]),
                             np.array([0.0224, 0.0159, 0.0155]))}

_RESPONSE = None


def charge_response():
    global _RESPONSE
    if _RESPONSE is None:
        _RESPONSE = ChargeResponse()
    return _RESPONSE


def shower_sigma(particle, p):
    """Relative shower-to-shower light fluctuation (0 where not measured)."""
    if particle not in SHOWER_FLUCTUATION:
        return 0.0
    e, s = SHOWER_FLUCTUATION[particle]
    if p <= e[0]:
        return float(s[0] * np.sqrt(e[0] / p))        # sampling-like below 300 MeV
    return float(np.interp(p, e, s))                  # flat beyond 1 GeV


def frame(direction):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, tmp)
    e1 /= np.linalg.norm(e1)
    return d, e1, np.cross(d, e1)


def _direction(d0, e1, e2, a, b):
    v = d0 + a * e1 + b * e2
    return v / np.linalg.norm(v)


def fisher(model, particle, vertex, direction, p, *, timing="measured"):
    """Return (I_charge, I_time, aux) for one event configuration.

    timing = "measured" (default) uses the measured information per hit;
             "ideal"   treats every hit as direct light timed by the digitiser
                       alone -- the optimistic limit, for comparison only;
             "none"    drops the time channel.
    """
    light = model.light[particle]
    resp = charge_response()
    v0 = np.asarray(vertex, float)
    d0, e1, e2 = frame(direction)
    p0 = float(p)

    def state(k, s):
        v, a, b, pp, t = v0.copy(), 0.0, 0.0, p0, 0.0
        if k == 0:
            t += s
        elif k in (1, 2, 3):
            v[k - 1] += s
        elif k == 4:
            a += s
        elif k == 5:
            b += s
        else:
            pp *= 1.0 + s
        return t, v, _direction(d0, e1, e2, a, b), pp

    mu0 = light.charge(v0, d0, p0)
    n = light.det.n
    dmu = np.zeros((7, n))
    dT = np.zeros((7, n))
    for k in range(7):
        h = STEPS[k]
        tp, vp, dp, pp = state(k, h)
        tm, vm, dm, pm = state(k, -h)
        step = 2.0 * h * (p0 if k == 6 else 1.0)
        if k != 0:
            dmu[k] = (light.charge(vp, dp, pp) - light.charge(vm, dm, pm)) / step
        dT[k] = (light.time(tp, vp, dp, pp) - light.time(tm, vm, dm, pm)) / step

    wq = resp.weight(mu0)
    phit = resp.p_hit(mu0)
    if timing == "measured":
        wt = phit * model.timing[particle](light.delta_alpha(v0, d0, p0), mu0)
    elif timing == "ideal":
        wt = _ideal_timing_weight(mu0)
    elif timing == "none":
        wt = np.zeros(n)
    else:
        raise ValueError(f"unknown timing mode {timing!r}")

    Iq = np.einsum("aj,bj,j->ab", dmu, dmu, wq)
    It = np.einsum("aj,bj,j->ab", dT, dT, wt)
    aux = dict(mu=mu0, phit=phit, wq=wq, wt=wt, dmu=dmu, dT=dT, d=d0, e1=e1, e2=e2,
               npe=float(mu0.sum() - light.mu_dark * n), nhit=float(phit.sum()))
    return Iq, It, aux


def _ideal_timing_weight(mu, nmax=60):
    """SUM_{n>=1} Pois(n; mu) / sigma_SKI(n)^2: every photon direct, digitiser only."""
    from scipy.stats import poisson
    n = np.arange(1, nmax + 1)
    pn = poisson.pmf(n[None, :], np.clip(mu, 1e-9, None)[:, None])
    return pn @ (1.0 / sigma_t_ski(n) ** 2)


def bound(model, particle, vertex, direction, p, *, timing="measured", shower=True, floor=True):
    """The Cramer-Rao bound for one event: a dict of covariances and 1-sigma widths.
    With `floor` (default) and a model that carries floors, the physics floors are included;
    cov_pre is the covariance before the additive floors (light-scale nuisances included)."""
    Iq, It, aux = fisher(model, particle, vertex, direction, p, timing=timing)
    I = Iq + It
    fl = model.floors.get(particle) if floor and getattr(model, "floors", None) else None
    sig_sh = shower_sigma(particle, p) if shower else 0.0
    if fl is not None and fl.sigmas(p)["yld"] > 0:
        sig_sh = float(np.hypot(sig_sh, fl.sigmas(p)["yld"]))
    if sig_sh > 0:
        # Append the light scale xi: d mu_j / d xi = mu_j (dark noise excluded),
        # prior information 1/sigma^2, then marginalise by inverting the 8x8.
        light = model.light[particle]
        sig = aux["mu"] - light.mu_dark
        col = np.einsum("aj,j,j->a", aux["dmu"], sig, aux["wq"])
        corner = np.sum(sig * sig * aux["wq"]) + 1.0 / sig_sh ** 2
        I8 = np.zeros((8, 8))
        I8[:7, :7] = I
        I8[:7, 7] = I8[7, :7] = col
        I8[7, 7] = corner
        cov = _inverse(I8)[:7, :7]
    else:
        cov = _inverse(I)
    d, e1, e2 = aux["d"], aux["e1"], aux["e2"]
    cov_pre = cov
    if fl is not None:
        from ..io.geometry import towall
        tw = float(towall(*vertex, *d))
        cov = cov + fl.covariance(d, e1, e2, p, towall=tw)
    rot = np.stack([d, e1, e2])
    cov_track = rot @ cov[1:4, 1:4] @ rot.T           # (along, perp1, perp2)
    return dict(
        cov=cov, cov_pre=cov_pre, cov_track=cov_track, I=I, I_charge=Iq, I_time=It,
        floor=fl.sigmas(p) if fl is not None else None,
        sigma_t=float(np.sqrt(cov[0, 0])),
        sigma_long=float(np.sqrt(cov_track[0, 0])),
        sigma_perp=float(np.sqrt(0.5 * (cov_track[1, 1] + cov_track[2, 2]))),
        sigma_angle=float(np.sqrt(cov[4, 4] + cov[5, 5])),   # rad, quadrature of tilts
        sigma_p_rel=float(np.sqrt(cov[6, 6]) / p),
        diag=1.0 / np.sqrt(np.diag(I)),
        npe=aux["npe"], nhit=aux["nhit"], shower_sigma=sig_sh,
        charge_fraction=np.diag(Iq) / np.maximum(np.diag(I), 1e-300))


def _inverse(I):
    """Inverse after scaling to unit diagonal: the parameters differ by ~1e8 in
    natural units, which alone would make a raw inverse ill-conditioned."""
    s = np.sqrt(np.diag(I))
    return np.linalg.inv(I / np.outer(s, s)) / np.outer(s, s)
