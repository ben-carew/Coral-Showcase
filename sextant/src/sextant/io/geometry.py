"""Inner-detector geometry and the two distance-to-wall variables.

`dwall` and `towall` are different quantities and resolution behaves differently
against each, so both are provided and named apart:

    dwall   shortest distance from the vertex to any wall.
    towall  distance from the vertex to the wall *along the particle direction*
            -- how much detector the ring has to develop in.

The existing notebook computes `min(R - r, Z - |z|)` and labels it "to-wall",
which is dwall. Anything reproducing those plots should ask for `dwall`.
"""

from __future__ import annotations

import numpy as np

#: HyperK HybridmPMT_WithOD_Realistic inner detector, in cm: the PMT surfaces
#: (max PMT radius and |z| in the geometry tree), the wall Coral's corrections use.
DEFAULT_RADIUS_CM = 3242.96
DEFAULT_HALF_HEIGHT_CM = 3296.47


def dwall(x, y, z, radius=DEFAULT_RADIUS_CM, half_height=DEFAULT_HALF_HEIGHT_CM):
    """Shortest distance from a point to the detector wall (negative outside)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    r = np.hypot(x, y)
    return np.minimum(radius - r, half_height - np.abs(z))


def towall(x, y, z, dx, dy, dz,
           radius=DEFAULT_RADIUS_CM, half_height=DEFAULT_HALF_HEIGHT_CM):
    """Distance from a point to the wall along a direction.

    Solves for the smallest positive intersection with the barrel cylinder and
    the two end caps. A direction of exactly zero (a failed fit, or a truth ring
    with no direction) gives NaN rather than an arbitrary large number.
    """
    x, y, z, dx, dy, dz = (np.asarray(v, float) for v in (x, y, z, dx, dy, dz))
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy, uz = dx / norm, dy / norm, dz / norm

    # Barrel: |(x,y) + t (ux,uy)| = radius  ->  a t^2 + b t + c = 0
    a = ux ** 2 + uy ** 2
    b = 2.0 * (x * ux + y * uy)
    c = x ** 2 + y ** 2 - radius ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = b ** 2 - 4.0 * a * c
        sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
        t_barrel = np.where(a > 0, (-b + sqrt_disc) / (2.0 * a), np.inf)
    t_barrel = np.where(disc >= 0, t_barrel, np.inf)

    # Caps: z + t uz = +/- half_height
    with np.errstate(invalid="ignore", divide="ignore"):
        t_top = np.where(uz > 0, (half_height - z) / uz, np.inf)
        t_bottom = np.where(uz < 0, (-half_height - z) / uz, np.inf)

    t = np.minimum(t_barrel, np.minimum(t_top, t_bottom))
    t = np.where(np.isfinite(t) & (t > 0), t, np.nan)
    return np.where(norm > 0, t, np.nan)
