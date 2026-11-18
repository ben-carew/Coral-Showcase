"""fiTQun's Cherenkov-profile tuning, evaluated smoothly.

The CProfile tuning (`CProf_{11,13}_fit_WCSim.root`) is the direct-light model both
fitters use: for a vertex a distance R0 from a PMT seen at cos(theta0) from the track
axis, three integrals I_n(R0, cos theta0; p) of the emission profile over the track
combine with the PMT acceptance at three points along the track (see `model.py`).

Each (R0, cos theta0) bin stores piecewise polynomials in ln p. Both fitters read the
table by NEAREST BIN: fiTQun's bilinear interpolation (`fiTQun_shared::EvalIn`) is
commented out for the fitted tables, and Coral's `_get_integrals_core` copies that.
The predicted charge is therefore a staircase in angle with steps of 0.04 in
cos theta0 (~3.4 deg at the ring), and flat between them -- which is harmless for a
derivative-free minimiser and fatal for a derivative. A Fisher matrix is built from
derivatives, so here the table is interpolated bilinearly in (R0, cos theta0) from
bin centres. At a bin centre the value is exactly the table's.
"""

from __future__ import annotations

import numpy as np


class CherenkovProfile:
    """The three profile integrals for one particle type, smoothly interpolated."""

    def __init__(self, npars, pars, nsect, r0_edges, th0_edges,
                 nphot_pars, maxd_pars, logp_range):
        self.npars = int(npars)
        self.pars = np.asarray(pars, float)          # (3, nR0, nth0, ncoef)
        self.nsect = np.asarray(nsect, int)           # (3, nR0, nth0)
        self.r0_edges = np.asarray(r0_edges, float)
        self.th0_edges = np.asarray(th0_edges, float)
        self.r0_ctr = 0.5 * (self.r0_edges[1:] + self.r0_edges[:-1])
        self.th0_ctr = 0.5 * (self.th0_edges[1:] + self.th0_edges[:-1])
        self.nphot_pars = np.asarray(nphot_pars, float)
        self.maxd_pars = np.asarray(maxd_pars, float)
        self.logp_range = tuple(logp_range)
        self._cache_logp = None
        self._cache_table = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_root(cls, path):
        import uproot
        with uproot.open(path) as f:
            npars = int(f["hprofinf"].values()[0])
            pars, nsect = [], []
            for n in range(3):
                p, r0e, th0e, _ = f[f"hI3d_par_{n}"].to_numpy()
                s, _, _ = f[f"hI3d_nsect_{n}"].to_numpy()
                pars.append(p)
                nsect.append(np.rint(s).astype(int))
            _, _, _, pedges = f["I_0"].to_numpy()
            nphot = f["gNphot_pars"].to_numpy()[0][:-2]
            maxd = f["gsthr_pars"].to_numpy()[0][:-2]
        logp = np.log(0.5 * (pedges[1:] + pedges[:-1]))
        return cls(npars, np.array(pars), np.array(nsect), r0e, th0e,
                   nphot, maxd, (logp[0], logp[-1]))

    def to_arrays(self, prefix):
        return {f"{prefix}_npars": np.array(self.npars),
                f"{prefix}_pars": self.pars.astype(np.float64),
                f"{prefix}_nsect": self.nsect.astype(np.int16),
                f"{prefix}_r0_edges": self.r0_edges,
                f"{prefix}_th0_edges": self.th0_edges,
                f"{prefix}_nphot_pars": self.nphot_pars,
                f"{prefix}_maxd_pars": self.maxd_pars,
                f"{prefix}_logp_range": np.array(self.logp_range)}

    @classmethod
    def from_arrays(cls, arrays, prefix):
        g = lambda k: arrays[f"{prefix}_{k}"]
        return cls(int(g("npars")), g("pars"), g("nsect"), g("r0_edges"),
                   g("th0_edges"), g("nphot_pars"), g("maxd_pars"), g("logp_range"))

    # -- 1-D momentum functions -------------------------------------------
    def n_photons(self, p):
        """Cherenkov photons emitted by a track of momentum p (MeV/c)."""
        return np.exp(np.polyval(self.nphot_pars[::-1], np.log(p)))

    def track_length(self, p):
        """fiTQun's `maxd`: the length over which the track emits, cm."""
        return np.exp(np.polyval(self.maxd_pars[::-1], np.log(p)))

    # -- the integrals -------------------------------------------------------
    def table(self, p):
        """I_n at every (R0, cos theta0) bin centre for one momentum: (3, nR0, nth0)."""
        logp = float(np.log(p))
        if self._cache_logp == logp:
            return self._cache_table
        npars = self.npars
        ns = self.nsect                                       # (3, nR0, nth0)
        smax = int(ns.max())
        # Section boundaries sit after the coefficients: pars[..., ns*npars + k].
        k = np.arange(smax + 1)
        idx = ns[..., None] * npars + np.minimum(k, ns[..., None])
        bounds = np.take_along_axis(self.pars, idx, axis=-1)  # (3,nR0,nth0,smax+1)
        valid = k[None, None, None, :] <= ns[..., None]
        bounds = np.where(valid, bounds, np.inf)
        # Section index exactly as the fitters choose it: clip below the first
        # boundary to section 0 and above the last to the final section.
        sec = (logp >= bounds[..., 1:]).sum(axis=-1)
        sec = np.minimum(sec, ns - 1)
        powers = logp ** np.arange(npars)
        cidx = sec[..., None] * npars + np.arange(npars)
        coefs = np.take_along_axis(self.pars, cidx, axis=-1)
        out = coefs @ powers
        self._cache_logp, self._cache_table = logp, out
        return out

    def integrals(self, p, r0, costh0):
        """Bilinearly interpolated I_n for arrays of (R0, cos theta0): (3, N)."""
        tab = self.table(p)
        x = _frac_index(r0, self.r0_ctr)
        y = _frac_index(costh0, self.th0_ctr)
        i0 = np.floor(x).astype(int)
        j0 = np.floor(y).astype(int)
        fx = x - i0
        fy = y - j0
        i1 = i0 + 1
        j1 = j0 + 1
        return ((1 - fx) * (1 - fy) * tab[:, i0, j0] + fx * (1 - fy) * tab[:, i1, j0]
                + (1 - fx) * fy * tab[:, i0, j1] + fx * fy * tab[:, i1, j1])


def _frac_index(values, centres):
    """Fractional index into equally spaced bin centres, clamped to the table."""
    step = centres[1] - centres[0]
    x = (np.asarray(values, float) - centres[0]) / step
    return np.clip(x, 0.0, len(centres) - 1 - 1e-9)
