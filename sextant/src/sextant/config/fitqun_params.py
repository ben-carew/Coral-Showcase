"""Reading a fiTQun parameter-override file.

fiTQun's runtime parameters live in a bespoke `< name = value >` format rather
than an ini, so a small parser earns its place: it lets `sextant check` report
what a params file actually switches on, which is otherwise invisible until the
jobs are running.

That matters more than it sounds. The difference between
`Official_HyperK_2024_wscat` and `Official_HyperK_2024_wscat_1Remu` is whether
the pi0, pi+, multi-ring and sequential multi-ring fits run, and on the HK
detector that is the difference between a few seconds and **~20 minutes per
event** (measured on a 1 GeV electron, not estimated). Choosing the wrong one
for a single-ring comparison does not fail -- it just never finishes.
"""

from __future__ import annotations

import re
from pathlib import Path

#: `< fiTQun.DoMuon1RFit = 1 >  trailing commentary`
PARAM_RE = re.compile(r"<\s*([\w.]+)\s*=\s*([^>]*?)\s*>")

#: Fits that cost real time, with what they add. Everything else is cheap.
EXPENSIVE_FITS = {
    "fiTQun.DoMRFit": "multi-ring fit",
    "fiTQun.DoMoreMRFit": "sequential / simultaneous multi-ring refit",
    "fiTQun.DoPi0Fit": "pi0 two-ring fit",
    "fiTQun.DoPi0Fit2": "pi0 mass-constrained fit",
    "fiTQun.DoPionPlus1RFit": "pi+ single-ring fit",
    "fiTQun.DoMSFit": "multi-segment muon fit",
    "fiTQun.DoPDK_MuGammaFit": "proton-decay mu-gamma fit",
}


def read(path) -> dict:
    """{parameter: value} from a fiTQun override file. Later lines win."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fiTQun parameter file not found: {path}")
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for name, value in PARAM_RE.findall(line):
            values[name] = value.strip()
    return values


def _is_on(value) -> bool:
    try:
        return float(str(value).rstrip(".")) != 0.0
    except (TypeError, ValueError):
        return False


def enabled_fits(values: dict) -> list:
    """The expensive fits this parameter set switches on."""
    return [(name, EXPENSIVE_FITS[name]) for name in EXPENSIVE_FITS
            if _is_on(values.get(name, 0))]


def describe(path) -> dict:
    values = read(path)
    return {"path": str(path), "n_parameters": len(values),
            "expensive_fits": enabled_fits(values), "values": values}
