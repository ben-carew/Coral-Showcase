#!/usr/bin/env python
"""Split a merged sub-event-0 hit-set CSV into <nparts> npz files (event % nparts) so fit jobs read only their part.

    python r48_split.py --hits hit_sets.csv --nparts 200 --outdir parts/
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--hits", required=True); ap.add_argument("--nparts", type=int, required=True)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
h = pd.read_csv(a.hits, on_bad_lines="skip", dtype={"event": "int64", "sub_event": "int16", "tubeid": "int32",
                                                     "q": "float32", "t": "float32"})
h = h[h.sub_event == 0].drop(columns="sub_event")
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
for k, g in h.groupby(h.event % a.nparts):
    np.savez(out/f"hits_{k}.npz", **{c: g[c].to_numpy() for c in g.columns})
print(len(h), "rows,", h.event.nunique(), "events ->", out)
