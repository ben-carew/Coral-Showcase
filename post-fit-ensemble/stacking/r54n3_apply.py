#!/usr/bin/env python
"""Apply the exported Coral-only stack (r54n3_export.py) to a merged Coral reconstruction made with both output
corrections OFF (vertex-bias curves, R54). Features come from that raw file's sub-event-0 fits of both hypotheses;
each hypothesis' sub-event-0 fit is replaced by its corrected value (every event with both fits, like R54). The
corrected rows are written into --base (default: the raw file itself), e.g. the vertex-bias file from
tn_postprocess.py, so later sub-events keep whatever that file has. The stack replaces the vertex-bias curve and
R54 on sub-event 0: never apply either on top. Numpy only (no sklearn / torch).

    python r54n3_apply.py --reco <raw>.root --truth <Sextant truth> [--nhits <csv event,nhits>] \
        --models <dir with stack_coral_{e,mu}.npz> [--base <x_vb.root>] --out <x_stack.root>
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, uproot
from r54n import fit_block, r54_features, frame
from r54n3 import coral
from r54n3_export import predict

CODE = {"e": 0, "mu": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reco", required=True); ap.add_argument("--truth"); ap.add_argument("--nhits")
    ap.add_argument("--models", required=True); ap.add_argument("--base"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.nhits:
        nh = pd.read_csv(a.nhits).set_index("event")["nhits"]
    else:
        nh = uproot.open(a.truth)["events"].arrays(["event", "nhits_trigger0"], library="pd").set_index("event").nhits_trigger0
    fits = coral(a.reco)
    ev = fits["e"].index.intersection(fits["mu"].index).intersection(nh.index).sort_values()
    new = {}
    for h, o in (("e", "mu"), ("mu", "e")):
        arr = dict(np.load(Path(a.models)/f"stack_coral_{h}.npz"))
        own, oth = fit_block(fits[h].loc[ev], ""), fit_block(fits[o].loc[ev], "")
        X = r54_features(own, oth, nh.loc[ev].to_numpy(float)).replace([np.inf, -np.inf], np.nan)
        P = predict(arr, X[list(arr["features"])].to_numpy(float))
        _, v, d, p, _ = own
        e1, e2 = frame(d)
        vc = v - (P[:, 1, None]*d + P[:, 2, None]*e1 + P[:, 3, None]*e2)
        dc = d + np.radians(P[:, 4, None])*e1 + np.radians(P[:, 5, None])*e2
        new[h] = pd.DataFrame(np.c_[vc, dc/np.linalg.norm(dc, axis=1)[:, None], p*np.exp(-P[:, 0])],
                              index=ev, columns=["x", "y", "z", "px", "py", "pz", "p"])
        print(f"{h}- hypothesis: {len(ev)} events corrected")
    fin = uproot.open(a.base or a.reco)
    out = fin["fits"].arrays(library="pd")
    for h, c in CODE.items():
        rows = out.index[(out.sub_event == 0) & (out.particle_code == c) & out.event.isin(ev)]
        rows = out.loc[rows].drop_duplicates("event").index
        out.loc[rows, ["x", "y", "z", "px", "py", "pz", "p"]] = new[h].loc[out.loc[rows, "event"]].to_numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(a.out) as fo:
        def tree(name, d):
            fo.mktree(name, {k: x.dtype for k, x in d.items()}).extend(d)
        tree("fits", {k: out[k].to_numpy() for k in out.columns})
        for key in fin.keys(cycle=False):
            if key != "fits" and fin[key].classname == "TTree":
                tree(key, fin[key].arrays(library="np"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
