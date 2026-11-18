#!/usr/bin/env python
"""WCSim files -> one WatChMaL HDF5 (hkfd_jun26 layout) for the HK ResNet-152 models.

Trigger-0 Cherenkov digits only, PMT id 0-based (tube id - 1), raw digit time (the models subtract 950 ns),
raw charge. Truth (vertex, direction, total energy) from the Sextant truth file, joined on the global event number
100*file + EvtNum (as in Coral's output).

    python resnet_h5.py --sample electron --files 700-739 --out /path/events.h5
"""
import argparse, os
import numpy as np, h5py, uproot, ROOT

PROD = "/sps/hyperk/common/hyperk.org/beta-production/prod_sensitvity_1/pgun/wcsim_1.12.29_cwcs1.1/HyperK_HybridmPMT_WithOD_Realistic"
MASS = {"electron": 0.511, "muon": 105.658}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--files", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn/truth_{sample}_v1.12.29.root")
    a = ap.parse_args()
    if os.environ.get("WCSIMROOT"):
        ROOT.gSystem.AddDynamicPath(os.path.join(os.environ["WCSIMROOT"], "lib"))
    ROOT.gSystem.Load("libWCSimRoot")
    tr = uproot.open(a.truth.format(sample=a.sample))["Tracks"].arrays(library="pd")
    tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
    tag = {"electron": "elpos", "muon": "muneg"}[a.sample]  # the registry's electron = e+, muon = mu-
    lo, hi = (int(x) for x in a.files.split("-"))
    ev, idx, pmt, tim, chg, vtx = [], [0], [], [], [], []
    for fi in range(lo, hi + 1):
        f = ROOT.TFile.Open(f"{PROD}/id_{tag}/rwcs/hkfd_id_{tag}_00000000-{fi:04d}_rwcs.root")
        t = f.Get("wcsimT")
        for i in range(t.GetEntries()):
            t.GetEntry(i)
            trig = t.wcsimrootevent.GetTrigger(0)
            n = trig.GetHeader().GetEvtNum()
            d = trig.GetCherenkovDigiHits()
            ev.append(100*fi + n)
            vtx.append([trig.GetVtx(k) for k in range(3)])
            pmt.append(np.fromiter((h.GetTubeId() - 1 for h in d), np.int32))
            tim.append(np.fromiter((h.GetT() for h in d), np.float32))
            chg.append(np.fromiter((h.GetQ() for h in d), np.float32))
            idx.append(idx[-1] + len(pmt[-1]))
        f.Close()
    ev = np.array(ev)
    T = tr.reindex(ev)
    d = T[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float)
    d /= np.linalg.norm(d, axis=1)[:, None]
    with h5py.File(a.out, "w") as h:
        h["event_hits_index"] = np.array(idx[:-1], np.int64)
        h["hit_pmt"] = np.concatenate(pmt); h["hit_time"] = np.concatenate(tim); h["hit_charge"] = np.concatenate(chg)
        h["labels"] = np.full(len(ev), 1 if a.sample == "electron" else 2, np.int32)
        h["positions"] = T[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(np.float32)[:, None, :]
        h["angles"] = np.c_[np.arccos(d[:, 2]), np.arctan2(d[:, 1], d[:, 0])].astype(np.float32)[:, None, :]
        h["energies"] = np.sqrt(T.P.to_numpy()**2 + MASS[a.sample]**2).astype(np.float32)[:, None]
        h["evtnum"] = ev
        h["wcsim_vtx"] = np.array(vtx, np.float32)
    np.savez(a.out.replace(".h5", "_split.npz"), test_idxs=np.arange(len(ev)))
    print(f"{a.out}: {len(ev)} events, {idx[-1]} hits, truth missing {int(T.P.isna().sum())}")


if __name__ == "__main__":
    main()
