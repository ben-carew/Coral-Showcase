#!/usr/bin/env python
"""R48 step 2: e/mu PID from the learned likelihoods vs Coral's, held-out e- and mu- (files 700-739).

dnLL = nLL(e-) - nLL(mu-) per event (large = mu-like): the network's from its own e- and mu- fits (each seeded at
Coral's fit of that hypothesis), Coral's from its written fits. mu-eff at 99% e-eff, cut re-derived per resample,
electrons and muons resampled jointly (method-traps). Pre-registered full PASS: network not worse beyond CI.

    python r48_pid.py --dir . --coral-e r48_pid_e.root --coral-mu r48_pid_mu.root
"""
import argparse, glob
import numpy as np, pandas as pd, uproot

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="."); ap.add_argument("--coral-e", required=True); ap.add_argument("--coral-mu", required=True)
a = ap.parse_args()
rng = np.random.default_rng(482)


def nn(pattern):
    return pd.concat([pd.read_csv(f) for f in glob.glob(f"{a.dir}/{pattern}")]).drop_duplicates("event").set_index("event").coral_nll


def coral(path):
    f = pd.DataFrame(uproot.open(path)["fits"].arrays(["event", "sub_event", "particle_code", "nL"], library="np"))
    f = f[f.sub_event == 0].drop_duplicates(["event", "particle_code"])
    p = f.pivot(index="event", columns="particle_code", values="nL")
    return p[0] - p[1]


d = {}
for smp, path in (("e", a.coral_e), ("mu", a.coral_mu)):
    nn_d = nn(f"pid_{smp}_e_*.csv") - nn(f"pid_{smp}_mu_*.csv")
    c_d = coral(path)
    ev = nn_d.dropna().index.intersection(c_d.dropna().index)
    d[smp] = (nn_d.loc[ev].to_numpy(), c_d.loc[ev].to_numpy())
    print(f"{smp} sample: {len(ev)} events with both hypotheses in both codes")


def mueff(de, dm):
    return 100*np.mean(dm > np.quantile(de, 0.99))


(ne, ce), (nm, cm) = d["e"], d["mu"]
v_nn, v_c = mueff(ne, nm), mueff(ce, cm)
bs = []
for _ in range(2000):
    ie, im = rng.integers(0, len(ne), len(ne)), rng.integers(0, len(nm), len(nm))
    bs.append(mueff(ne[ie], nm[im]) - mueff(ce[ie], cm[im]))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"mu-eff at 99% e-eff: network {v_nn:.2f}%, Coral {v_c:.2f}%, difference {v_nn - v_c:+.2f} [{lo:+.2f},{hi:+.2f}] pt")
for name, (x, y) in (("e- mis-ID as mu (network dnLL > 0)", (ne, None)), ("mu- mis-ID as e (network dnLL < 0)", (nm, None))):
    pass
print(f"simple sign cut: e- with dnLL>0 network {100*np.mean(ne > 0):.2f}% / Coral {100*np.mean(ce > 0):.2f}%;"
      f" mu- with dnLL<0 network {100*np.mean(nm < 0):.2f}% / Coral {100*np.mean(cm < 0):.2f}%")
print("FULL PASS" if lo > 0 or (lo <= 0 <= hi) else "FAIL (PID worse beyond CI)")
