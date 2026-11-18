#!/usr/bin/env python
"""One verdict for a paired A/B of reconstruction arms (R79; supersedes the r15 / r26 / r28 / r40 / r55 / r59n pair scripts).

THE RULE (CLAUDE.md, confirmed by bcarew 11-14). ONE primary metric, named in the item BEFORE the run, judged on its
95% paired-bootstrap interval against a bar; every other metric is a GUARD that is broken only when its 99% interval
excludes "not worse". With ~14 guards that is a 13% family-wise false alarm; "no metric worse at 95%" was 51%.

What is printed for the primary: the estimate, both intervals, and two separate statements, because "at least X%"
can be read two ways and has been (R90):
  MET          the estimate reaches the bar AND the 95% interval excludes "no change";
  ESTABLISHED  the 95% interval itself lies beyond the bar.

Tables: (1) every shared event (failures included: what an analysis gets), (2) events converged (< 20 deg from truth,
true p > 100 MeV/c) in EVERY arm (the widths the items quote). Widths are ratios arm / reference; failures and PID are
differences in points. Momentum is given as half-widths (the most powerful paired statistics) AND as a Gaussian core
(Sextant's binned fit): a change that moves the half-width and not the core sits in the shoulders, not the peak.

Guards against the traps that produced wrong verdicts (method-traps): the join (a merge not marked entry-numbered is
refused unless --allow-unnumbered), dead switches (share of bit-identical fits, key-level diff of the frozen configs),
time (seconds from ordinary cluster runs are never printed: evaluation counts only; time bars need the `bench` skill).

    python paired_verdict.py --exp r76_val --sample-e elneg-r1 --ref base --arms s100f s100t --primary e.trans68 --bar -5
    python paired_verdict.py --e base=PATH new=PATH --truth-e PATH [--mu base=PATH new=PATH --truth-mu PATH] \\
                             --ref base --primary e.p_core_pbins_bulk --bar -5 [--also fiTQun=PATH] [--fv 200]
Metrics: <sample>.<name>, sample e | mu; names vtx68 trans68 along68 ang68 p_iqr68 p_core p_core_pbins
p_core_pbins_bulk p_h68dt, p_core_pbins_far p_h68dt_far (true towall > 40 m) (converged table), the same with _all (every shared event: vtx68 trans68 ang68 p_iqr68),
fail fail_p100 (pp; every event, true p > 100 MeV/c), pid.flat pid.lin pid.lin_p200 (mu-eff at 99% e-eff, points; needs both samples and both hypotheses).
"""
import argparse
import configparser
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Software/Sextant/src"))
sys.path.insert(0, str(ROOT / "Analysis/Scripts/cluster"))
from sextant import Arm, Truth, analysis                      # noqa: E402
from sextant.metrics.stats import gauss_sigma                 # noqa: E402
import runlib                                                 # noqa: E402

def core_sigma(x):
    """Gaussian core: Sextant's binned fit (iterated +-2.5 sigma window). An unbinned truncated-sd core was tried for
    R79 and REJECTED by its own acceptance test (11-15): 1.3x more precise on ONE sample, but in a paired ratio its hard
    window makes it 1.9x noisier when most fits are shared (r75) and no better when none are (R90). The half-widths
    (p_iqr68, p_h68dt) are the most powerful paired statistics; the core says where in the distribution a change sits."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return gauss_sigma(x) if len(x) >= 30 else np.nan


def half68(x):
    lo, hi = np.percentile(x, [16, 84])
    return (hi - lo)/2


def numbering_ok(path, production, sample):
    """True when the file's event ids can be joined to truth: the merge says they are tree entries (R89), the file was
    renumbered by hand, it is a fiTQun merge (numbered by entry), or none of its input files skipped an event number."""
    if str(path).endswith("_renum.root") or "fitqun" in Path(path).name:
        return True
    try:
        with uproot.open(path) as f:
            if "numbering" in f:
                return bool(f["numbering"]["event_is_entry"].array(library="np")[0])
            files = np.unique(f["fits"]["event"].array(library="np") // runlib.event_stride(production))
    except Exception:
        return False
    return all(runlib.numbering_scanned(production, sample, i) and runlib.entry_lookup(production, sample, i) is None
               for i in files)


def load(paths, truth_path, own, allow_unnumbered, production, sample):
    truth = Truth.from_file(truth_path)
    R, other = {}, {}
    for name, path in paths.items():
        if not numbering_ok(path, production, sample) and not allow_unnumbered:
            sys.exit(f"REFUSED: {path} is not marked entry-numbered. In a WCSim file that skipped an event number its "
                     f"events join the wrong truth (R89). Run Analysis/Scripts/evtnum/coral_renumber.py or re-merge; "
                     f"--allow-unnumbered only for samples known to have no gap.")
        arm = Arm.from_file(name, path)
        R[name] = analysis.compute_residuals([arm], truth, hypothesis=own)[name].set_index("event")
        try:
            alt = "mu-" if own == "e-" else "e-"
            other[name] = analysis.compute_residuals([arm], truth, hypothesis=alt)[name].set_index("event")
        except Exception:
            other[name] = None
    return R, other


class Sample:
    """The residuals of every arm on one sample, on the events all of them share."""

    def __init__(self, tag, R, other, judged, fv, boot, seed):
        self.tag, self.judged = tag, judged
        ev = None
        for name in R:
            ev = R[name].index if ev is None else ev.intersection(R[name].index)
        ev = ev[~ev.duplicated()]
        self.ev = ev
        self.R = {a: r.loc[ev] for a, r in R.items()}
        self.other = {a: (o.reindex(ev) if o is not None else None) for a, o in other.items()}
        g = self.R[next(iter(R))]
        self.pt, self.dw, self.tw = g.p_true.to_numpy(), g.dwall.to_numpy(), g.towall.to_numpy()
        self.sel = np.ones(len(ev), bool) if fv is None else (self.dw > fv)
        self.E = {a: r.p_res_pct.to_numpy() for a, r in self.R.items()}
        self.V = {a: r.vertex_dist.to_numpy() for a, r in self.R.items()}
        self.T = {a: r.vertex_trans.to_numpy() for a, r in self.R.items()}
        self.L = {a: r.vertex_long.to_numpy() for a, r in self.R.items()}
        self.A = {a: r.angle_deg.to_numpy() for a, r in self.R.items()}
        geo = np.all([np.isfinite(self.A[a]) for a in judged], axis=0)
        ok = geo & np.all([np.isfinite(self.E[a]) for a in judged], axis=0)      # the momentum residual needs true p > 100 MeV/c
        self.geo = np.flatnonzero(self.sel & geo)                                # every shared event: failures, vertex, angle
        self.all = np.flatnonzero(self.sel & ok)
        conv = np.all([self.A[a] < 20 for a in judged], axis=0) & (self.pt > 100) & ok
        self.conv = np.flatnonzero(self.sel & conv)
        self.bulk = np.flatnonzero(self.sel & conv & (self.tw > 1500))
        self.far = np.flatnonzero(self.sel & conv & (self.tw > 4000))
        rng = np.random.default_rng(seed)
        self.boot = {k: [rng.choice(i, len(i)) for _ in range(boot)] for k, i in
                     (("geo", self.geo), ("all", self.all), ("conv", self.conv), ("bulk", self.bulk), ("far", self.far))}
        edges = np.percentile(self.pt[self.conv], np.linspace(0, 100, 7))
        edges[-1] += 1e-6
        self.pbin = np.digitize(self.pt, edges) - 1

    # statistics: f(arm, idx) -> number
    def scaled(self, e):
        return 1 + np.median(e)/100

    def p_iqr68(self, a, i):
        return half68(self.E[a][i])/self.scaled(self.E[a][i])

    def p_core(self, a, i):
        return core_sigma(self.E[a][i])/self.scaled(self.E[a][i])

    def p_core_pbins(self, a, i):
        e, b = self.E[a][i], self.pbin[i]
        return math.sqrt(np.nanmean([core_sigma(e[b == k])**2 for k in range(6)]))

    def p_h68dt(self, a, i):
        e = self.E[a][i]
        return half68(e - pd.Series(e).groupby(np.floor(self.pt[i]/25).astype(int)).transform("median").to_numpy())

    def metrics(self):
        p68 = lambda X: (lambda a, i: np.percentile(X[a][i], 68))
        m = {"vtx68": ("conv", p68(self.V)), "trans68": ("conv", p68(self.T)),
             "along68": ("conv", lambda a, i: half68(self.L[a][i])), "ang68": ("conv", p68(self.A)),
             "p_iqr68": ("conv", self.p_iqr68), "p_core": ("conv", self.p_core),
             "p_core_pbins": ("conv", self.p_core_pbins), "p_core_pbins_bulk": ("bulk", self.p_core_pbins),
             "p_h68dt": ("conv", self.p_h68dt),
             "p_core_pbins_far": ("far", self.p_core_pbins), "p_h68dt_far": ("far", self.p_h68dt),
             "vtx68_all": ("geo", p68(self.V)), "trans68_all": ("geo", p68(self.T)), "ang68_all": ("geo", p68(self.A)),
             "p_iqr68_all": ("all", self.p_iqr68)}
        return m


def interval(bs):
    return [float(v) for v in np.percentile(bs, [2.5, 97.5, 0.5, 99.5])]


def ratio_rows(S, arm, ref):
    rows = {}
    for name, (pool, f) in S.metrics().items():
        idx = getattr(S, pool)
        if len(idx) < 50:
            continue
        r0, a0 = f(ref, idx), f(arm, idx)
        bs = [100*(f(arm, i)/f(ref, i) - 1) for i in S.boot[pool]]
        rows[f"{S.tag}.{name}"] = dict(kind="ratio", ref=float(r0), arm=float(a0), est=float(100*(a0/r0 - 1)), ci=interval(bs),
                                       n=int(len(idx)))
    fa, fr = (S.A[arm] > 20), (S.A[ref] > 20)
    for name, pool in (("fail", S.geo), ("fail_p100", S.all)):
        bs = [100*(fa[i].mean() - fr[i].mean()) for i in S.boot["geo" if name == "fail" else "all"]]
        rows[f"{S.tag}.{name}"] = dict(kind="diff", ref=float(100*fr[pool].mean()), arm=float(100*fa[pool].mean()),
                                       est=float(100*(fa[pool].mean() - fr[pool].mean())), ci=interval(bs), n=int(len(pool)))
    return rows


def pid_table(Se, Sm, arm, ref, boot, seed):
    """mu-eff at 99% e-eff in points: flat cut (re-derived per joint resample), linear cut in p_e (slope and
    threshold cross-fitted on file parity, muon bootstrap), and the linear cut for true p_mu > 200 MeV/c."""
    def table(S, a):
        own, oth = S.R[a], S.other[a]
        if oth is None:
            return None
        e, m = (own, oth) if S.tag == "e" else (oth, own)
        d = pd.DataFrame({"dnll": m.nll.to_numpy() - e.nll.to_numpy(), "pe": e.p.to_numpy(), "pt": S.pt,
                          "par": (S.ev.to_numpy() // 100) % 2}, index=S.ev)
        return d[S.sel & np.isfinite(d.dnll.to_numpy()) & np.isfinite(d.pe.to_numpy())]
    T = {a: (table(Se, a), table(Sm, a)) for a in (arm, ref)}
    if any(t is None for pair in T.values() for t in pair):
        return {}
    ce = T[arm][0].index.intersection(T[ref][0].index)
    cm = T[arm][1].index.intersection(T[ref][1].index)
    T = {a: (e.loc[ce], m.loc[cm]) for a, (e, m) in T.items()}
    rng = np.random.default_rng(seed)
    ne, nm = len(ce), len(cm)
    rows = {}
    eff = lambda de, dm: 100*np.mean(dm <= np.percentile(de, 1.0))
    d0 = {a: eff(T[a][0].dnll.to_numpy(), T[a][1].dnll.to_numpy()) for a in T}
    bs = []
    for _ in range(boot):
        ie, im = rng.integers(0, ne, ne), rng.integers(0, nm, nm)
        bs.append(eff(T[arm][0].dnll.to_numpy()[ie], T[arm][1].dnll.to_numpy()[im])
                  - eff(T[ref][0].dnll.to_numpy()[ie], T[ref][1].dnll.to_numpy()[im]))
    rows["pid.flat"] = dict(kind="gain", ref=d0[ref], arm=d0[arm], est=d0[arm] - d0[ref], ci=interval(bs), n=nm)
    ok = {}
    for a, (e, m) in T.items():
        parts = []
        for cal, sc in ((0, 1), (1, 0)):
            E, M, ms = e[e.par == cal], m[m.par == cal], m[m.par == sc]
            def lin(b):
                t = np.percentile(E.dnll - b*E.pe, 1.0)
                return ((M.dnll - b*M.pe) <= t).mean(), t
            b = max(np.linspace(0, 0.8, 41), key=lambda b: lin(b)[0])
            parts.append(pd.Series(((ms.dnll - b*ms.pe) <= lin(b)[1]).to_numpy(), index=ms.index))
        ok[a] = pd.concat(parts).reindex(cm).to_numpy().astype(float)
    pt = T[ref][1].pt.to_numpy()
    for name, s in (("pid.lin", np.ones(nm, bool)), ("pid.lin_p200", pt > 200)):
        d = (ok[arm] - ok[ref])[s]
        bs = [100*d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)]
        rows[name] = dict(kind="gain", ref=float(100*ok[ref][s].mean()), arm=float(100*ok[arm][s].mean()),
                          est=float(100*d.mean()), ci=interval(bs), n=int(s.sum()))
    return rows


def config_diff(exp, smp, arm, ref):
    """Key-level diff of the two frozen configs, when the run directories were pulled."""
    out = []
    cfg = {}
    for a in (arm, ref):
        hits = sorted((ROOT / f"Output/Coral/Jobs/{exp}_{smp}/{a}").glob("config*.ini")) if exp else []
        if not hits:
            return None
        c = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
        c.optionxform = str
        c.read(hits[0])
        cfg[a] = {f"{s}.{k}": v.strip() for s in c.sections() for k, v in c[s].items()}
    for k in sorted(set(cfg[arm]) | set(cfg[ref])):
        if cfg[arm].get(k) != cfg[ref].get(k):
            out.append(f"{k}: {cfg[ref].get(k, '<absent>')} -> {cfg[arm].get(k, '<absent>')}")
    return out


def evaluations(exp, smp, sample, a):
    p = ROOT / f"Output/Coral/Timing/{exp}_{smp}/coral_{sample}_{a}.csv" if exp else None
    if p is None or not p.exists():
        return None
    c = pd.read_csv(p)
    return float(c.nll_evals.mean()) if "nll_evals" in c else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", help="experiment prefix: arms are Output/Coral/Reco/<exp>_{e,mu}/coral_<sample>_<arm>.root")
    ap.add_argument("--production", default="v1.12.29")
    ap.add_argument("--sample-e", default="electron")
    ap.add_argument("--sample-mu", default="muon")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--e", nargs="*", default=[], metavar="ARM=PATH")
    ap.add_argument("--mu", nargs="*", default=[], metavar="ARM=PATH")
    ap.add_argument("--truth-e")
    ap.add_argument("--truth-mu")
    ap.add_argument("--also", nargs="*", default=[], metavar="NAME=PATH_E[,PATH_MU]", help="reference-only columns (fiTQun)")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--primary", required=True, help="e.g. e.trans68 (named in the item before the run)")
    ap.add_argument("--bar", type=float, required=True, help="percent (ratios) or points; negative = lower is the goal")
    ap.add_argument("--fv", type=float, default=None, help="true dwall > this [cm] for every number")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--allow-unnumbered", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    paths = {"e": dict(x.split("=", 1) for x in a.e), "mu": dict(x.split("=", 1) for x in a.mu)}
    samples = {"e": a.sample_e, "mu": a.sample_mu}
    if a.exp:
        for smp in ("e", "mu"):
            for arm in [a.ref] + a.arms:
                p = ROOT / f"Output/Coral/Reco/{a.exp}_{smp}/coral_{samples[smp]}_{arm}.root"
                if p.exists():
                    paths[smp].setdefault(arm, str(p))
    judged = [k for k in paths["e"] or paths["mu"] if k != a.ref]
    also = {}
    for x in a.also:
        name, p = x.split("=", 1)
        pe, _, pm = p.partition(",")
        also[name] = {"e": pe, "mu": pm or None}
    truth = {"e": a.truth_e or str(ROOT / f"Output/Truth/truth_{a.sample_e}_v1.12.29.root"),
             "mu": a.truth_mu or str(ROOT / f"Output/Truth/truth_{a.sample_mu}_v1.12.29.root")}

    S = {}
    for k, (smp, own) in enumerate((("e", "e-"), ("mu", "mu-"))):
        if a.ref not in paths[smp]:
            continue
        allp = dict(paths[smp])
        allp.update({n: p[smp] for n, p in also.items() if p[smp]})
        R, other = load(allp, truth[smp], own, a.allow_unnumbered, a.production, samples[smp])
        S[smp] = Sample(smp, R, other, [a.ref] + [j for j in judged if j in paths[smp]], a.fv, a.boot, 79 + k)
        s = S[smp]
        print(f"[{smp}] {len(s.ev)} shared events" + (f", FV dwall > {a.fv:.0f} cm" if a.fv else "")
              + f": {len(s.geo)} judged ({len(s.all)} with true p > 100 MeV/c); converged in every arm {len(s.conv)}; "
              f"bulk (towall > 15 m) {len(s.bulk)}")

    out = {"primary": a.primary, "bar": a.bar, "ref": a.ref, "arms": {}}
    lower_better = a.bar <= 0
    for arm in judged:
        rows = {}
        for smp, s in S.items():
            if arm in s.R:
                rows.update(ratio_rows(s, arm, a.ref))
        if "e" in S and "mu" in S and arm in S["e"].R and arm in S["mu"].R:
            rows.update(pid_table(S["e"], S["mu"], arm, a.ref, a.boot, 7979))
        print(f"\n=== {arm} vs {a.ref} ===")
        for smp, s in S.items():
            if arm not in s.R:
                continue
            same = np.mean(s.R[arm].p.to_numpy() == s.R[a.ref].p.to_numpy())
            line = f"[{smp}] bit-identical fits {100*same:.1f}%"
            if same > 0.98:
                line += "  <-- DEAD SWITCH? the arms should differ"
            ev = [evaluations(a.exp, smp, samples[smp], x) for x in (arm, a.ref)]
            if all(ev):
                line += f"; likelihood evaluations {ev[0]/ev[1]:.3f}x (seconds are not comparable between cluster arms: bench skill)"
            print(line)
            d = config_diff(a.exp, smp, arm, a.ref)
            if d is not None:
                print(f"[{smp}] frozen config, {len(d)} key(s) differ:" + "".join(f"\n      {x}" for x in d[:25]))
        print(f"{'metric':24s} {'n':>6s} {'ref':>8s} {'arm':>8s} {'change':>8s}  95% interval       99% interval       verdict")
        verdicts = {}
        for name, r in rows.items():
            lo95, hi95, lo99, hi99 = r["ci"]
            unit = "%" if r["kind"] == "ratio" else "pt"
            better_is_negative = r["kind"] != "gain"
            if name == a.primary:
                e, lo, hi = (r["est"], lo95, hi95) if lower_better else (-r["est"], -hi95, -lo95)
                bar = a.bar if lower_better else -a.bar
                met = e <= bar and hi < 0
                est = hi <= bar
                v = "PRIMARY " + ("MET" if met else "NOT MET") + (", ESTABLISHED" if est else ", not established at 95%" if met else "")
                out["arms"].setdefault(arm, {})["primary_met"] = bool(met)
                out["arms"][arm]["primary_established"] = bool(est)
            else:
                worse = (lo99 > 0) if better_is_negative else (hi99 < 0)
                v = "guard BROKEN (99%)" if worse else "ok"
                if worse:
                    out["arms"].setdefault(arm, {}).setdefault("guards_broken", []).append(name)
            verdicts[name] = v
            print(f"{name:24s} {r['n']:6d} {r['ref']:8.3f} {r['arm']:8.3f} {r['est']:+7.2f}{unit:2s} [{lo95:+6.2f},{hi95:+6.2f}]    "
                  f"[{lo99:+6.2f},{hi99:+6.2f}]    {v}")
        for smp, s in S.items():
            if arm in s.R:
                b = {x: np.median(s.E[x][s.conv]) for x in (a.ref, arm)}
                al = {x: np.median(s.L[x][s.conv]) for x in (a.ref, arm)}
                print(f"[{smp}] median p bias {b[a.ref]:+.2f} -> {b[arm]:+.2f}%; along-track vertex {al[a.ref]:+.1f} -> {al[arm]:+.1f} cm")
        if a.primary not in rows:
            print(f"PRIMARY {a.primary} was not computed (available: {', '.join(rows)})")
        out["arms"].setdefault(arm, {})["rows"] = rows
        broken = out["arms"][arm].get("guards_broken", [])
        print(f"--> {arm}: primary {'MET' if out['arms'][arm].get('primary_met') else 'NOT MET'}"
              f"{' and established' if out['arms'][arm].get('primary_established') else ''}; "
              f"{len(broken)} guard(s) broken{': ' + ', '.join(broken) if broken else ''}")
    for name in also:
        print(f"\nreference column {name} (same events, not judged):")
        for smp, s in S.items():
            if name in s.R:
                vals = {m: f(name, getattr(s, pool)) for m, (pool, f) in s.metrics().items() if len(getattr(s, pool)) >= 50}
                refv = {m: f(a.ref, getattr(s, pool)) for m, (pool, f) in s.metrics().items() if len(getattr(s, pool)) >= 50}
                print(f"[{smp}] " + "  ".join(f"{m} {vals[m]:.3f} ({a.ref} {refv[m]:.3f})" for m in
                                               ("vtx68", "trans68", "ang68", "p_iqr68", "p_core_pbins", "p_core_pbins_bulk") if m in vals)
                      + f"  fail {100*np.mean(s.A[name][s.geo] > 20):.2f}%")
    if a.out or a.exp:
        p = Path(a.out) if a.out else ROOT / f"Output/Sextant/{a.exp}/verdict.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
