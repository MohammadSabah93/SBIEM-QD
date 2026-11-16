"""Ensemble statistics of the segmented fault (configuration B), 300 < t <= 1000 yr.

Members: the unperturbed runs (seed 0, data/sbiem_runs/v2) and runs whose initial ln(theta) is perturbed by
+-1 % with seeds 1-8 (SRD-QD, standard QD) or 1-2 (FD), and by +-10 % with seeds 11-14 (SRD-QD, standard
QD), in data/sbiem_runs/review. For every member the
statistics of seg_stats.stats; for every model and barrier width the ensemble mean and range of the
fraction of through-going events, the median triggering delay, the mean stress drop and the mean sequence
interval. Also the SRD-QD runs with event thresholds 1e-4 m/s (memory kept longer, data/sbiem_runs/review).
Writes data/analysis/seg_ensemble.json; runs from any folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS, ROOT, RUNS  # noqa: E402
from cycle_analysis import catalogue  # noqa: E402
from seg_stats import GAPS, T1, stats  # noqa: E402

V2, RV = RUNS / "v2", RUNS / "review"
KEYS = ("through_going_percent", "delay_median_days", "stress_drop_mean", "interval_mean", "events")


def members(model, gap):
    if model == "FD":
        out = [V2 / f"fd_seg_g{gap}.npz"] + [RV / f"fd_seg_g{gap}_s{s}.npz" for s in (1, 2)]
    else:
        m = model.lower()
        out = ([V2 / f"explore/{m}_seg_g{gap}_a0.025.npz"] + [RV / f"{m}_seg_g{gap}_s{s}.npz" for s in range(1, 9)]
               + [RV / f"{m}_seg_g{gap}_L{s}.npz" for s in range(11, 15)])  # L: ln theta +- 10 %
    return [p for p in out if p.exists()]


def member_stats(path):
    cat = [e for e in catalogue(str(path)) if e["potency"] > 10e3 and e["t_yr"] <= T1]
    return stats(cat)


def summary(rows):
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
    return out


def main():
    res = {}
    for gap in GAPS:
        res[gap] = {}
        for model in ("FD", "SRD", "QD"):
            rows = [dict(file=p.relative_to(ROOT).as_posix(), **member_stats(p)) for p in members(model, gap)]
            if rows:
                res[gap][model] = dict(members=rows, summary=summary(rows))
        p = RV / f"srd_seg_g{gap}_v1e-4.npz"
        if p.exists():
            res[gap]["SRD_v1e-4"] = dict(members=[dict(file=p.relative_to(ROOT).as_posix(), **member_stats(p))])
    (ANALYSIS / "seg_ensemble.json").write_text(json.dumps(res, indent=1))
    for gap, d in res.items():
        print(f"== barrier {gap} km")
        for model, v in d.items():
            if "summary" in v:
                s = v["summary"]
                txt = ", ".join(f"{k} {s[k]['mean']:.3g} [{s[k]['min']:.3g}, {s[k]['max']:.3g}]" for k in KEYS if k in s)
                print(f"  {model:9s} n={len(v['members'])}: {txt}")
            else:
                r = v["members"][0]
                print(f"  {model:9s}: " + ", ".join(f"{k} {r[k]:.3g}" for k in KEYS if r.get(k) is not None))


if __name__ == "__main__":
    main()