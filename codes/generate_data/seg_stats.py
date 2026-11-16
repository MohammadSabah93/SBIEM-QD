"""Segment-interaction statistics of the segmented-fault runs (configuration B), 300 < t <= 1000 yr.

For every barrier width and model: number of events, fraction running through the barrier,
sequences (events within 30 d of each other), their moments, recurrence intervals and the
triggering delay between the first and second event of a sequence. For barrier widths with an
FD run, also the event-by-event agreement of SRD-QD with FD over the whole run.
Writes data/analysis/seg_stats.json; runs from any folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS, RUNS  # noqa: E402
from cycle_analysis import catalogue  # noqa: E402

V2 = RUNS / "v2"
GAPS = (4, 6, 8, 12)
T0, T1 = 300.0, 1000.0
SEQ_DAYS = 30.0
MU = 32.04e9  # shear modulus of the BP1 analogue (Pa), as P.mu in make_figures_v2


def runs(gap):
    out = {"SRD": V2 / f"explore/srd_seg_g{gap}_a0.025.npz", "QD": V2 / f"explore/qd_seg_g{gap}_a0.025.npz",
           "FD": V2 / f"fd_seg_g{gap}.npz"}
    return {k: v for k, v in out.items() if v.exists()}


def sequences(cat):
    seq = []
    for e in cat:
        if seq and (e["t_yr"] - seq[-1][-1]["t_yr"]) * 365.25 < SEQ_DAYS:
            seq[-1].append(e)
        else:
            seq.append([e])
    return seq


def stats(cat):
    ev = [e for e in cat if T0 < e["t_yr"] <= T1]
    seq = [s for s in sequences(cat) if T0 < s[0]["t_yr"] <= T1]
    starts = np.array([s[0]["t_yr"] for s in seq])
    iv = np.diff(starts)
    delays = [(s[1]["t_yr"] - s[0]["t_yr"]) * 365.25 for s in seq if len(s) > 1]
    moments = [sum(e["moment"] for e in s) for s in seq]
    return dict(
        events=len(ev),
        through_going_percent=100.0 * float(np.mean([e["segments"] == [1, 1] for e in ev])) if ev else None,
        sequences=len(seq),
        interval_mean=float(iv.mean()) if iv.size else None,
        interval_halfrange=float(0.5 * (iv.max() - iv.min())) if iv.size else None,
        interval_std=float(iv.std()) if iv.size else None,
        delay_median_days=float(np.median(delays)) if delays else None,
        delay_range_days=[float(min(delays)), float(max(delays))] if delays else None,
        sequence_moment_max=float(max(moments)) if moments else None,
        sequence_moment_mean=float(np.mean(moments)) if moments else None,
        stress_drop_mean=float(np.mean([e["stress_drop"] for e in ev])) if ev else None,
    )


def event_by_event(cat_a, cat_b):
    """Agreement of two catalogues event by event until the ruptured segments first differ."""
    n = 0
    dt, dm, dv = [], [], []
    for a, b in zip(cat_a, cat_b):
        if a["segments"] != b["segments"]:
            break
        n += 1
        dt.append(abs(a["t_yr"] - b["t_yr"]))
        dm.append(abs(a["moment"] / b["moment"] - 1.0))
        dv.append(abs(a["vmax"] / b["vmax"] - 1.0))
    return dict(matched_events=n, matched_until_yr=float(cat_b[n - 1]["t_yr"]) if n else None,
                total_events=[len(cat_a), len(cat_b)],
                max_time_difference_yr=float(max(dt)) if dt else None,
                max_moment_difference=float(max(dm)) if dm else None,
                max_peak_slip_rate_difference=float(max(dv)) if dv else None,
                time_differences_yr=[float(x) for x in dt])


def main():
    out = {}
    for gap in GAPS:
        cats = {k: [e for e in catalogue(str(p)) if e["potency"] > 10e3 and e["t_yr"] <= T1]
                for k, p in runs(gap).items()}
        out[gap] = {k: stats(c) for k, c in cats.items()}
        if "FD" in cats and "SRD" in cats:
            out[gap]["SRD_vs_FD"] = event_by_event(cats["SRD"], cats["FD"])
    (ANALYSIS / "seg_stats.json").write_text(json.dumps(out, indent=1))
    for gap, d in out.items():
        print(f"== gap {gap} km")
        for k, s in d.items():
            if k == "SRD_vs_FD":
                print(f"  SRD vs FD: {s['matched_events']} events matched until {s['matched_until_yr']} yr "
                      f"(dt <= {s['max_time_difference_yr']}, dM <= {s['max_moment_difference']}, dV <= {s['max_peak_slip_rate_difference']})")
            else:
                print(f"  {k:3s}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in s.items()))


if __name__ == "__main__":
    main()