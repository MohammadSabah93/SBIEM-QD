"""Memory omitted by restarting the FD memory relaxed at an event that follows another one closely.

In the FD reference the history is restarted with the current slip rate at the onset of every event
phase. If an event follows a previous one after a delay t_d, the exact memory still carries the tail of
the previous event: for a slip increment that is short compared with t_d, the omitted dynamic stress of
mode k is
    tau_rem,k = (mu |k| / 2) K(|k| c_s t_d) Delta delta_k,
with K the in-plane kernel (eq. kernel of the paper), K(T) ~ sqrt(2 / (pi T)) sin(T - pi/4) for large T.
This script evaluates tau_rem on the frictional part at the onset of every event of the FD runs of the
segmented fault that starts within 30 d of the end of the previous one, and writes
data/analysis/memory_remnant.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS, RUNS as RUN_DIR  # noqa: E402
from sbiem_bp1 import BP1, Grid, kernel_Q2  # noqa: E402

YEAR = 365.0 * 86400.0
RUNS = {"g6": RUN_DIR / "v2" / "fd_seg_g6.npz", "g4": RUN_DIR / "v2" / "fd_seg_g4.npz"}
SEGS = {"g6": [[-25000.0, -3000.0], [3000.0, 18000.0]], "g4": [[-24000.0, -2000.0], [2000.0, 17000.0]]}


def kernel(T, r):
    """In-plane kernel K(T): derivative of the closed-form Q below T = 600, asymptotic form above."""
    T = np.asarray(T, float)
    out = np.sqrt(2.0 / (np.pi * np.maximum(T, 1e-12))) * np.sin(T - np.pi / 4)
    low = T < 600.0
    if np.any(low):
        h = 0.05
        out[low] = (kernel_Q2(T[low] + h, r) - kernel_Q2(np.maximum(T[low] - h, 0.0), r)) / (
            T[low] + h - np.maximum(T[low] - h, 0.0))
    return out


def main():
    out = {}
    for name, path in RUNS.items():
        d = np.load(path)
        p = BP1(mode="II", segments=SEGS[name], a_gap=0.025)
        g = Grid(dz=float(d["dz"]), lam=float(d["lam"]))
        nf = d["e0_slip0"].size
        i0 = g.Nh - nf // 2
        r = p.cs / p.cd
        rows = []
        n = int(d["n_events"])
        for i in range(1, n):
            t_on = float(d[f"e{i}_t_start"])
            t_prev0, t_prev1 = float(d[f"e{i - 1}_t_start"]), float(d[f"e{i - 1}_t_end"])
            delay = t_on - t_prev1
            if delay > 30 * 86400.0:
                continue
            ds = np.zeros(g.N)
            ds[i0: i0 + nf] = d[f"e{i - 1}_slip1"] - d[f"e{i - 1}_slip0"]
            if np.max(ds) < 0.05:  # previous event negligible (end-of-event re-crossings)
                continue
            dsh = np.fft.rfft(ds)
            td = t_on - 0.5 * (t_prev0 + t_prev1)  # from the middle of the previous event phase
            T = g.k * p.cs * td
            kk = np.zeros_like(g.k)
            kk[1:] = kernel(T[1:], r)
            tau = np.fft.irfft(0.5 * p.mu * g.k * kk * dsh, n=g.N)[i0: i0 + nf]
            arr = d[f"e{i}_arrival"][0]  # arrival of 1e-3 m/s: the hypocentre is the first cell
            z = (np.arange(nf) - nf // 2 + 0.5) * g.dz
            hyp = z[np.nanargmin(arr)] if np.any(np.isfinite(arr)) else np.nan
            near = np.abs(z - hyp) <= 2e3
            rows.append(dict(event=i, t_yr=t_on / YEAR, delay_s=delay, previous_slip_max=float(np.max(ds)),
                             hypocentre_km=float(hyp / 1e3), tau_rem_max_Pa=float(np.max(np.abs(tau))),
                             tau_rem_hypo_Pa=float(np.max(np.abs(tau[near]))) if near.any() else None))
        out[name] = rows
        big = [r_ for r_ in rows if r_["previous_slip_max"] > 0.5]
        print(f"{name}: {len(big)} events after a large event within 30 d")
        for r_ in big:
            print(f"   t {r_['t_yr']:8.2f} yr  delay {r_['delay_s'] / 3600:8.3f} h  hypo {r_['hypocentre_km']:6.1f} km  "
                  f"tau_rem max {r_['tau_rem_max_Pa'] / 1e3:8.2f} kPa, near hypocentre {r_['tau_rem_hypo_Pa'] / 1e3:8.2f} kPa")
    (ANALYSIS / "memory_remnant.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()