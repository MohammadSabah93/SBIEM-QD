"""Event catalogues of SBIEM earthquake-cycle runs (v2 of the paper).

For every seismic event (peak slip rate above 0.1 m/s somewhere) of a run saved by sbiem_bp1:
start time, hypocentre, rupture extent (cells reached by 0.1 m/s), potency P = int(slip) dz and
moment per unit length mu P over the whole fault, slip-weighted static stress drop
    stress_drop = int(dtau * dslip) dz / int(dslip) dz,   dtau = -(S[slip1] - S[slip0]),
peak slip rate, slip at z = 0, and for segmented faults the segments that ruptured (more than half
of their cells reached by 0.1 m/s). Symmetric runs are mirrored, so P covers -wf < z < wf.

usage: python cycle_analysis.py run.npz [run2.npz ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sbiem_bp1 import YEAR, BP1, Grid, Model, Settings

V_SEIS = 0.1  # m/s, threshold index 2 of the recorder
THRESHOLDS = (1e-3, 1e-2, 1e-1, 1.0)  # arrival thresholds of the recorder (m/s)


def run_model(path):
    d = np.load(path, allow_pickle=True)
    a = json.loads(str(d["args"]))
    p = BP1(**json.loads(a["params"]))
    return d, Model(p, Grid(dz=a["dz"], lam=a["lam"]), Settings(mode="qd"))


def catalogue(path, v_seis=V_SEIS):
    d, m = run_model(path)
    p = m.p
    z = m.z
    w = np.full(z.size, m.g.dz) * (2.0 if m.sym else 1.0)  # quadrature weights over the fault
    segs = sorted(tuple(s) for s in p.segments) if p.segments is not None else None
    out = []
    for i in range(int(d["n_events"])):
        pv = d[f"e{i}_peak_V"]
        if np.max(pv) < v_seis:
            continue
        s0, s1 = d[f"e{i}_slip0"], d[f"e{i}_slip1"]
        t0, t1 = float(d[f"e{i}_t_start"]), float(d[f"e{i}_t_end"])
        ds = s1 - s0
        dtau = -(m.static_stress(s1, t1) - m.static_stress(s0, t0))
        arr = d[f"e{i}_arrival"][THRESHOLDS.index(v_seis)]
        hit = np.isfinite(arr)
        ev = dict(index=i, t_yr=t0 / YEAR, duration_s=t1 - t0, z_hyp_km=float(z[np.nanargmin(arr)]) / 1e3,
                  z_min_km=float(z[hit].min()) / 1e3, z_max_km=float(z[hit].max()) / 1e3,
                  potency=float(np.sum(w * ds)), moment=float(p.mu * np.sum(w * ds)),
                  stress_drop=float(np.sum(w * dtau * ds) / np.sum(w * ds)),
                  vmax=float(np.max(pv)), slip_max=float(np.max(ds)),
                  slip_centre=float(ds[np.argmin(np.abs(z))]))
        if m.sym:
            ev["z_min_km"] = -ev["z_max_km"]
        if segs is not None:
            ev["segments"] = [int(np.mean(hit[(z >= a) & (z <= b)]) > 0.5) for a, b in segs]
        key = f"e{i}_energy_dyn"
        if key in d.files:
            ev["E_dyn"] = float(d[key][-1]) * (2.0 if m.sym else 1.0)
            ev["E_fric"] = float(d[f"e{i}_energy_fric"]) * (2.0 if m.sym else 1.0)
        out.append(ev)
    return out


def summary(cat, t_min=0.0):
    """Mean recurrence interval, potency and stress drop of the events after t_min (yr)."""
    ev = [e for e in cat if e["t_yr"] >= t_min]
    t = np.array([e["t_yr"] for e in ev])
    return dict(n=len(ev), mean_interval=float(np.mean(np.diff(t))) if t.size > 1 else None,
                mean_potency=float(np.mean([e["potency"] for e in ev])) if ev else None,
                mean_stress_drop=float(np.mean([e["stress_drop"] for e in ev])) if ev else None,
                mean_vmax=float(np.mean([e["vmax"] for e in ev])) if ev else None)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"== {f}")
        for e in catalogue(f):
            seg = e.get("segments", "")
            print(f"{e['t_yr']:9.2f} yr  hyp {e['z_hyp_km']:6.1f}  [{e['z_min_km']:6.1f},{e['z_max_km']:6.1f}] km  "
                  f"P {e['potency'] / 1e3:7.1f} m km  dtau {e['stress_drop'] / 1e6:5.2f} MPa  "
                  f"Vmax {e['vmax']:6.2f}  slip(0) {e['slip_centre']:5.2f}  {seg}")