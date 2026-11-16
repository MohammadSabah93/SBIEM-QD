"""Shared event metrics for SBIEM (.npz from sbiem_bp1.save_events) and PorePy
(events/event_XX.npz from bp1_qd_fd.EventRecorder) results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CS = 3464.0


def load_sbiem(path, index=0):
    d = np.load(path, allow_pickle=True)
    e = {k[len(f"e{index}_"):]: d[k] for k in d.files if k.startswith(f"e{index}_")}
    e["z"] = d["z"]
    e["thresholds"] = np.array((1e-3, 1e-2, 1e-1, 1.0))
    e["slip"] = e["slip1"] - e["slip0"]
    return e


def load_porepy(run_dir, index=0):
    d = np.load(Path(run_dir) / "events" / f"event_{index:02d}.npz")
    e = {k: d[k] for k in d.files}
    order = np.argsort(e["depth"])
    for key in ("arrival",):
        e[key] = e[key][:, order]
    for key in ("peak_V", "peak_t", "slip0", "slip1", "shear0", "shear1", "shear_max"):
        e[key] = e[key][order]
    for key in ("snap_V", "snap_slip", "snap_shear"):
        if e[key].size:
            e[key] = e[key][:, order]
    e["z"] = e["depth"][order]
    e["slip"] = e["slip1"] - e["slip0"]
    return e


def rupture_speed(z, T, window=1000.0):
    """|dz/dT| by a local linear fit of z(T) over +-window/2 around every cell."""
    v = np.full(z.size, np.nan)
    ok = np.isfinite(T)
    for i in np.flatnonzero(ok):
        sel = ok & (np.abs(z - z[i]) <= window / 2)
        if sel.sum() < 3:
            continue
        A = np.vstack([T[sel] - T[i], np.ones(sel.sum())]).T
        # fit T(z): dT/dz slope, robust to steep fronts
        slope = np.polyfit(z[sel], T[sel], 1)[0]
        v[i] = 1.0 / abs(slope) if slope != 0 else np.inf
    return v


def metrics(e, threshold_index=2, window=1000.0, z_ranges=((2e3, 8e3), (8e3, 12e3))):
    z = e["z"]
    T = e["arrival"][threshold_index]
    hyp = z[np.nanargmin(T)]
    v_r = rupture_speed(z, T, window)
    out = dict(hypocenter_km=hyp / 1e3, t0=float(np.nanmin(T)))
    for lo, hi in z_ranges:
        sel = (z >= lo) & (z <= hi) & np.isfinite(T)
        if sel.sum() > 2:
            slope = np.polyfit(z[sel], T[sel], 1)[0]
            out[f"vr_{lo / 1e3:g}_{hi / 1e3:g}km"] = 1 / abs(slope)
            out[f"Vpk_{lo / 1e3:g}_{hi / 1e3:g}km"] = float(np.mean(e["peak_V"][sel]))
    out["Vpk_max"] = float(np.max(e["peak_V"]))
    out["Vpk_surface"] = float(e["peak_V"][np.argmin(z)])
    out["slip_surface"] = float(e["slip"][np.argmin(z)])
    out["slip_max"] = float(np.max(e["slip"]))
    out["t_surface"] = float(T[np.argmin(z)] - np.nanmin(T))
    return out, v_r


def major_events(path, min_duration=1.0, min_slip=0.05):
    """Indices of SBIEM events that are real ruptures (drops tail flickers)."""
    d = np.load(path, allow_pickle=True)
    keep = []
    for i in range(int(d["n_events"])):
        dur = float(d[f"e{i}_t_end"]) - float(d[f"e{i}_t_start"])
        slip = np.max(d[f"e{i}_slip1"] - d[f"e{i}_slip0"])
        if dur >= min_duration and slip >= min_slip:
            keep.append(i)
    return keep