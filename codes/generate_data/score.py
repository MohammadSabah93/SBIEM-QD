"""Profile-wise errors of QD variants against the FD reference.

For one event, on the seismogenic section z in [z_lo, z_hi] (excluding the
nucleation zone around the FD hypocentre), report
  e_v  = RMS relative error of the local rupture speed v_r(z),
  e_V  = RMS relative error of the peak slip rate V_pk(z),
  e_d  = RMS relative error of the coseismic slip,
  dT   = relative error of the up-dip rupture time to the free surface,
plus the surface peak slip rate ratio.

usage: python score.py FD.npz QD1.npz [QD2.npz ...] [--event-fd i] [--event-qd j]
"""

import sys
from pathlib import Path

import numpy as np

from analysis import load_porepy, load_sbiem, metrics

CS = 3464.0


def load(path, index):
    p = Path(path)
    return load_porepy(p, index) if p.is_dir() else load_sbiem(p, index)


def profile(e, th=2):
    m, vr = metrics(e, threshold_index=th)
    return m, vr


def score(fd, qd, z_lo=1e3, z_hi=14e3, r_excl=2e3, th=2):
    mf, vf = profile(fd, th)
    mq, vq = profile(qd, th)
    zf, zq = fd["z"], qd["z"]
    z = zf
    vq_i = np.interp(z, zq, vq)
    Vq_i = np.interp(z, zq, qd["peak_V"])
    dq_i = np.interp(z, zq, qd["slip"])
    hyp = mf["hypocenter_km"] * 1e3
    sel = (z >= z_lo) & (z <= z_hi) & (np.abs(z - hyp) > r_excl) & np.isfinite(vf) & np.isfinite(vq_i)
    sel &= vf < 2 * CS  # exclude apparent speeds of simultaneous arrivals
    rel = lambda a, b: np.sqrt(np.mean(((a - b) / b) ** 2))
    return dict(
        e_v=rel(vq_i[sel], vf[sel]),
        e_V=rel(Vq_i[sel], fd["peak_V"][sel]),
        e_d=rel(dq_i[sel], fd["slip"][sel]),
        dT=(mq["t_surface"] - mf["t_surface"]) / mf["t_surface"],
        Vsurf=mq["Vpk_surface"] / mf["Vpk_surface"],
        n=int(sel.sum()),
    )


def main(argv):
    ef = eq = 0
    files = []
    it = iter(argv)
    for a in it:
        if a == "--event-fd":
            ef = int(next(it))
        elif a == "--event-qd":
            eq = int(next(it))
        else:
            files.append(a)
    fd = load(files[0], ef)
    print(f"reference {Path(files[0]).stem}")
    print(f"{'model':30s} {'e_v':>6s} {'e_V':>6s} {'e_slip':>6s} {'dT_surf':>8s} {'Vsurf/FD':>8s}")
    for f in files[1:]:
        try:
            s = score(fd, load(f, eq))
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(f).stem[:30]:30s} failed: {exc}")
            continue
        print(f"{Path(f).stem[:30]:30s} {s['e_v']:6.2f} {s['e_V']:6.2f} {s['e_d']:6.2f} "
              f"{s['dT']:+8.2f} {s['Vsurf']:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])