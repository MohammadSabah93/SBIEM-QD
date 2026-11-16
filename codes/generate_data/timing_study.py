"""Cost of standard QD, SRD-QD and the FD history convolution on one core (v2 of the paper).

(1) Per-step wall time of the event phase for cell sizes 100-12.5 m, FD memory windows
    q = 20-320 and SRD-QD with J = 1-12 memory terms: n_steps event steps from the initial
    state (the FD kernel weights are computed before the clock starts and timed separately).
(2) Complete first-event runs at 50 m (192 yr of loading plus the event), split into the slow
    (interseismic) phase and the event phase by the counters of sbiem_bp1.Model.timing.

Writes data/analysis/timing_v2.json. Run on an otherwise idle machine, one process at a time:
    OMP_NUM_THREADS=1 python timing_study.py
"""

from __future__ import annotations

import json
import os
import platform
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS  # noqa: E402
from sbiem_bp1 import YEAR, BP1, Grid, Model, Settings, SpectralMemoryTreatment, Treatment  # noqa: E402

OUT = ANALYSIS / "timing_v2.json"


def model(mode, dz, q=4.0, kind=None, **kw):
    p = BP1(mode="II")
    tr = None if mode == "fd" else (Treatment() if kind is None else SpectralMemoryTreatment(kind=kind))
    return Model(p, Grid(dz=dz), Settings(mode=mode, q_window=q, verbose=False, **kw), tr)


def per_step(mode, dz, q=4.0, kind=None, n_steps=400):
    m = model(mode, dz, q, kind, v_off=-1.0, max_event_steps=n_steps)
    p = m.p
    dt = m.s.beta * dz / p.cs
    w = time.perf_counter()
    hist = 0
    if mode == "fd":
        m.setup_convolution(dt)
        hist = int(m.conv["sizes"].sum())
    setup = time.perf_counter() - w
    m._ev, m._prev, m.rec_t, m.rec_vmax = None, None, [], []
    v = np.full(m.n, p.vinit)
    w = time.perf_counter()
    m.event_phase(0.0, np.zeros(m.n), p.theta0(m.z), v)
    el = time.perf_counter() - w
    J = 0 if kind is None else int(m.treatment.c.size)
    return dict(mode=mode, dz=dz, N=m.g.N, q=q if mode == "fd" else None, kind=kind, J=J,
                s_per_step=el / m.timing["event_steps"], setup_s=setup, history_complex=hist,
                memory_complex=J * m.g.k.size)


def full_run(mode, q=4.0, kind=None):
    m = model(mode, 50.0, q, kind, t_final=400 * YEAR, max_events=1)
    w = time.perf_counter()
    m.run()
    return dict(mode=mode, q=q if mode == "fd" else None, kind=kind, total_s=time.perf_counter() - w,
                t_event_yr=m.events[0]["t_start"] / YEAR,
                event_duration_s=float(m.events[0]["t_end"] - m.events[0]["t_start"]), **m.timing)


def retime_srd():
    """Re-time the SRD-QD cases (decay factors cached since 2026-10-22) and merge into OUT."""
    res = json.loads(OUT.read_text())
    per_step("qd", 100.0, 4.0, "pex5", n_steps=50)
    res["per_step_uncached"] = [r for r in res["per_step"] if r["kind"]]
    keep = [r for r in res["per_step"] if not r["kind"]]
    for dz in (100.0, 50.0, 25.0, 12.5):
        keep.append(per_step("qd", dz, 4.0, "pex5"))
    for k in ("pade1", "pex2", "pex3", "pex4", "pex6"):
        keep.append(per_step("qd", 50.0, 4.0, k))
    res["per_step"] = keep
    res["full_uncached"] = [r for r in res["full"] if r["kind"]]
    res["full"] = [r for r in res["full"] if not r["kind"]] + [full_run("qd", 4.0, "pade1"), full_run("qd", 4.0, "pex5")]
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["per_step"][-10:], indent=0), json.dumps(res["full"][-2:]))


if __name__ == "__main__":
    import sys
    if "--srd" in sys.argv:
        retime_srd()
        sys.exit()
    res = dict(cpu=platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
               python=platform.python_version(), numpy=np.__version__, threads=os.environ.get("OMP_NUM_THREADS"),
               per_step=[], full=[])
    per_step("qd", 100.0, 4.0, "pex5", n_steps=50)  # warm-up: numba compilation, FFT plans
    per_step("fd", 100.0, 20.0, None, n_steps=50)
    cases = []
    for dz in (100.0, 50.0, 25.0, 12.5):
        cases += [("qd", dz, 4.0, None), ("qd", dz, 4.0, "pex5"), ("fd", dz, 160.0, None)]
    cases += [("fd", 50.0, q, None) for q in (20.0, 40.0, 80.0, 320.0)]
    cases += [("qd", 50.0, 4.0, k) for k in ("pade1", "pex2", "pex3", "pex4", "pex6")]
    for c in cases:
        r = per_step(*c)
        res["per_step"].append(r)
        print(json.dumps(r), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    for c in (("qd", 4.0, None), ("qd", 4.0, "pade1"), ("qd", 4.0, "pex5"), ("fd", 80.0, None), ("fd", 160.0, None)):
        r = full_run(*c)
        res["full"].append(r)
        print(json.dumps(r), flush=True)
        OUT.write_text(json.dumps(res, indent=1))