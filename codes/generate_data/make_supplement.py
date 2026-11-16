"""Machine-readable supplement of the paper (memory constants, certificates, environment).

Writes data/supplement/:
  coefficients/<set>.json   constants c_j, tau_j of every set used in the paper, the exact constraints
                            and their residuals, the fitting inputs and the continuous certificate,
  coefficients/all_sets.csv the same constants as one table (one row per relaxation term),
  certificates_v2.json      continuous and per-mode discrete certificates (certify_discrete.py),
  mode_pde_check*.json      independent single-mode check (mode_pde_check.py),
  timing_v2.json            single-core timings (timing_study.py),
  environment.txt           software and hardware.
Run after certify_discrete.py, mode_pde_check.py and timing_study.py.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
from importlib.metadata import version
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS, POLES, SUPPLEMENT  # noqa: E402
from certify_discrete import SETS  # noqa: E402

DATA = ANALYSIS
OUT = SUPPLEMENT
FIT_INPUTS = dict(
    misfit="Phi = sum_i [Q_J(T_i) - Q(T_i)]^2 + w^2 sum_l |(Z_J - Z)(-i x_l) / (mu |k| / 2)|^2",
    T_points="401 uniform points on 0 <= T <= 40 and 300 log-spaced points on 40 <= T <= T_max",
    front_speeds="60 uniform x_l = v_l / c_s between 0.02 c_R and 0.97 c_R",
    equalities="sum c = 1, sum c tau = (1 + r^4)/2, sum c / tau = 2 (1 - r^2), eliminated exactly",
    passivity="D_J(x) >= eps x^2 / (1 + x^2), eps = 1e-5, plus S2 = sum c tau^2 >= eps; cutting planes on "
              "2e5 log-spaced x in [1e-6, 1e4] and 2e4 points in [0.9, 1.1]",
    poles="Gauss-Newton (variable projection), 8 starts (7 random, 1 earlier fit), Re(1/tau) > 1e-4",
    script="fit_memory_passive.py --K <pairs> --nu <nu> --tmax <T_max> --wfront <w>",
)

README = """# Supplement: passive spectral memory for in-plane quasi-dynamic fault models

Machine-readable data of the paper (version 2, 22 October 2026).

* `coefficients/<set>.json`: memory constants of every set used in the paper, with the convention
  M_J(p) = sum_j c_j / (1 + tau_j p), p = s / (|k| c_s); the exact constraints and their residuals; the
  pole decay rates and frequencies; the fit report and fitting inputs; the continuous certificate and the
  discrete certificates (number of certified Fourier modes per simulation grid and step).
  Set names: `II_K<pairs>_pex_T<T_max>_w<front weight>_nu<Poisson ratio>`. The constants of the paper are
  the five-pair sets with T_max = 600 and w = 0.5.
* `coefficients/all_sets.csv`: the same constants as one table, one row per relaxation term (the terms
  come in complex-conjugate pairs; five pairs = ten complex terms).
* `certificates_v2.json`: continuous certificates of every set and the per-mode discrete certificates
  (40 000 in total, all passive; Sturm and continued-fraction root isolation agree on every 50th step).
* `mode_pde_check*.json`: independent check of the fault response of one Fourier mode against a direct
  solution of the plane-strain wave equations (nu = 0.25 and 0.35).
* `timing_v2.json`: single-core wall-clock times per step and of the first event.
* `environment.txt`: software and hardware.

Scripts (paper_QD_imitating_FD/codes/): generate_data/fit_memory_passive.py, certify_discrete.py,
mode_pde_check.py, sbiem_bp1.py, cycle_analysis.py, timing_study.py, make_supplement.py;
plot_figures/make_figures_rev.py, make_figures_v2.py; build_paper/make_tables_v2.py; run scripts in
generate_data/launch/rev and generate_data/launch/v2.
"""


def main():
    (OUT / "coefficients").mkdir(parents=True, exist_ok=True)
    cert = json.loads((DATA / "certificates_v2.json").read_text())
    cont = {r["set"]: r for r in cert["continuous"]}
    rows = []
    for name, nu in SETS:
        d = json.loads((POLES / name).read_text())
        c = np.array(d["c_re"]) + 1j * np.array(d["c_im"])
        tau = np.array(d["tau_re"]) + 1j * np.array(d["tau_im"])
        r = np.sqrt((1 - 2 * nu) / (2 * (1 - nu)))
        tag = Path(name).stem.replace("memory_poles_", "")
        out = dict(set=tag, mode="II", nu=nu, r=r, pairs=d["K"], terms=2 * d["K"], T_max=d["t_max"],
                   w_front=d["w_front"], eps=d["eps"],
                   convention="M_J(p) = sum_j c_j / (1 + tau_j p), p = s / (|k| c_s); terms in conjugate pairs",
                   c_re=d["c_re"], c_im=d["c_im"], tau_re=d["tau_re"], tau_im=d["tau_im"],
                   constraints=dict(sum_c=float(np.real(c.sum())), sum_c_target=1.0,
                                    sum_c_tau=float(np.real((c * tau).sum())), sum_c_tau_target=(1 + r ** 4) / 2,
                                    sum_c_over_tau=float(np.real((c / tau).sum())), sum_c_over_tau_target=2 * (1 - r * r),
                                    S2=float(np.real((c * tau ** 2).sum()))),
                   poles=dict(min_decay_rate=float(np.min(np.real(1 / tau))),
                              frequencies=sorted(set(np.round(np.abs(np.imag(1 / tau)), 6).tolist()))),
                   fit_report=d["report"], fit_inputs=FIT_INPUTS,
                   certificate_continuous=cont[name],
                   certificate_discrete=[x for x in cert["discrete"] if x["set"] == name])
        (OUT / "coefficients" / f"{tag}.json").write_text(json.dumps(out, indent=1))
        for j in range(c.size):
            rows.append([tag, nu, d["K"], d["t_max"], d["w_front"], j + 1, repr(float(c[j].real)),
                         repr(float(c[j].imag)), repr(float(tau[j].real)), repr(float(tau[j].imag))])
    with open(OUT / "coefficients" / "all_sets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "nu", "pairs", "T_max", "w_front", "j", "Re_c", "Im_c", "Re_tau", "Im_tau"])
        w.writerows(rows)
    for f in ("certificates_v2.json", "mode_pde_check.json", "mode_pde_check_nu0.35.json", "timing_v2.json"):
        if (DATA / f).exists():
            shutil.copy(DATA / f, OUT / f)
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    env = [f"python {platform.python_version()}", f"platform {platform.platform()}", f"cpu {cpu}"]
    env += [f"{p} {version(p)}" for p in ("numpy", "scipy", "numba", "sympy", "matplotlib")]
    (OUT / "environment.txt").write_text("\n".join(env) + "\n")
    (OUT / "README.md").write_text(README)
    print("\n".join(env))
    print(len(SETS), "sets,", len(rows), "terms")


if __name__ == "__main__":
    main()