"""Universal relaxation spectra of the elastodynamic fault memory (modes III and II).

The FD memory of Fourier mode k is M(p), p = s / (|k| c_s) (paper, Prop. 1). It is
approximated by M(p) ~ sum_j c_j / (1 + tau_j p) with K complex-conjugate pairs,
fitted once to the exact cumulative kernel Q(T) on 0 <= T <= 40 and to the exact
steady-front impedance at subsonic phase speeds, with three exact constraints:
sum c = 1 (static cancellation), sum c tau = tau_1 (first moment), sum c / tau = m0
(initial kernel value). The result depends only on the mode and on nu; no FD
simulation is used. Writes data/memory_poles/memory_poles.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import POLES  # noqa: E402
from sbiem_bp1 import BP1, kernel_Q, kernel_Q2  # noqa: E402


def m_exact(p, mode, r):
    if mode == "III":
        return 1 - (np.sqrt(1 + p * p) - 1) / p
    bs, bd = np.sqrt(1 + p * p), np.sqrt(1 + r * r * p * p)
    N = 1 / bs + 2 * bs + bs ** 3 - 4 * bd
    return 1 - (N / p ** 2 - 2 * (1 - r * r)) / p


def unpack(x, K):
    taus, cs = [], []
    for i in range(K):
        t = np.exp(x[4 * i]) * np.exp(1j * x[4 * i + 1])
        c = x[4 * i + 2] + 1j * x[4 * i + 3]
        taus += [t, np.conj(t)]
        cs += [c, np.conj(c)]
    return np.array(cs, complex), np.array(taus, complex)


def fit(mode, nu=0.25, K=3, trials=40, seed=0, t_max=40.0, n_t=801, passive=0.0):
    p0 = BP1(mode=mode, nu=nu)
    r = p0.cs / p0.cd
    m0 = 1.0 if mode == "III" else 2 * (1 - r * r)
    zst = 1.0 if mode == "III" else 2 * (1 - r * r)
    tau1 = 0.5 if mode == "III" else (1 + r ** 4) / 2
    T = np.unique(np.r_[np.linspace(0, 40, 401), np.geomspace(40, t_max, n_t)]) if t_max > 40 else np.linspace(0, 40, 801)
    Q = kernel_Q(T) if mode == "III" else kernel_Q2(T, r)
    x_sub = np.linspace(0.02, 0.97, 60) * (p0.c_lim / p0.cs)
    st_ex = zst + (-1j * x_sub) * (1 - m_exact(-1j * x_sub, mode, r))

    w_pass = np.r_[np.linspace(0.01, 3.0, 1500), np.geomspace(3.0, 300.0, 200)]

    def resid(x):
        c, tau = unpack(x, K)
        Qm = np.real(np.sum(c[None, :] * (1 - np.exp(-T[:, None] / tau[None, :])), axis=1))
        Mm = np.sum(c[None, :] / (1 + tau[None, :] * (-1j * x_sub)[:, None]), axis=1)
        st = zst + (-1j * x_sub) * (1 - Mm)
        stab = np.minimum(np.real(1 / tau), 0) * 100
        if passive > 0:
            # passivity: resistance Im Z(i w) / w >= 0 for all w (no energy creation)
            pw = 1j * w_pass
            Zp = zst + pw * (1 - np.sum(c[None, :] / (1 + tau[None, :] * pw[:, None]), axis=1))
            stab = np.concatenate([stab, passive * np.minimum(Zp.imag / w_pass, 0.0)])
        cons = [30 * (np.real(c.sum()) - 1), 30 * (np.real((c * tau).sum()) - tau1),
                30 * (np.real((c / tau).sum()) - m0)]
        return np.concatenate([Qm - Q, 0.5 * (st.real - st_ex.real), 0.5 * (st.imag - st_ex.imag),
                               cons, stab])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        x0 = []
        for _ in range(K):
            x0 += [np.log(rng.uniform(0.3, 3)), rng.uniform(-1.4, 1.4), rng.uniform(-0.5, 0.5),
                   rng.uniform(-0.5, 0.5)]
        sol = least_squares(resid, np.array(x0), max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    c, tau = unpack(best.x, K)
    rr = resid(best.x)
    n = T.size
    report = dict(Q_rms=float(np.sqrt(np.mean(rr[:n] ** 2))),
                  steady_re_err=float(2 * np.max(np.abs(rr[n:n + 60]))),
                  steady_im_err=float(2 * np.max(np.abs(rr[n + 60:n + 120]))),
                  min_decay=float(np.min(np.real(1 / tau))))
    return dict(c_re=c.real.tolist(), c_im=c.imag.tolist(), tau_re=tau.real.tolist(),
                tau_im=tau.imag.tolist(), tau1=tau1, m0=m0, nu=nu, report=report)


if __name__ == "__main__":
    import sys

    # usage: python fit_memory_kernel.py [K ...] [--passive]
    #   -> data/memory_poles/memory_poles_II_K{K}_long[_passive].json
    passive = "--passive" in sys.argv
    for K in (int(k) for k in sys.argv[1:] if not k.startswith("--")):
        res = fit("II", K=K, t_max=600.0, n_t=300, trials=30, passive=50.0 if passive else 0.0)
        tag = "_passive" if passive else ""
        (POLES / f"memory_poles_II_K{K}_long{tag}.json").write_text(json.dumps(res, indent=1))
        print(K, res["report"], flush=True)