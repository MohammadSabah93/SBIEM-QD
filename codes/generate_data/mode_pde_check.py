"""Independent check of the in-plane fault response: one Fourier mode, time-domain elastodynamics.

The fault response used by the FD reference (kernel of Geubelle & Rice 1995 / Lapusta & Liu 2009,
closed form Q(T) of the paper) and by SRD-QD is compared with a direct finite-difference solution
of the plane-strain wave equations for one Fourier mode along the fault, which uses no kernel.

Mode k of an antisymmetric in-plane slip delta(z, t) = d(t) e^{ikz}: by symmetry the half-space
y > 0 carries u_z = U(y, t) e^{ikz}, u_y = i V(y, t) e^{ikz} with U(0, t) = d(t)/2 and
sigma_yy(0, t) = 0. With k = c_s = rho = 1 (mu = 1, eta0 = 1/2, c_d = 1/r):
    U_tt = U_yy - U / r^2 - (1/r^2 - 1) V_y,
    V_tt = V_yy / r^2 - V + (1/r^2 - 1) U_y,
    V_y(0) = -(1 - 2 r^2) U(0)          (sigma_yy = 0),
and the fault shear stress change is f = U_y(0) - V(0). Second-order central differences in y
and time (leapfrog), rigid far end beyond the reach of the waves. The dynamic part
g = (f + Z_st d) / eta0 = -dd/dt + int K V is compared with
  kernel:  g = -V + int_0^t Q(t') V'(t - t') dt'       (Q closed form, Appendix of the paper),
  FD code: the discrete convolution of sbiem_bp1 (setup_convolution / history_sum) for one mode,
  SRD-QD:  Q replaced by Q_J (five pairs, one term), and standard QD (g = -V).

Writes data/analysis/mode_pde_check.json and the arrays data/analysis/mode_pde_check.npz.
usage: python mode_pde_check.py [--nu 0.25] [--tend 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numba as nb
import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS  # noqa: E402
from sbiem_bp1 import BP1, Grid, Model, Settings, history_push, history_sum, kernel_Q2, memory_poles  # noqa: E402

W, T0 = 0.3, 1.5  # slip-rate rise: V(t) = (1 + erf((t - T0)/W)) / 2, V(0) ~ 1e-12


def rate(t):
    return 0.5 * (1 + erf((t - T0) / W))


def slip(t):
    x = t - T0
    F = lambda x: 0.5 * (x + x * erf(x / W) + W / np.sqrt(np.pi) * np.exp(-(x / W) ** 2))
    return F(x) - F(-T0)


def accel(t):
    return np.exp(-((t - T0) / W) ** 2) / (W * np.sqrt(np.pi))


@nb.njit(cache=True)
def _leapfrog(h, dt, n_steps, J, r, d_hist, out_f):
    c2 = 1.0 / (r * r)
    lm = c2 - 1.0  # (lambda + mu) / mu
    U0 = np.zeros(J + 1)
    U1 = np.zeros(J + 1)
    V0 = np.zeros(J + 1)
    V1 = np.zeros(J + 1)
    U2 = np.zeros(J + 1)
    V2 = np.zeros(J + 1)
    for n in range(n_steps):
        # interior and fault node (ghost for V from sigma_yy = 0)
        ub = U1[0]
        vy0 = -(1.0 - 2.0 * r * r) * ub
        for j in range(J):
            if j == 0:
                vm = V1[1] - 2.0 * h * vy0
                uy = (-3.0 * U1[0] + 4.0 * U1[1] - U1[2]) / (2.0 * h)
                V2[0] = 2 * V1[0] - V0[0] + dt * dt * (c2 * (V1[1] - 2 * V1[0] + vm) / (h * h) - V1[0] + lm * uy)
                continue
            uyy = (U1[j + 1] - 2 * U1[j] + U1[j - 1]) / (h * h)
            vyy = (V1[j + 1] - 2 * V1[j] + V1[j - 1]) / (h * h)
            uy = (U1[j + 1] - U1[j - 1]) / (2 * h)
            vy = (V1[j + 1] - V1[j - 1]) / (2 * h)
            U2[j] = 2 * U1[j] - U0[j] + dt * dt * (uyy - c2 * U1[j] - lm * vy)
            V2[j] = 2 * V1[j] - V0[j] + dt * dt * (c2 * vyy - V1[j] + lm * uy)
        U2[0] = 0.5 * d_hist[n + 1]
        U2[J] = 0.0
        V2[J] = 0.0
        out_f[n + 1] = (-3.0 * U2[0] + 4.0 * U2[1] - U2[2]) / (2.0 * h) - V2[0]
        U0, U1, U2 = U1, U2, U0
        V0, V1, V2 = V1, V2, V0
    return out_f


def pde(r, t_end, h):
    cd = 1.0 / r
    L = 0.5 * cd * t_end + 20.0
    J = int(np.ceil(L / h))
    dt = 0.4 * h * r
    n = int(np.ceil(t_end / dt))
    t = dt * np.arange(n + 1)
    f = np.zeros(n + 1)
    _leapfrog(h, dt, n, J, r, slip(t), f)
    return t, f


def conv_Q(t, Qfun, n=2001):
    """int_0^t K(t') V(t - t') dt' = int_0^t Q(t - u) V'(u) du (V(0) ~ 1e-12 neglected), on the
    support of the narrow Gaussian V'."""
    out = np.zeros(t.size)
    for i, ti in enumerate(t):
        lo, hi = max(0.0, T0 - 8 * W), min(ti, T0 + 8 * W)
        if hi > lo:
            u = np.linspace(lo, hi, n)
            out[i] = np.trapezoid(Qfun(ti - u) * accel(u), u)
    return out


def fd_code(r, t_end, nu, beta=0.25, q=160.0, lam_k=2000.0, dz=50.0):
    """Dynamic stress of one mode from the FD convolution of sbiem_bp1 (prescribed slip rate).

    Mode with wavelength lam_k on the 50 m grid; the code's mid-step rule is applied with the exact
    mid-step slip rate. Returns dimensionless time T = |k| c_s t and g = f_dyn / eta0."""
    p = BP1(mode="II", nu=nu)
    m = Model(p, Grid(dz=dz), Settings(mode="fd", q_window=q, beta=beta))
    dt = beta * dz / p.cs
    m.setup_convolution(dt)
    c = m.conv
    mi = int(np.argmin(np.abs(m.g.k - 2 * np.pi / lam_k)))
    k = m.g.k[mi]
    one = lambda a: np.array([a[mi]])
    sizes, offs = one(c["sizes"]), np.array([0], np.int64)
    weights = c["weights"][c["offsets"][mi]: c["offsets"][mi] + c["sizes"][mi]].copy()
    tails, w0 = one(c["tails"]), c["w0"][mi]
    hist = np.zeros(int(sizes[0]), np.complex128)
    pos = np.zeros(1, np.int64)
    S = np.zeros(1, np.complex128)
    sc = k * p.cs  # T = sc * t
    n = int(np.ceil(t_end / (sc * dt)))
    T, g = [0.0], [0.0]
    for i in range(n):
        history_sum(hist, offs, sizes, weights, tails, pos, S)
        vmid = rate(sc * dt * (i + 0.5))
        mem = w0 * vmid + S[0].real
        T.append(sc * dt * (i + 1))
        g.append(-rate(T[-1]) + mem)  # f_dyn / eta0 = -V + M[V]
        history_push(hist, offs, sizes, pos, np.array([vmid + 0j]))
    return np.array(T), np.array(g), dict(k=k, Delta=sc * dt, lam_k=2 * np.pi / k)


def srd_code(c, tau, t_end, Delta):
    """Exact exponential update of the paper with the exact mid-step slip rate."""
    n = int(np.ceil(t_end / Delta))
    a = np.exp(-Delta / tau)
    y = np.zeros(c.size, complex)
    T, g = [0.0], [0.0]
    for i in range(n):
        vmid = rate(Delta * (i + 0.5))
        y = y * a + c * vmid * (1 - a)
        T.append(Delta * (i + 1))
        g.append(-rate(T[-1]) + y.sum().real)
    return np.array(T), np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.25)
    ap.add_argument("--tend", type=float, default=200.0)
    args = ap.parse_args()
    p = BP1(mode="II", nu=args.nu)
    r = p.cs / p.cd
    m0 = 2 * (1 - r * r)
    res = dict(nu=args.nu, r=r, t_end=args.tend, rise_W=W, rise_T0=T0)
    arrays = {}
    # PDE on two grids (Richardson estimate of its own error)
    sols = {}
    hs = (0.01, 0.005, 0.0025)
    for h in hs:
        t, f = pde(r, args.tend, h)
        sols[h] = (t, (f + 0.5 * m0 * slip(t)) / 0.5)  # g = (f + Z_st d) / eta0, Z_st = m0 / 2
    tc = np.linspace(0, args.tend, 4001)
    g_pde = {h: np.interp(tc, *sols[h]) for h in sols}
    e1 = np.max(np.abs(g_pde[hs[0]] - g_pde[hs[1]]))
    e2 = np.max(np.abs(g_pde[hs[1]] - g_pde[hs[2]]))
    res["pde_observed_order"] = float(np.log2(e1 / e2))
    rich = lambda a, b: g_pde[b] + (g_pde[b] - g_pde[a]) / 3  # second-order Richardson
    g_fine = rich(hs[1], hs[2])
    res["pde_raw_error_finest_max"] = float(np.max(np.abs(g_pde[hs[2]] - g_fine)))
    res["pde_richardson_spread_max"] = float(np.max(np.abs(rich(hs[0], hs[1]) - g_fine)))
    g_ker = -rate(tc) + conv_Q(tc, lambda x: kernel_Q2(x, r))
    res["kernel_vs_pde_max"] = float(np.max(np.abs(g_ker - g_fine)))
    res["kernel_vs_pde_rms"] = float(np.sqrt(np.mean((g_ker - g_fine) ** 2)))
    arrays.update(T=tc, g_pde=g_fine, g_kernel=g_ker, g_qd=-rate(tc))
    for key, kind in (("srd5", "pex5"), ("srd1", "pade1")):
        cc, tt = memory_poles(p, kind)
        QJ = lambda x: np.real(np.sum(cc[None, :] * (1 - np.exp(-np.asarray(x)[:, None] / tt[None, :])), axis=1))
        g = -rate(tc) + conv_Q(tc, QJ)
        arrays[f"g_{key}"] = g
        res[f"{key}_vs_pde_max"] = float(np.max(np.abs(g - g_fine)))
        res[f"{key}_vs_pde_rms"] = float(np.sqrt(np.mean((g - g_fine) ** 2)))
    res["qd_vs_pde_max"] = float(np.max(np.abs(-rate(tc) - g_fine)))
    # the discrete operators of the simulations (FD convolution, SRD exponential update)
    Tf, gf, info = fd_code(r, args.tend, args.nu)
    res["fd_code"] = info
    gi = np.interp(Tf, tc, g_fine)
    res["fd_code_vs_pde_max"] = float(np.max(np.abs(gf - gi)))
    cc, tt = memory_poles(p, "pex5")
    Ts, gs = srd_code(cc, tt, args.tend, info["Delta"])
    res["srd_code_vs_pde_max"] = float(np.max(np.abs(gs - np.interp(Ts, tc, g_fine))))
    arrays.update(T_fd_code=Tf, g_fd_code=gf, T_srd_code=Ts, g_srd_code=gs)
    tag = f"_nu{args.nu:g}" if args.nu != 0.25 else ""
    Path(ANALYSIS, f"mode_pde_check{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(Path(ANALYSIS, f"mode_pde_check{tag}.npz"), **arrays)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()