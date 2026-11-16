"""Passive relaxation spectra of the in-plane (mode II) fault memory, exact constraints.

Revision of ``fit_memory_kernel.py``. The memory M(p), p = s / (|k| c_s), is approximated by

    M_J(p) = sum_j c_j / (1 + tau_j p)        (K complex-conjugate pairs, J = 2K terms).

For fixed relaxation times tau_j, the misfit to the exact kernel, the three exact constraints

    sum c_j = 1,   sum c_j tau_j = tau_1,   sum c_j / tau_j = m_0,

and the passivity condition D(x) = 1 - Re M_J(ix) >= 0 are all *linear* in the weights c_j,
so the weights solve a small convex quadratic programme (QP). The relaxation times are
optimized in an outer loop (variable projection). The equalities are then enforced to
machine precision by an exact projection, and passivity is certified for *all* real x by
a Sturm-sequence count on the numerator polynomial of D, in exact rational arithmetic on
the stored double-precision constants (``certify``). No factor on the radiation damping
is needed: D(0) = 0 exactly (exact quasi-static limit) and D(x) >= 0 for all x.

``discrete_dissipation`` evaluates the same function for the time-stepping scheme of the
paper (exact exponential update with mid-step slip rate), for every dimensionless step
Delta = |k| c_s dt in (0, pi beta].

usage:  python fit_memory_passive.py [--K 5] [--nu 0.25] [--tmax 600] [--wfront 0.5]
        -> data/memory_poles/memory_poles_II_K{K}_pex_T{tmax}_w{wfront}_nu{nu}.json
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import POLES, ROOT  # noqa: E402
from sbiem_bp1 import BP1, kernel_Q2  # noqa: E402



def m_exact(p, r):
    bs, bd = np.sqrt(1 + p * p), np.sqrt(1 + r * r * p * p)
    N = 1 / bs + 2 * bs + bs ** 3 - 4 * bd
    return 1 - (N / p ** 2 - 2 * (1 - r * r)) / p


def constants(nu):
    p0 = BP1(mode="II", nu=nu)
    r = p0.cs / p0.cd
    return dict(r=r, m0=2 * (1 - r * r), zst=2 * (1 - r * r), tau1=(1 + r ** 4) / 2,
                xR=p0.c_lim / p0.cs)


def D_of(c, tau, x):
    """Normalized dissipation D(x) = Im Z(k, i w) / (eta0 w) = 1 - Re M_J(ix), x = w / (|k| c_s)."""
    out = np.empty(x.size)
    for i in range(0, x.size, 50000):
        xx = x[i:i + 50000]
        out[i:i + 50000] = 1 - np.real(np.sum(c[None, :] / (1 + 1j * tau[None, :] * xx[:, None]), axis=1))
    return out


def poles_from(theta):
    """theta = [log|tau_1|, arg tau_1, ...] -> one tau per pair."""
    th = np.asarray(theta).reshape(-1, 2)
    return np.exp(th[:, 0]) * np.exp(1j * th[:, 1])


def pair_columns(f):
    """Columns for the real unknowns (Re c, Im c) of a pair term c f(tau) + conj(c) f(conj tau).

    f: callable tau -> complex array of the response of c = 1 for one pole tau."""

    def cols(tau):
        a, b = f(tau), f(np.conj(tau))
        return np.stack([a + b, 1j * (a - b)], axis=-1)  # c = cr + i ci

    return cols


class Problem:
    def __init__(self, nu=0.25, t_max=600.0, n_t=300, w_front=0.5, eps=1e-5):
        k = constants(nu)
        self.__dict__.update(k)
        self.nu, self.t_max, self.w_front, self.eps = nu, t_max, w_front, eps
        r = self.r
        T = (np.unique(np.r_[np.linspace(0, 40, 401), np.geomspace(40, t_max, n_t)])
             if t_max > 40 else np.linspace(0, t_max, 801))
        self.T, self.Q = T, kernel_Q2(T, r)
        self.x_sub = np.linspace(0.02, 0.97, 60) * self.xR
        ps = -1j * self.x_sub
        self.st_ex = self.zst + ps * (1 - m_exact(ps, r))
        # passivity grid: dense through the subsonic band and around the S-wave point x = 1
        self.x_pass = np.unique(np.r_[np.geomspace(1e-3, 0.9, 600), np.linspace(0.9, 1.1, 4001),
                                      np.geomspace(1.1, 1e3, 800)])

    # linear maps from the real unknowns u = [cr_1, ci_1, cr_2, ci_2, ...] -------------
    def blocks(self, taus):
        T, xs, xp = self.T, self.x_sub, self.x_pass
        Aq, As, Ap, E = [], [], [], []
        for tau in taus:
            Aq.append(pair_columns(lambda t: 1 - np.exp(-T / t))(tau).real)
            # steady front at p = -ix: st = zst + p (1 - M) -> linear part -p M
            As.append(pair_columns(lambda t: -(-1j * xs) / (1 - 1j * t * xs))(tau))
            Ap.append(pair_columns(lambda t: 1 / (1 + 1j * t * xp))(tau).real)  # Re M(ix)
            E.append(np.stack([pair_columns(lambda t: np.array([1.0 + 0j]))(tau)[0],
                               pair_columns(lambda t: np.array([t]))(tau)[0],
                               pair_columns(lambda t: np.array([1 / t]))(tau)[0],
                               pair_columns(lambda t: np.array([t * t]))(tau)[0]]).real)
        Aq, As, Ap = (np.concatenate(B, axis=1) for B in (Aq, As, Ap))
        E = np.concatenate(E, axis=1)  # rows: sum c, sum c tau, sum c / tau, sum c tau^2
        ps = -1j * xs
        st0 = self.zst + ps  # st = st0 + As u
        A = np.vstack([Aq, self.w_front * As.real, self.w_front * As.imag])
        b = np.r_[self.Q, self.w_front * (self.st_ex.real - st0.real),
                  self.w_front * (self.st_ex.imag - st0.imag)]
        eq_A, eq_b = E[:3], np.array([1.0, self.tau1, self.m0])
        # D(x) = 1 - Ap u >= eps x^2 / (1 + x^2);  S2 = E[3] u >= eps (D ~ S2 x^2 near 0)
        G = np.vstack([Ap, -E[3:4]])
        h = np.r_[1 - self.eps * xp ** 2 / (1 + xp ** 2), -self.eps]
        return A, b, eq_A, eq_b, G, h

    def solve_weights(self, taus):
        """QP for the weights with the equalities eliminated exactly: u = u_p + N w."""
        A, b, eq_A, eq_b, G, h = self.blocks(taus)
        u_p = np.linalg.lstsq(eq_A, eq_b, rcond=None)[0]
        _, sv, Vt = np.linalg.svd(eq_A)
        N = Vt[eq_A.shape[0]:].T  # null space of the equality rows
        An, bn = A @ N, b - A @ u_p
        Gn, hn = G @ N, h - G @ u_p
        w0 = np.linalg.lstsq(An, bn, rcond=None)[0]
        if np.any(Gn @ w0 > hn):  # unconstrained optimum not passive: solve the QP
            H, g = An.T @ An, An.T @ bn
            res = minimize(lambda w: 0.5 * w @ H @ w - g @ w, np.zeros(N.shape[1]),
                           jac=lambda w: H @ w - g, method="SLSQP",
                           constraints=[dict(type="ineq", fun=lambda w: hn - Gn @ w, jac=lambda w: -Gn)],
                           options=dict(maxiter=1000, ftol=1e-15))
            w = res.x
            status = int(res.status)
        else:
            w, status = w0, -1  # -1: passivity constraints inactive
        u = u_p + N @ w
        viol = float(np.max(G @ u - h))
        return u, A @ u - b, dict(status=status, viol=viol)

    def solve_certified(self, taus, max_iter=30):
        """Cutting-plane (exchange) loop for the semi-infinite constraint D(x) >= margin(x):
        after each QP, the local minima of D on a very fine grid are located, refined and
        added to the constraint grid, until D >= margin / 2 everywhere on the fine grid."""
        base = self.x_pass.copy()
        extra = np.empty(0)
        xf = np.unique(np.r_[np.geomspace(1e-6, 1e4, 200001), np.linspace(0.9, 1.1, 20001)])
        th = np.ravel([[np.log(abs(t)), np.angle(t)] for t in taus])
        for it in range(max_iter):
            self.x_pass = np.unique(np.r_[base, extra])
            u, r, info = self.solve_weights(taus)
            c, tau = self.unpack(th, u)
            D = D_of(c, tau, xf)
            marg = self.eps * xf ** 2 / (1 + xf ** 2)
            bad = D < 0.5 * marg
            if not np.any(bad):
                break
            # local minima of D - margin among the bad points
            g = D - marg
            loc = np.flatnonzero(bad[1:-1] & (g[1:-1] <= g[:-2]) & (g[1:-1] <= g[2:])) + 1
            new = []
            for i in loc:
                res = minimize_scalar(lambda x: D_of(c, tau, np.array([x]))[0]
                                      - self.eps * x * x / (1 + x * x),
                                      bounds=(xf[i - 1], xf[i + 1]), method="bounded",
                                      options=dict(xatol=1e-12))
                new.append(res.x)
            extra = np.r_[extra, new, xf[loc]]
        self.x_pass = base
        info.update(cut_iterations=it + 1, n_cuts=int(extra.size),
                    D_min_fine=float(np.min(D)), margin_ok=bool(not np.any(bad)))
        return u, r, info

    def residual(self, theta):
        taus = poles_from(theta)
        if np.any(np.real(1 / taus) <= 1e-4):  # every memory variable must decay
            return 1e3 * np.ones(self.T.size + 2 * self.x_sub.size)
        u, r, info = self.solve_weights(taus)
        return r + 1e3 * max(0.0, info["viol"])

    def unpack(self, theta, u):
        taus = poles_from(theta)
        cp = u[0::2] + 1j * u[1::2]
        c = np.ravel(np.column_stack([cp, np.conj(cp)]))
        tau = np.ravel(np.column_stack([taus, np.conj(taus)]))
        return c, tau


# ---------------------------------------------------------------------------------------
# Certification of passivity for all real x (exact rational arithmetic)
# ---------------------------------------------------------------------------------------
def certify(c, tau):
    """Certify D(x) = 1 - Re M_J(ix) >= 0 for all real x, M_J = sum c_j / (1 + tau_j p).

    With p = ix, Re M(ix) = [M(p) + M(-p)] / 2 (real coefficients), so
    D = [Q(p)Q(-p) - (P(p)Q(-p) + P(-p)Q(p)) / 2] / [Q(p)Q(-p)], M = P/Q. The numerator is an
    even polynomial in p, i.e. a polynomial R(u) in u = p^2 = -x^2 <= 0, and the denominator
    |Q(ix)|^2 > 0. D >= 0 for all x  <=>  R(u) >= 0 on u <= 0. We count the real roots of R in
    (-inf, 0) with odd multiplicity by Sturm's theorem (sympy, exact rationals of the doubles).
    Returns a dict with the root count, R(0), the leading behaviour and the minimum of D on a
    fine grid."""
    import sympy as sp

    p = sp.symbols("p", real=True)
    terms = []
    for cj, tj in zip(c, tau):
        cr, ci = (sp.Rational(Fraction(float(v))) for v in (cj.real, cj.imag))
        tr, ti = (sp.Rational(Fraction(float(v))) for v in (tj.real, tj.imag))
        terms.append([cr, ci, tr, ti])
    # The stored doubles satisfy sum c = 1 to rounding (1e-16). Certify the constants with
    # sum c = 1 exactly: the largest real weight absorbs the rational rounding residual,
    # shared equally by its conjugate pair.
    resid = 1 - sum(t[0] for t in terms)
    jmax = int(np.argmax([abs(float(t[0])) for t in terms]))
    for j, t in enumerate(terms):
        if t[2] == terms[jmax][2] and abs(t[3]) == abs(terms[jmax][3]):
            t[0] += resid / 2
    terms = [((cr + sp.I * ci), (tr + sp.I * ti)) for cr, ci, tr, ti in terms]
    Qp = sp.Integer(1)
    for _, tj in terms:
        Qp *= (1 + tj * p)
    Pp = sp.Integer(0)
    for j, (cj, _) in enumerate(terms):
        prod = sp.Integer(1)
        for i, (_, ti) in enumerate(terms):
            if i != j:
                prod *= (1 + ti * p)
        Pp += cj * prod
    Pp, Qp = sp.expand(Pp), sp.expand(Qp)
    Pm, Qm = Pp.subs(p, -p), Qp.subs(p, -p)
    num = sp.expand(Qp * Qm - (Pp * Qm + Pm * Qp) / 2)
    num = sp.expand(sp.re(num) if num.has(sp.I) else num)
    # imaginary parts cancel for conjugate pairs; keep the real rational coefficients
    poly = sp.Poly(sp.nsimplify(num, rational=True), p)
    coeffs = poly.all_coeffs()  # highest first, in p
    deg = poly.degree()
    # polynomial in u = p^2 (odd coefficients must vanish)
    odd = [coeffs[i] for i in range(len(coeffs)) if (deg - i) % 2 == 1]
    u = sp.symbols("u")
    R = sp.Poly(sum(coeffs[i] * u ** ((deg - i) // 2) for i in range(len(coeffs))
                    if (deg - i) % 2 == 0), u)
    # factor out the root at u = 0 (D(0) = 0 when sum c = 1)
    mult0 = 0
    while R.eval(0) == 0 and R.degree() > 0:
        R = sp.Poly(sp.quo(R.as_expr(), u), u)
        mult0 += 1
    if R.degree() > 0:
        sq = sp.Poly(sp.sqf_part(R.as_expr()), u)  # square-free part: sign changes only
        n_neg = sq.count_roots(-sp.oo, 0)
    else:
        n_neg = 0
    # sign of R on (-inf, 0) away from roots: evaluate at u = -1
    sign_R = sp.sign(R.eval(-1))
    max_odd = max((abs(float(v)) for v in odd), default=0.0)
    x = np.r_[0, np.geomspace(1e-6, 1e4, 400001)]
    D = D_of(c, tau, x)
    return dict(roots_in_u_lt_0=int(n_neg), mult_u0=int(mult0), sign_R_at_u_minus1=int(sign_R),
                max_abs_odd_coeff=max_odd, D0=float(D[0]), D_min_grid=float(D[1:].min()),
                x_at_min=float(x[1:][np.argmin(D[1:])]),
                passive=bool(n_neg == 0 and sign_R * (-1) ** mult0 >= 0))


def discrete_dissipation(c, tau, beta=0.25, n_delta=200, n_theta=4001):
    """Dissipation of the discrete memory update of the paper, Eq. (update).

    y^{n+1} = e^{-h} y^n + c (1 - e^{-h}) (V^n + V^{n+1}) / 2, h = Delta / tau_j, Delta =
    |k| c_s dt; the memory stress at level n is eta0 sum_j y_j^n and the damping -eta0 V^n.
    For V^n = z^n, z = e^{i phi}: M_d(z) = sum_j c_j (1 - e^{-h_j}) (1 + z) / (2 (z - e^{-h_j})).
    Returns min over phi in [0, pi], Delta in (0, pi beta] of D_d = 1 - Re M_d, its argmin,
    and D_d at phi -> 0 (static limit)."""
    deltas = np.geomspace(np.pi * beta / n_delta, np.pi * beta, n_delta)
    x = np.r_[np.geomspace(1e-4, 1e4, n_theta), np.linspace(0.9, 1.1, 2001)]
    worst = (np.inf, None, None)
    d0 = []
    for dl in deltas:
        # frequencies phi = x Delta (dimensionless x = omega / |k| c_s) up to Nyquist
        phi = np.unique(np.r_[0.0, np.minimum(x * dl, np.pi), np.linspace(0, np.pi, 2001)])
        z = np.exp(1j * phi)
        a = np.exp(-dl / tau)
        Md = np.sum(c[None, :] * (1 - a[None, :]) * (1 + z[:, None]) / (2 * (z[:, None] - a[None, :])),
                    axis=1)
        Dd = 1 - Md.real
        d0.append(Dd[0])
        i = np.argmin(Dd[1:]) + 1
        if Dd[i] < worst[0]:
            worst = (float(Dd[i]), float(dl), float(phi[i]))
    return dict(Dd_min=worst[0], delta_at_min=worst[1], phi_at_min=worst[2],
                Dd_static_max_abs=float(np.max(np.abs(d0))))


def _outer(args):
    prob_kw, th0 = args
    prob = Problem(**prob_kw)
    prob.x_pass = prob.x_pass[::4]  # coarser passivity grid in the outer loop
    sol = least_squares(prob.residual, th0, diff_step=1e-4, max_nfev=200, xtol=1e-10, ftol=1e-12)
    return float(sol.cost), sol.x


def fit(nu=0.25, K=5, t_max=600.0, w_front=0.5, eps=1e-5, starts=None, seed=0, n_random=6,
        workers=8):
    from concurrent.futures import ProcessPoolExecutor

    prob_kw = dict(nu=nu, t_max=t_max, w_front=w_front, eps=eps)
    prob = Problem(**prob_kw)
    th_starts = []
    if starts is not None:
        th_starts.append(np.ravel([[np.log(abs(t)), np.angle(t)] for t in starts]))
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        th_starts.append(np.ravel([[np.log(rng.uniform(0.05, 3)), rng.uniform(-1.5, 1.5)]
                                   for _ in range(K)]))
    with ProcessPoolExecutor(max_workers=min(workers, len(th_starts))) as ex:
        results = list(ex.map(_outer, [(prob_kw, th) for th in th_starts]))
    costs = [rc[0] for rc in results]
    best_x = results[int(np.argmin(costs))][1]
    taus = poles_from(best_x)
    u, r, info = prob.solve_certified(taus)
    c, tau = prob.unpack(best_x, u)
    n = prob.T.size
    report = dict(Q_rms=float(np.sqrt(np.mean(r[:n] ** 2))), Q_max=float(np.max(np.abs(r[:n]))),
                  sum_c=float(np.real(c.sum())), sum_ctau=float(np.real((c * tau).sum())),
                  sum_c_over_tau=float(np.real((c / tau).sum())), S2=float(np.real((c * tau ** 2).sum())),
                  min_decay=float(np.min(np.real(1 / tau))), start_costs=costs, **info)
    xs = prob.x_sub
    Mj = np.sum(c[None, :] / (1 + tau[None, :] * (-1j * xs)[:, None]), axis=1)
    st = prob.zst + (-1j * xs) * (1 - Mj)
    report["front_stiff_relerr_max"] = float(np.max(np.abs(st.real - prob.st_ex.real) / prob.zst))
    report["front_damp_max"] = float(np.max(np.abs(st.imag)) / prob.zst)
    sub = np.linspace(0, prob.xR, 2001)
    report["subsonic_D_max"] = float(np.max(D_of(c, tau, sub)))
    return dict(c_re=c.real.tolist(), c_im=c.imag.tolist(), tau_re=tau.real.tolist(),
                tau_im=tau.imag.tolist(), tau1=prob.tau1, m0=prob.m0, nu=nu, K=K, t_max=t_max,
                w_front=w_front, eps=eps, report=report)


def poles_file(K, t_max, w_front, nu):
    return POLES / f"memory_poles_II_K{K}_pex_T{t_max:g}_w{w_front:g}_nu{nu:g}.json"


def load(path):
    d = json.loads(Path(path).read_text())
    return np.array(d["c_re"]) + 1j * np.array(d["c_im"]), np.array(d["tau_re"]) + 1j * np.array(d["tau_im"]), d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--nu", type=float, default=0.25)
    ap.add_argument("--tmax", type=float, default=600.0)
    ap.add_argument("--wfront", type=float, default=0.5)
    ap.add_argument("--random", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", default="data/memory_poles/memory_poles_II_K5_long_passive.json",
                    help="previous fit whose relaxation times seed the outer loop ('' = none)")
    args = ap.parse_args()
    starts = None
    if args.start and Path(ROOT / args.start).exists():
        c0, t0, d0 = load(ROOT / args.start)
        if len(t0) == 2 * args.K:
            starts = t0[0::2]
    res = fit(nu=args.nu, K=args.K, t_max=args.tmax, w_front=args.wfront, starts=starts,
              n_random=args.random, workers=args.workers)
    c, tau = np.array(res["c_re"]) + 1j * np.array(res["c_im"]), np.array(res["tau_re"]) + 1j * np.array(res["tau_im"])
    res["certificate"] = certify(c, tau)
    res["discrete"] = discrete_dissipation(c, tau, beta=0.25)
    out = poles_file(args.K, args.tmax, args.wfront, args.nu)
    out.write_text(json.dumps(res, indent=1))
    print(out.name)
    print(json.dumps(res["report"], indent=1))
    print(json.dumps(res["certificate"], indent=1))
    print(json.dumps(res["discrete"], indent=1))