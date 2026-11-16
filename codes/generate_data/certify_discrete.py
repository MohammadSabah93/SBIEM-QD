"""Exact passivity certificates of the SRD-QD memory, continuous and discrete (v2 of the paper).

Continuous model: M_J(p) = sum_j c_j / (1 + tau_j p); passive iff D(x) = 1 - Re M_J(ix) >= 0.

Discrete model (exact exponential update with the mid-step slip rate, step Delta = |k| c_s dt):
M_d(z) = sum_j c_j (1 - a_j)(1 + z) / (2 (z - a_j)), a_j = exp(-Delta / tau_j). With the bilinear
variable w = (z - 1)/(z + 1), i.e. w = i tan(phi/2) for z = e^{i phi},

    c_j (1 - a_j)(1 + z) / (2 (z - a_j)) = c_j / (1 + sigma_j w),   sigma_j = (1 + a_j)/(1 - a_j),

so M_d(e^{i phi}) = M_J(i tan(phi/2)) with tau_j replaced by sigma_j = coth(Delta / (2 tau_j)): the
discrete scheme at step Delta is passive iff the continuous memory with relaxation times sigma_j is
passive. |a_j| < 1 (Re 1/tau_j > 0) gives Re 1/sigma_j > 0, so the discrete poles are stable too.

Every certificate is a Sturm count, in exact rational arithmetic, of the roots on u < 0 of the
numerator R(u) of D (u = p^2 or w^2), after the exact root at u = 0 (sum c = 1) is divided out.
The stored double-precision constants are used, with sum c renormalized to exactly 1, and for the
discrete scheme the decay factors a_j are the double-precision values computed as in sbiem_bp1.

A simulation uses only the steps Delta_m = |k_m| c_s dt = 2 pi beta m / N, m = 1..N/2, so all of
its Fourier modes are certified one by one.

Writes data/analysis/certificates_v2.json.  usage: python certify_discrete.py [--workers 20]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # codes/: qd_paths.py, which adds the code folders
from qd_paths import ANALYSIS, POLES  # noqa: E402
from sbiem_bp1 import BP1, Grid  # noqa: E402



# ---------------------------------------------------------------- exact complex polynomials
def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def pmul(A, B):
    C = [(Fraction(0), Fraction(0))] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ab = cmul(a, b)
            C[i + j] = (C[i + j][0] + ab[0], C[i + j][1] + ab[1])
    return C


def frac(x):
    return Fraction(float(x))


def exact_terms(c, tau):
    """Exact rationals of (c_j, tau_j) in conjugate pairs, with sum c = 1 exactly."""
    terms = []
    for j in range(0, len(c), 2):
        cr, ci, tr, ti = frac(c[j].real), frac(c[j].imag), frac(tau[j].real), frac(tau[j].imag)
        terms.append([cr, ci, tr, ti])
    resid = 1 - 2 * sum(t[0] for t in terms)
    jmax = int(np.argmax([abs(float(t[0])) for t in terms]))
    terms[jmax][0] += resid / 2
    out = []
    for cr, ci, tr, ti in terms:
        out += [((cr, ci), (tr, ti)), ((cr, -ci), (tr, -ti))]
    return out


def numerator_R(terms):
    """Coefficients (u^0, u^1, ...) of R(u), D = R(u) / (Q(p) Q(-p)), u = p^2."""
    one = (Fraction(1), Fraction(0))
    lin = [[one, tj] for _, tj in terms]  # 1 + tau_j p
    Q = [one]
    for L in lin:
        Q = pmul(Q, L)
    P = [(Fraction(0), Fraction(0))] * len(Q)
    for j, (cj, _) in enumerate(terms):
        prod = [cj]
        for i, L in enumerate(lin):
            if i != j:
                prod = pmul(prod, L)
        P = [(P[i][0] + prod[i][0], P[i][1] + prod[i][1]) if i < len(prod) else P[i] for i in range(len(P))]
    neg = lambda A: [((-1) ** i * a[0], (-1) ** i * a[1]) for i, a in enumerate(A)]
    QQ, PQ, QP = pmul(Q, neg(Q)), pmul(P, neg(Q)), pmul(neg(P), Q)
    R = [(QQ[i][0] - (PQ[i][0] + QP[i][0]) / 2, QQ[i][1] - (PQ[i][1] + QP[i][1]) / 2) for i in range(len(QQ))]
    if any(r[1] != 0 for r in R):
        raise ValueError("imaginary part of R does not cancel: terms are not conjugate pairs")
    if any(R[i][0] != 0 for i in range(1, len(R), 2)):
        raise ValueError("odd coefficients of R do not vanish")
    return [R[i][0] for i in range(0, len(R), 2)]


def sturm_certificate(terms, method="sturm"):
    """Exact certificate of D >= 0 on the imaginary axis for M = sum c_j / (1 + s_j p).

    method "sturm": Sturm count of the square-free part on (-inf, 0]; "vas": exact isolation of the
    real roots in (-inf, 0] by continued fractions (Vincent-Akritas-Strzebonski) on the integer
    polynomial with cleared denominators, about ten times faster."""
    Ru = numerator_R(terms)
    mult0 = 0
    while Ru and Ru[0] == 0:
        Ru = Ru[1:]
        mult0 += 1
    u = sp.symbols("u")
    R = sp.Poly(list(reversed([sp.Rational(r.numerator, r.denominator) for r in Ru])), u, domain="QQ")
    if R.degree() == 0:
        n_neg = 0
    elif method == "vas":
        n_neg = len(R.clear_denoms(convert=True)[1].intervals(sup=0))
    else:
        n_neg = int(sp.Poly(sp.sqf_part(R.as_expr()), u).count_roots(-sp.oo, 0))
    sign = int(sp.sign(R.eval(-1)))
    stable = all(float((1 / complex(float(t[0]), float(t[1]))).real) > 0 for _, t in terms)
    return dict(passive=bool(n_neg == 0 and sign * (-1) ** mult0 > 0 and stable), roots_u_lt_0=n_neg,
                mult_u0=mult0, stable=stable)


def discrete_terms(c, tau, k, cs, dt):
    """Exact (c_j, sigma_j) of the discrete update for one mode; a_j as computed in sbiem_bp1."""
    rates = k * cs / tau
    a = np.exp(-rates * dt)
    base = exact_terms(c, tau)
    out = []
    for j in range(0, len(c), 2):
        ar, ai = frac(a[j].real), frac(a[j].imag)
        # sigma = (1 + a) / (1 - a), exactly in rationals (conjugate member by symmetry)
        num, den = (1 + ar, ai), (1 - ar, -ai)
        d2 = den[0] ** 2 + den[1] ** 2
        sr = (num[0] * den[0] + num[1] * den[1]) / d2
        si = (num[1] * den[0] - num[0] * den[1]) / d2
        out += [(base[j][0], (sr, si)), (base[j + 1][0], (sr, -si))]
    return out


def check_identity(c, tau, Delta):
    """Numerical check of M_d(e^{i phi}) = sum c_j / (1 + sigma_j i tan(phi/2))."""
    phi = np.linspace(1e-3, np.pi - 1e-3, 200)
    z = np.exp(1j * phi)
    a = np.exp(-Delta / tau)
    Md = np.sum(c[None] * (1 - a[None]) * (1 + z[:, None]) / (2 * (z[:, None] - a[None])), axis=1)
    sig = (1 + a) / (1 - a)
    Mb = np.sum(c[None] / (1 + sig[None] * 1j * np.tan(phi / 2)[:, None]), axis=1)
    return float(np.max(np.abs(Md - Mb)))


def load(name):
    d = json.loads((POLES / name).read_text())
    return np.array(d["c_re"]) + 1j * np.array(d["c_im"]), np.array(d["tau_re"]) + 1j * np.array(d["tau_im"]), d


def _job(args):
    name, nu, dz, lam, beta, modes, check = args
    c, tau, _ = load(name)
    p = BP1(mode="II", nu=nu)
    g = Grid(dz=dz, lam=lam)
    dt = beta * dz / p.cs
    bad, mismatch = [], []
    for m in modes:
        terms = discrete_terms(c, tau, g.k[m], p.cs, dt)
        cert = sturm_certificate(terms, "vas")
        if not cert["passive"]:
            bad.append(int(m))
        if m % check == 0 and sturm_certificate(terms, "sturm")["passive"] != cert["passive"]:
            mismatch.append(int(m))  # cross-check of the two exact methods on every check-th mode
    return name, dz, beta, len(modes), bad, mismatch


SETS = ([(f"memory_poles_II_K5_pex_T600_w0.5_nu{nu:g}.json", nu) for nu in (0.15, 0.2, 0.25, 0.3, 0.35)]
        + [(f"memory_poles_II_K{K}_pex_T600_w0.5_nu0.25.json", 0.25) for K in (2, 3, 4, 6)]
        + [(f"memory_poles_II_K5_pex_T{T}_w0.5_nu0.25.json", 0.25) for T in (40, 100, 200, 1200)]
        + [(f"memory_poles_II_K5_pex_T600_w{w}_nu0.25.json", 0.25) for w in ("0", "2")])
# (dz, lam, beta) of every simulation in the paper, per constant set
BASE = "memory_poles_II_K5_pex_T600_w0.5_nu0.25.json"
CONFIGS = {BASE: [(50.0, 160e3, 0.25), (50.0, 160e3, 0.125), (50.0, 160e3, 0.0625), (25.0, 160e3, 0.25),
                  (25.0, 160e3, 0.125), (12.5, 160e3, 0.25)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()
    out = dict(continuous=[], discrete=[], identity_check=None)
    c, tau, _ = load(BASE)
    out["identity_check"] = max(check_identity(c, tau, D) for D in (1e-3, 0.1, np.pi / 4, np.pi / 16))
    # continuous certificates and pole stability of every constant set (and the one-term version)
    for name, nu in SETS:
        c, tau, d = load(name)
        cert = sturm_certificate(exact_terms(c, tau))
        out["continuous"].append(dict(set=name, nu=nu, min_decay_rate=float(np.min(np.real(1 / tau))),
                                      max_osc=float(np.max(np.abs(np.imag(1 / tau)))),
                                      sum_c=float(np.real(c.sum())), sum_ctau=float(np.real((c * tau).sum())),
                                      sum_c_over_tau=float(np.real((c / tau).sum())), **cert))
        print(out["continuous"][-1], flush=True)
    jobs = []
    for name, nu in SETS:
        for dz, lam, beta in CONFIGS.get(name, [(50.0, 160e3, 0.25)]):
            M = Grid(dz=dz, lam=lam).k.size
            modes = np.arange(1, M)
            for i in range(0, modes.size, args.chunk):
                jobs.append((name, nu, dz, lam, beta, modes[i:i + args.chunk], 50))
    agg = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, dz, beta, n, bad, mis in ex.map(_job, jobs):
            r = agg.setdefault((name, dz, beta), dict(set=name, dz=dz, beta=beta, n_modes=0, not_passive=[],
                                                      sturm_vas_mismatch=[]))
            r["n_modes"] += n
            r["not_passive"] += bad
            r["sturm_vas_mismatch"] += mis
    out["discrete"] = list(agg.values())
    for r in out["discrete"]:
        print(r["set"], r["dz"], r["beta"], r["n_modes"], "failures:", len(r["not_passive"]), flush=True)
    out["n_discrete_certificates"] = int(sum(r["n_modes"] for r in out["discrete"]))
    out["all_passive"] = bool(all(r["passive"] for r in out["continuous"]) and
                              all(not r["not_passive"] for r in out["discrete"]))
    (ANALYSIS / "certificates_v2.json").write_text(json.dumps(out, indent=1))
    print("all passive:", out["all_passive"], "discrete certificates:", out["n_discrete_certificates"])


if __name__ == "__main__":
    main()