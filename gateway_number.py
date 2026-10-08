#!/usr/bin/env python3
"""
gateway_number -- reference implementation of the catalysis Gateway number G
============================================================================

Companion code for

    "Quantum spin regulates the catalytic cycle beyond binding energy."

Archived at:  https://doi.org/10.5281/zenodo.21440491

The Gateway number is the basis-invariant weight of non-dissipative circulation
(L, antisymmetric) relative to dissipation (D, symmetric), formed from the
relaxation operator (D + L) g with g the positive-definite local metric:

    G = sqrt( -Tr((L g)^2) / Tr((D g)^2) )                  [main text, Eq. (gateway)]

The module provides

* the invariants G and kappa for operators of any dimension, with input checks;
* the exact two-coordinate (charge--spin) results of Supplemental Material
  Sec. S1: relaxation spectrum, critical Gateway number, response-slope
  extraction and the reversed-magnetisation (+-M) pairing;
* the symmetric-triad critical Gateway number;
* a deterministic verification report reproducing every number quoted in
  Supplemental Material Secs. S1 and S6.

Run ``python gateway_number.py`` for the report (exit status 0 when every check
passes), ``python gateway_number.py --json`` for machine-readable results.

All checks use closed-form linear algebra; no finite differences.

INTEGRITY NOTE. No fabricated catalyst. The operator entries of the S6
example are illustrative normalised values. The Fe line of S6C is a dimensional
scaling illustration from zeta_SOC/Delta_cf, not a computed operator and not a
prediction; the value for a real site requires the reactive prefactor, the
dissipative coefficients and the entropy-metric normalisation, which the
companion open-system electronic-response calculation (in preparation) is
designed to supply.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.1.0"
__all__ = [
    "assemble_D", "assemble_L", "gateway_number", "circulation_rate",
    "closed_form_G", "entropy_normed", "relaxation_eigenvalues",
    "transform", "susceptibility", "gateway_from_response_slope",
    "two_coordinate_spectrum", "two_coordinate_Gcrit", "triad_Gcrit",
    "pairing_estimate", "verify_two_coordinate", "verify_s6", "main",
]

# Channel order for the triad helpers: 0 = C (chemical), 1 = E (electric), 2 = S (spin).
_RTOL = 1e-10


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------
def _as_square(name: str, M) -> np.ndarray:
    A = np.asarray(M, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains non-finite entries")
    return A


def _scale(A: np.ndarray) -> float:
    return max(1.0, float(np.max(np.abs(A))))


def _check_symmetric(name: str, A: np.ndarray, rtol: float) -> None:
    if np.max(np.abs(A - A.T)) > rtol * _scale(A):
        raise ValueError(f"{name} must be symmetric")


def _check_antisymmetric(name: str, A: np.ndarray, rtol: float) -> None:
    if np.max(np.abs(A + A.T)) > rtol * _scale(A):
        raise ValueError(f"{name} must be antisymmetric")


def _check_metric(g: np.ndarray, rtol: float) -> None:
    _check_symmetric("g", g, rtol)
    if np.min(np.linalg.eigvalsh((g + g.T) / 2)) <= 0.0:
        raise ValueError("g must be positive definite")


def _prepare(D, L, g, check: bool, rtol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = _as_square("D", D)
    L = _as_square("L", L)
    n = D.shape[0]
    g = np.eye(n) if g is None else _as_square("g", g)
    if L.shape != D.shape or g.shape != D.shape:
        raise ValueError(f"D, L and g must share one shape, got {D.shape}, {L.shape}, {g.shape}")
    if check:
        _check_symmetric("D", D, rtol)
        _check_antisymmetric("L", L, rtol)
        _check_metric(g, rtol)
    return D, L, g


# ---------------------------------------------------------------------------
# operator assembly (triad, channel order C, E, S)
# ---------------------------------------------------------------------------
def assemble_D(D_CC: float, D_EE: float, D_SS: float, D_CE: float) -> np.ndarray:
    """Symmetric dissipative block of the {C,E,S} triad; spin off-diagonals vanish by parity."""
    return np.array([[D_CC, D_CE, 0.0],
                     [D_CE, D_EE, 0.0],
                     [0.0, 0.0, D_SS]], dtype=float)


def assemble_L(L_SC: float, L_SE: float) -> np.ndarray:
    """Antisymmetric circulating block of the {C,E,S} triad; L_CE = 0 by parity."""
    return np.array([[0.0, 0.0, -L_SC],
                     [0.0, 0.0, -L_SE],
                     [L_SC, L_SE, 0.0]], dtype=float)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def gateway_number(D, L, g=None, *, check: bool = True, rtol: float = _RTOL) -> float:
    """Gateway number G = sqrt(-Tr((Lg)^2) / Tr((Dg)^2)) for operators of any dimension.

    ``g`` defaults to the identity (entropy-normed frame). Raises ``ValueError`` for
    inconsistent shapes, broken symmetry, a non-positive-definite metric or vanishing
    dissipation.
    """
    D, L, g = _prepare(D, L, g, check, rtol)
    Dg, Lg = D @ g, L @ g
    den = float(np.trace(Dg @ Dg))
    if den <= 0.0:
        raise ValueError("Tr((D g)^2) must be positive (non-vanishing dissipation)")
    num = max(0.0, -float(np.trace(Lg @ Lg)))   # -Tr((Lg)^2) >= 0; clip round-off
    return float(np.sqrt(num / den))


def circulation_rate(L, g=None, *, check: bool = True, rtol: float = _RTOL) -> float:
    """Circulation rate kappa = sqrt(-Tr((Lg)^2)/2); the spectral radius of Lg for rank-two L."""
    L = _as_square("L", L)
    g = np.eye(L.shape[0]) if g is None else _as_square("g", g)
    if g.shape != L.shape:
        raise ValueError(f"L and g must share one shape, got {L.shape}, {g.shape}")
    if check:
        _check_antisymmetric("L", L, rtol)
        _check_metric(g, rtol)
    Lg = L @ g
    return float(np.sqrt(max(0.0, -0.5 * float(np.trace(Lg @ Lg)))))


def closed_form_G(D_CC: float, D_EE: float, D_SS: float, D_CE: float,
                  L_SC: float, L_SE: float) -> float:
    """Closed form of G for the {C,E,S} triad in the entropy-normed frame [main text, Eq. (gateway-triad)]."""
    den = D_CC ** 2 + D_EE ** 2 + D_SS ** 2 + 2.0 * D_CE ** 2
    if den <= 0.0:
        raise ValueError("dissipative block must be non-zero")
    return float(np.sqrt(2.0 * (L_SC ** 2 + L_SE ** 2) / den))


def _sqrtm_spd(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return V @ np.diag(np.sqrt(w)) @ V.T


def entropy_normed(A, g) -> np.ndarray:
    """Entropy-normed operator A~ = g^{1/2} A g^{1/2}, similar to A g."""
    A = _as_square("A", A)
    g = _as_square("g", g)
    _check_metric(g, _RTOL)
    gh = _sqrtm_spd(g)
    return gh @ A @ gh


def _G_of_matrix(M: np.ndarray) -> float:
    """G of a general matrix from its symmetric and antisymmetric parts (identity metric)."""
    S, K = (M + M.T) / 2, (M - M.T) / 2
    den = float(np.trace(S @ S))
    if den <= 0.0:
        raise ValueError("symmetric part must be non-zero")
    return float(np.sqrt(max(0.0, -float(np.trace(K @ K))) / den))


def relaxation_eigenvalues(D, L, g=None, *, check: bool = True) -> np.ndarray:
    """Eigenvalues of the relaxation operator (D + L) g."""
    D, L, g = _prepare(D, L, g, check, _RTOL)
    return np.linalg.eigvals((D + L) @ g)


def transform(D, L, g, P) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change of coordinates x -> P x: D -> P D P^T, L -> P L P^T, g -> P^{-T} g P^{-1}."""
    P = _as_square("P", P)
    if abs(np.linalg.det(P)) == 0.0:
        raise ValueError("P must be invertible")
    Pinv = np.linalg.inv(P)
    return P @ D @ P.T, P @ L @ P.T, Pinv.T @ g @ Pinv


# ---------------------------------------------------------------------------
# response and two-coordinate theory [Supplemental Material, Sec. S1]
# ---------------------------------------------------------------------------
def susceptibility(A, g, omega: float) -> np.ndarray:
    """chi(omega) = (A g - i omega)^{-1} A for perturbations ~ exp(-i omega t), dx = chi h."""
    A = _as_square("A", A)
    g = _as_square("g", g)
    return np.linalg.inv(A @ g - 1j * omega * np.eye(A.shape[0])) @ A


def gateway_from_response_slope(slope) -> float:
    """G from the low-frequency slope S = A~^{-1} of the normalised inverse response.

    The normalised inverse response is I - i omega A~^{-1}. The function inverts the
    slope to A~ and forms G, which is valid in any dimension. For two coordinates
    G[A~^{-1}] = G[A~], so the slope itself gives G; for three or more coordinates
    the inversion is required.
    """
    S = _as_square("slope", slope)
    return _G_of_matrix(np.linalg.inv(S))


def two_coordinate_spectrum(mu1: float, mu2: float, kappa: float) -> Tuple[complex, complex]:
    """Relaxation eigenvalues of a two-coordinate block [Supplemental Material, Sec. S1]."""
    disc = complex(((mu1 - mu2) / 2.0) ** 2 - kappa ** 2)
    root = np.sqrt(disc)
    mean = (mu1 + mu2) / 2.0
    return complex(mean + root), complex(mean - root)


def two_coordinate_Gcrit(mu1: float, mu2: float) -> float:
    """Two-coordinate critical Gateway number |mu1 - mu2| / sqrt(2 (mu1^2 + mu2^2))."""
    den = 2.0 * (mu1 ** 2 + mu2 ** 2)
    if den <= 0.0:
        raise ValueError("dissipative eigenvalues must not both vanish")
    return float(abs(mu1 - mu2) / np.sqrt(den))


def triad_Gcrit(r: float) -> float:
    """Critical Gateway number of the symmetric triad, r = D_SS / D_CC [Supplemental Material, Sec. S1]."""
    if r < 0.0:
        raise ValueError("r = D_SS / D_CC must be non-negative")
    return float(abs(r - 1.0) / np.sqrt(2.0 * (r * r + 2.0)))


def pairing_estimate(A_plus, A_minus, i: int, j: int) -> Tuple[float, float]:
    """Reversed-magnetisation pairing for the even--odd pair (i, j).

    Returns (L_ij, eps_sym): the antisymmetric coefficient, even in M, and the
    paired average of the symmetric part, which vanishes for an exact calculation
    and measures the numerical error.
    """
    Ap = _as_square("A_plus", A_plus)
    Am = _as_square("A_minus", A_minus)
    L_ij = 0.25 * (Ap[i, j] - Ap[j, i]) + 0.25 * (Am[i, j] - Am[j, i])
    eps = 0.25 * (Ap[i, j] + Ap[j, i]) + 0.25 * (Am[i, j] + Am[j, i])
    return float(L_ij), float(eps)


# ---------------------------------------------------------------------------
# verification [Supplemental Material, Secs. S1 and S6]
# ---------------------------------------------------------------------------
_J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    M = rng.normal(size=(n, n))
    return M @ M.T + 0.05 * np.eye(n)


def verify_two_coordinate(seed: int = 7, samples: int = 2000) -> Dict[str, float]:
    """Closed-form checks of the two-coordinate theory and the triad threshold."""
    rs = np.random.default_rng(seed)
    e_spec = e_inv = e_resp = 0.0
    mismatch = 0
    for _ in range(samples):
        g2, D2, k = _spd(rs, 2), _spd(rs, 2), rs.uniform(0.0, 3.0)
        A2 = D2 + k * _J2
        At = entropy_normed(A2, g2)
        mu = np.sort(np.linalg.eigvalsh((At + At.T) / 2))[::-1]
        kap = circulation_rate(k * _J2, g2)
        pred = np.sort_complex(np.array(two_coordinate_spectrum(mu[0], mu[1], kap)))
        e_spec = max(e_spec, float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(A2 @ g2)) - pred))))
        Gc = two_coordinate_Gcrit(mu[0], mu[1])
        G2 = _G_of_matrix(At)
        cplx = ((mu[0] - mu[1]) / 2.0) ** 2 - kap ** 2 < 0.0
        if abs(G2 - Gc) > 1e-9 and cplx != (G2 > Gc):
            mismatch += 1
        e_inv = max(e_inv, abs(_G_of_matrix(np.linalg.inv(At)) - G2))
        w = rs.uniform(0.01, 5.0)
        chi = susceptibility(A2, g2, w)
        e_resp = max(e_resp, float(np.max(np.abs(np.linalg.inv(chi) - (g2 - 1j * w * np.linalg.inv(A2))))))

    e_tri = 0.0
    for r in np.linspace(0.05, 10.0, 200):
        if abs(r - 1.0) < 1e-6:
            continue
        Dt = np.diag([1.0, 1.0, r])
        a = triad_Gcrit(r) * np.sqrt(np.trace(Dt @ Dt) / 4.0)   # G^2 = 4 a^2 / Tr(D^2), L_SC = L_SE = a
        c3, c2, c1, c0 = np.poly(Dt + assemble_L(a, a))
        disc3 = (18 * c3 * c2 * c1 * c0 - 4 * c2 ** 3 * c0 + c2 ** 2 * c1 ** 2
                 - 4 * c3 * c1 ** 3 - 27 * c3 ** 2 * c0 ** 2)
        e_tri = max(e_tri, abs(disc3) / (1.0 + r) ** 6)

    dev = []
    for _ in range(samples):
        D3 = np.zeros((3, 3))
        D3[:2, :2] = _spd(rs, 2)
        D3[2, 2] = rs.uniform(0.05, 3.0)
        A3 = D3 + assemble_L(*rs.normal(size=2))
        dev.append(abs(_G_of_matrix(np.linalg.inv(A3)) - _G_of_matrix(A3)))

    e_pair = 0.0
    for _ in range(500):
        a_, s1 = rs.normal(size=2)
        AM = lambda M: np.array([[1.3, s1 * M + a_], [s1 * M - a_, 0.7]])  # noqa: E731
        Ap, Am = AM(+1.0), AM(-1.0)
        L_est, eps = pairing_estimate(Ap, Am, 0, 1)
        e_pair = max(e_pair, abs(L_est - a_), abs(eps), abs(Ap[0, 1] + Am[1, 0]))

    return {
        "spectrum_max_diff": e_spec,
        "complex_iff_G_gt_Gcrit_mismatches": float(mismatch),
        "samples": float(samples),
        "inversion_identity_max_diff": e_inv,
        "response_formula_max_diff": e_resp,
        "triad_discriminant_max_scaled": e_tri,
        "triad_inversion_fails_fraction": float(np.mean(np.array(dev) > 1e-9)),
        "pairing_max_diff": e_pair,
    }


# Illustrative normalised operator of Supplemental Material Sec. S6 (entropy-normed g = I).
S6_D = dict(D_CC=1.00, D_EE=1.00, D_SS=1.00, D_CE=0.30)
S6_L = dict(L_SC=0.20, L_SE=0.15)
S6_SUSC = dict(dm_dmu=0.80, dm_dphi=0.60)      # gamma = 0.25 reproduces S6_L
S6_GAMMAS = (0.00, 0.02, 0.05, 0.10, 0.20, 0.25)


def verify_s6(seed: int = 1) -> Dict[str, object]:
    """Reproduce every number of Supplemental Material Sec. S6."""
    D = assemble_D(**S6_D)
    L = assemble_L(**S6_L)
    g = np.eye(3)
    G = gateway_number(D, L, g)
    Gcf = closed_form_G(**S6_D, **S6_L)
    kap = circulation_rate(L, g)
    eig = np.linalg.eigvals(L @ g)
    eig = eig[np.argsort(np.abs(eig.imag))]

    rng = np.random.default_rng(seed)
    P = rng.normal(size=(3, 3))
    while abs(np.linalg.det(P)) < 0.3:
        P = rng.normal(size=(3, 3))
    Dt, Lt, gt = transform(D, L, g, P)
    Gt = gateway_number(Dt, Lt, gt)
    fro = float(np.sqrt(-np.trace(L @ L) / np.trace(D @ D)))
    frot = float(np.sqrt(-np.trace(Lt @ Lt) / np.trace(Dt @ Dt)))

    scaling = []
    for gm in S6_GAMMAS:
        Lg = assemble_L(gm * S6_SUSC["dm_dmu"], gm * S6_SUSC["dm_dphi"])
        scaling.append((gm, gateway_number(D, Lg, g)))

    return {"G": G, "G_closed_form": Gcf, "kappa": kap, "eig_Lg": eig,
            "detP": float(np.linalg.det(P)), "G_transformed": Gt,
            "frobenius": fro, "frobenius_transformed": frot, "scaling": scaling}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
_TOL = {"spectrum_max_diff": 1e-10, "inversion_identity_max_diff": 1e-10,
        "response_formula_max_diff": 1e-9, "triad_discriminant_max_scaled": 1e-10,
        "pairing_max_diff": 1e-12}


def _checks_pass(v: Dict[str, float], s6: Dict[str, object]) -> bool:
    ok = all(v[k] <= t for k, t in _TOL.items())
    ok &= v["complex_iff_G_gt_Gcrit_mismatches"] == 0.0
    ok &= v["triad_inversion_fails_fraction"] > 0.99
    ok &= abs(s6["G"] - s6["G_closed_form"]) < 1e-12
    ok &= abs(s6["G"] - s6["G_transformed"]) < 1e-12
    return bool(ok)


def _print_report(v: Dict[str, float], s6: Dict[str, object]) -> None:
    D, L = S6_D, S6_L
    print("=" * 72)
    print("S1X  TWO-COORDINATE THEORY AND TRIAD THRESHOLD  (exact, Sec. S1)")
    print("=" * 72)
    n = int(v["samples"])
    print(f"  spectrum lambda_+- vs eigenvalues        max |diff| = {v['spectrum_max_diff']:.1e}")
    print(f"  complex spectrum <=> G > G_crit          mismatches = {int(v['complex_iff_G_gt_Gcrit_mismatches'])} / {n}")
    print(f"  inversion identity G[A~^-1] = G[A~]      max |diff| = {v['inversion_identity_max_diff']:.1e}")
    print(f"  response chi^-1 = g - i w A^-1           max |diff| = {v['response_formula_max_diff']:.1e}")
    print(f"  triad G_crit: cubic discriminant at kappa_c   max |scaled| = {v['triad_discriminant_max_scaled']:.1e}")
    print(f"  triad: G[A^-1] differs from G[A] in {v['triad_inversion_fails_fraction']:.0%} of {n} random Gateway triads")
    print(f"  +-M pairing recovers L_ES, eps_sym = 0   max |diff| = {v['pairing_max_diff']:.1e}")
    print()
    print("=" * 72)
    print("S6A  MATHEMATICAL VERIFICATION  (exact)")
    print("=" * 72)
    G = s6["G"]
    print(f"  closed form vs trace invariant : {s6['G_closed_form']:.10f}  vs  {G:.10f}"
          f"   (|diff| = {abs(G - s6['G_closed_form']):.1e})")
    print("  spectrum of L g                : "
          + ", ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in s6["eig_Lg"])
          + f"   (kappa = {s6['kappa']:.5f})")
    print(f"  basis invariance   det(P) = {s6['detP']:+.4f}")
    print(f"    Original    G = {G:.5f}")
    print(f"    Transformed G = {s6['G_transformed']:.5f}        (|diff| = {abs(G - s6['G_transformed']):.1e})")
    print(f"    (bare Frobenius ratio, by contrast: {s6['frobenius']:.5f} -> "
          f"{s6['frobenius_transformed']:.5f}: NOT invariant)")
    print()
    print("=" * 72)
    print("S6B  ILLUSTRATIVE IMPLEMENTATION  (illustrative normalised operator)")
    print("=" * 72)
    print(f"  D = [[{D['D_CC']},{D['D_CE']},0],[{D['D_CE']},{D['D_EE']},0],[0,0,{D['D_SS']}]]")
    print(f"  L = skew(L_SC={L['L_SC']}, L_SE={L['L_SE']})    (L_CE = 0 by parity)")
    print("  g = I")
    print(f"  --> Gateway number  G = {G:.5f}   (illustrative; not a specific catalyst)")
    print()
    print("=" * 72)
    print("S6C  PHYSICAL SCALING  (dimensional scaling illustration, NOT a computed G)")
    print("=" * 72)
    print(f"  L_SC = gamma * dm/dmu,  L_SE = gamma * dm/dphi   "
          f"(dm/dmu={S6_SUSC['dm_dmu']:.2f}, dm/dphi={S6_SUSC['dm_dphi']:.2f})")
    print("   gamma      G          note")
    for gm, Gg in s6["scaling"]:
        note = "  <- 3d Fe scale (zeta/Delta ~ 1e-1)" if gm in (0.05, 0.10) \
            else ("  == S6B operator" if gm == 0.25 else "")
        print(f"   {gm:5.2f}   {Gg:.5f}{note}")
    print()
    print("  G is exactly linear in gamma, and G(0)=0 (pure dissipation).")
    print("  For a 3d Fe site, zeta_SOC/Delta_cf ~ 1e-1, so G ~ 1e-2 .. 1e-1:")
    print("  a dimensional scaling illustration only, not a prediction for a specific catalyst.")
    print("=" * 72)


def _to_jsonable(s6: Dict[str, object]) -> Dict[str, object]:
    out = dict(s6)
    out["eig_Lg"] = [[float(e.real), float(e.imag)] for e in s6["eig_Lg"]]
    out["scaling"] = [[float(gm), float(Gg)] for gm, Gg in s6["scaling"]]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verification report for the catalysis Gateway number (Supplemental Material Secs. S1 and S6).")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    v = verify_two_coordinate()
    s6 = verify_s6()
    ok = _checks_pass(v, s6)
    if args.json:
        print(json.dumps({"version": __version__, "all_checks_pass": ok,
                          "two_coordinate": v, "s6": _to_jsonable(s6)}, indent=2))
    else:
        _print_report(v, s6)
        if not ok:
            print("ONE OR MORE CHECKS FAILED", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
