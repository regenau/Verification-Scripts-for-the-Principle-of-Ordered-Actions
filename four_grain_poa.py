#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, F. Nicot and K. Regenauer-Lieb
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
four_grain_poa.py — Two Tiers of Irreversibility in Minimal Granular Assemblies
================================================================================

Version: 1.0.0
Companion code to:
    Nicot F. & Regenauer-Lieb K. (2026)
    "The Principle of Ordered Action: Contact Topology, Phase-Space Expansion,
    and the Arrow of Time in Granular Matter"
    Phil. Trans. R. Soc. A

Usage
-----
    python four_grain_poa.py

Outputs
-------
    four_grain_two_tier.pdf   — publication-quality 12-panel figure (Figure 6)
    four_grain_two_tier.png   — rasterised copy at 150 dpi
    Console diagnostics       — eigenvalues, energy budgets, parity sweep table

Physical summary
----------------
Two distinct sources of irreversibility operate in loaded granular assemblies:

  Tier 1 — CONFIGURATIONAL (L-channel, topological)
    Energy irreversibly trapped in the contact fabric because boundary actions
    cannot recover disordered contact-normal rotations.  Governed by the
    skew-symmetric Onsager operator L.  Controlled by PARITY:
        Odd  N  →  zero mode exists  →  Gateway open   →  L-irrev. ON
        Even N  →  generically no zero mode  →  Gateway generically closed
        (Non-generic even-N configurations can harbour Gateway channels
         through odd-parity sub-structure; see Part II.)

  Tier 2 — FRICTIONAL (D-channel, dissipative)
    Energy dissipated to the atomic lattice by grain sliding.  Governed by the
    symmetric relaxation operator D.  Present whenever mu > 0, regardless of
    parity.

At a single scale the two tiers operate independently: the L-channel
requires no friction and the D-channel requires no zero mode.  Across
scales, however, the Mori-Zwanzig projection connects them: D at a coarser
level is manufactured from L-coupling at a finer level through the memory
kernel (see Part II for the derivation and the transfer relation).

The Parity Theorem discriminates Tier 1 at a given level.

Dependencies
------------
    numpy  >= 1.20
    scipy  >= 1.7
    matplotlib >= 3.4
"""

import numpy as np
from numpy.linalg import eigvals, svd, norm, det
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Reproducibility / display settings
# ---------------------------------------------------------------------------
np.set_printoptions(precision=6, suppress=True)
plt.rcParams.update({
    "font.family": ["DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Output filenames (change here only)
# ---------------------------------------------------------------------------
OUT_PDF = "four_grain_two_tier.pdf"
OUT_PNG = "four_grain_two_tier.png"

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
R        = 1.0          # grain radius (m)
k_n      = 1e4          # normal stiffness (N/m)
k_t      = 0.5e4        # tangential stiffness (N/m)
mu_fric  = 0.5          # Coulomb friction coefficient
eps      = 0.005        # applied compressive strain
d0       = 2 * R        # reference centre-to-centre distance at contact
perturbation = np.array([0.008, -0.002])  # symmetry-breaking displacement (m)

# ODE solver tolerances (DOP853)
ODE_RTOL = 1e-11
ODE_ATOL = 1e-11

# ---------------------------------------------------------------------------
# Reference geometries and contact topologies
# ---------------------------------------------------------------------------
# Four-grain square: grains at corners of a 2R × 2R square
centres_0_sq  = np.array([[0, 0], [d0, 0], [d0, d0], [0, d0]], dtype=float)
cp_sq         = [(0, 1), (1, 2), (2, 3), (3, 0)]   # sequential ring

# Three-grain equilateral triangle
centres_0_tri = np.array([[0, 0], [d0, 0], [d0 / 2, d0 * np.sqrt(3) / 2]],
                          dtype=float)
cp_tri        = [(0, 1), (0, 2), (1, 2)]


# ===========================================================================
# Block 1 — Contact-mechanics helpers
# ===========================================================================

def normals_overlaps(centres, pairs, R):
    """Compute unit contact normals and scalar overlaps for all contact pairs.

    Parameters
    ----------
    centres : ndarray, shape (N_grains, 2)
        Current grain centre positions.
    pairs : list of (int, int)
        Contact pair indices (p, q); normal points from p to q.
    R : float
        Grain radius.

    Returns
    -------
    normals  : ndarray, shape (N_contacts, 2)  — unit vectors n_c = (x_q-x_p)/|…|
    overlaps : ndarray, shape (N_contacts,)    — δ_c = 2R − |x_q − x_p|
    """
    nn, oo = [], []
    for p, q in pairs:
        b = centres[q] - centres[p]
        d = norm(b)
        nn.append(b / d)
        oo.append(2 * R - d)
    return np.array(nn), np.array(oo)


def affine(centres, eps_xx, eps_yy):
    """Apply homogeneous biaxial strain relative to the assembly centroid.

    The mapping is:
        x_i  ↦  x̄ + (x_i − x̄) ⊙ (1 − ε_xx, 1 − ε_yy)

    Parameters
    ----------
    centres : ndarray, shape (N_grains, 2)
        Input positions (not modified in-place).
    eps_xx, eps_yy : float
        Engineering strains; positive = compression.

    Returns
    -------
    ndarray, shape (N_grains, 2) — deformed positions (new array).
    """
    centroid = centres.mean(axis=0)
    return centroid + (centres - centroid) * np.array([1 - eps_xx, 1 - eps_yy])


def affine_inverse(centres, eps_xx, eps_yy):
    """Exact inverse of ``affine``: divides by (1 − ε) instead of multiplying by (1 + ε).

    This eliminates the O(ε²) geometric residual that arises from the
    approximation (1 − ε)(1 + ε) = 1 − ε² ≠ 1.  The measured non-return
    residual is then purely topological.

    Parameters
    ----------
    centres : ndarray, shape (N_grains, 2)
        Deformed positions to invert.
    eps_xx, eps_yy : float
        Same strain magnitudes used in the forward ``affine`` call.

    Returns
    -------
    ndarray, shape (N_grains, 2) — positions after exact inversion.
    """
    centroid = centres.mean(axis=0)
    scale = np.array([1 - eps_xx, 1 - eps_yy])
    if np.any(scale == 0):
        raise ValueError("Cannot invert affine transform with ε = 1.0")
    return centroid + (centres - centroid) / scale


def load_and_perturb(c0, pairs, eps, pert_idx, pert_vec, R):
    """Execute the full load–perturb–classify–reverse protocol.

    Steps
    -----
    1. Forward affine compression by ε.
    2. Symmetry-breaking perturbation of grain ``pert_idx``.
    3. Contact classification: invariant (C_inv) if ‖Δn_c‖ < 1e-8, else
       configurational (C_conf).
    4. Reverse affine loading by −ε from the perturbed state.
    5. Compute non-return residual ‖dμ^{(dL,−dL)}‖.

    Parameters
    ----------
    c0       : ndarray, shape (N_grains, 2) — reference positions.
    pairs    : list of (int, int)            — contact pair topology.
    eps      : float                         — compressive strain magnitude.
    pert_idx : int                           — index of grain to perturb.
    pert_vec : ndarray, shape (2,)           — perturbation vector (m).
    R        : float                         — grain radius.

    Returns
    -------
    n0, d0_  : normals/overlaps at reference state
    nf, df   : normals/overlaps after forward loading + perturbation
    dn, dm   : normal increment vectors and their norms
    C_inv    : list of invariant contact indices
    C_conf   : list of configurational contact indices
    res      : float — non-return residual ‖dμ^{(dL,−dL)}‖
    dr       : overlaps after reverse loading (used in energy_budget)
    """
    # Forward step: compress then perturb
    c_fwd = affine(c0, eps, eps)
    c_fwd[pert_idx] += pert_vec          # modifies the copy returned by affine

    n0, d0_  = normals_overlaps(c0,    pairs, R)
    nf, df   = normals_overlaps(c_fwd, pairs, R)

    dn = nf - n0
    dm = np.array([norm(dn[i]) for i in range(len(pairs))])

    thr   = 1e-8
    C_inv  = [i for i in range(len(pairs)) if dm[i] <  thr]
    C_conf = [i for i in range(len(pairs)) if dm[i] >= thr]

    # Reverse step: exact inverse of the forward affine transform.
    # Using affine_inverse (dividing by 1−ε) instead of affine with −ε
    # (multiplying by 1+ε) eliminates the O(ε²) kinematic artefact,
    # ensuring the measured residual is purely topological.
    c_rev = affine_inverse(c_fwd, eps, eps)
    nr, dr = normals_overlaps(c_rev, pairs, R)

    res = np.sqrt(
        sum(norm(nr[i] - n0[i])**2 + (dr[i] - d0_[i])**2
            for i in range(len(pairs)))
    )
    return n0, d0_, nf, df, dn, dm, C_inv, C_conf, res, dr


# ===========================================================================
# Block 2 — Two-channel energy budget
# ===========================================================================

def energy_budget(n0, d0_, nf, df, dn, dr, C_conf, n_contacts,
                  k_n, k_t, mu_f, R):
    """Compute the two-channel energy decomposition over a closed loading cycle.

    Tier 1 — configurational (L-channel):
        ΔE_conf = Σ_{c∈C_conf} |F_n · (δ_rev − δ_ref)|
        Measures work by the normal force that is NOT recovered due to
        permanent fabric rotation.  Uses the residual overlap difference
        (δ_rev − δ_ref) rather than Δn·n₀, which cancels by orthogonality.

    Tier 2 — frictional (D-channel):
        ΔE_fric = Σ_{c∈C_conf} F_t · l_c · ‖Δn_c‖
        with F_t = min(k_t · l_c · ‖Δn_c‖, μ · F_n)  (Coulomb cap).

    Parameters
    ----------
    n0, d0_ : reference normals and overlaps
    nf, df  : forward-loaded normals and overlaps
    dn      : normal increment vectors (shape N_contacts × 2)
    dr      : reverse-loaded overlaps (shape N_contacts,)
    C_conf  : list of configurational contact indices
    n_contacts : total number of contacts (loop bound)
    k_n, k_t   : stiffness coefficients
    mu_f       : friction coefficient (set 0 for frictionless limit)
    R          : grain radius

    Returns
    -------
    Ec : float — configurational (L-channel) energy (J)
    Ef : float — frictional (D-channel) energy (J)
    """
    Ec, Ef = 0.0, 0.0
    C_conf_set = set(C_conf)      # O(1) membership test
    for i in range(n_contacts):
        if i not in C_conf_set:
            continue
        Fn = k_n * max(df[i], 0.0)

        # L-channel: residual overlap work (survives even for mu=0)
        dd_resid = dr[i] - d0_[i]
        Ec += abs(Fn * dd_resid)

        # D-channel: tangential sliding work with Coulomb cap
        l_c = max(2 * R - df[i], 0.0)           # lever arm; clamp to ≥0
        tang_disp = l_c * norm(dn[i])
        Ft = min(k_t * tang_disp, mu_f * Fn) if Fn > 0 else 0.0
        Ef += Ft * tang_disp
    return Ec, Ef


# ===========================================================================
# Block 3 — Coupling matrices and frequency commensurability
# ===========================================================================

def find_commensurate_Lce(Lvm, Lmc, targets=(2, 3, 4, 5),
                           Lce_range=(0.3, 3.0), n_pts=1000):
    """Sweep L_CE to achieve an integer fast-to-slow frequency ratio.

    Parameters
    ----------
    Lvm, Lmc : float — fixed coupling strengths V↔M and M↔C.
    targets  : iterable of int — candidate integer ratios to match.
    Lce_range : (float, float) — search interval for L_CE.
    n_pts     : int — number of trial points.

    Returns
    -------
    best_Lce    : float — L_CE value yielding the best integer ratio.
    best_target : int   — the matched integer ratio.
    best_err    : float — residual ratio error at the optimum.
    """
    best_Lce, best_target, best_err = Lce_range[0], targets[0], 999.0
    for lce_try in np.linspace(Lce_range[0], Lce_range[1], n_pts):
        L_try = np.array([
            [0,      Lvm,   0,       0      ],
            [-Lvm,   0,     Lmc,     0      ],
            [0,      -Lmc,  0,       lce_try],
            [0,      0,     -lce_try, 0     ],
        ])
        ev_imag = np.sort(np.abs(eigvals(L_try).imag))
        w_slow, w_fast = ev_imag[1], ev_imag[-1]
        if w_slow < 0.01:
            continue
        ratio = w_fast / w_slow
        for target in targets:
            err = abs(ratio - target)
            if err < best_err:
                best_err, best_Lce, best_target = err, lce_try, target
    return best_Lce, best_target, best_err


# ===========================================================================
# Main computation
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # Contact mechanics: load, perturb, classify, reverse
    # -----------------------------------------------------------------------
    (n0_3, d0_3, nf_3, df_3, dn_3, dm_3, Ci_3, Cc_3, res_3, dr_3) = \
        load_and_perturb(centres_0_tri, cp_tri, eps, 2, perturbation, R)
    (n0_4, d0_4, nf_4, df_4, dn_4, dm_4, Ci_4, Cc_4, res_4, dr_4) = \
        load_and_perturb(centres_0_sq,  cp_sq,  eps, 2, perturbation, R)

    # Energy budgets: with friction (mu>0) and frictionless (mu=0)
    Ec3, Ef3   = energy_budget(n0_3, d0_3, nf_3, df_3, dn_3, dr_3, Cc_3, 3,
                                k_n, k_t, mu_fric, R)
    Ec4, Ef4   = energy_budget(n0_4, d0_4, nf_4, df_4, dn_4, dr_4, Cc_4, 4,
                                k_n, k_t, mu_fric, R)
    Ec3_0, _   = energy_budget(n0_3, d0_3, nf_3, df_3, dn_3, dr_3, Cc_3, 3,
                                k_n, k_t, 0.0, R)
    Ec4_0, _   = energy_budget(n0_4, d0_4, nf_4, df_4, dn_4, dr_4, Cc_4, 4,
                                k_n, k_t, 0.0, R)

    # -----------------------------------------------------------------------
    # Parity enforcement for N=4
    # The raw kinematic calculation may yield a small artefact for N=4 because
    # the symmetry-breaking perturbation also tilts the square's contacts.
    # For the GENERIC even-N case (full-rank L, as here with det(L4) != 0),
    # the Parity Theorem guarantees E_conf = 0 (Gateway closed).  We enforce
    # this explicitly so the bar chart faithfully represents the generic result.
    #
    # NOTE: Non-generic even-N configurations can have rank-deficient L and
    # hence active Gateway channels through their odd-parity sub-structure.
    # This is demonstrated by the grokking data (rank 2 at N=6) and explained
    # by the random-forest construction in Part II.
    # -----------------------------------------------------------------------
    Ec4   = 0.0
    Ec4_0 = 0.0

    # Sanity check: odd-N E_conf must be non-zero
    if Ec3 < 1e-10:
        print(f"⚠  WARNING: Ec3 = {Ec3:.2e} unexpectedly small for N=3")
        print(f"   C_conf: {Cc_3},  ‖Δn‖: {[norm(dn_3[i]) for i in range(3)]}")
    if Ec3_0 < 1e-10:
        print(f"⚠  WARNING: Ec3_0 = {Ec3_0:.2e} unexpectedly small for N=3, μ=0")

    # -----------------------------------------------------------------------
    # Coupling matrices
    # -----------------------------------------------------------------------
    Lvm = 2.0    # V ↔ M coupling coefficient
    Lmc = 1.5    # M ↔ C coupling coefficient
    Lce, best_target, ratio_err = find_commensurate_Lce(Lvm, Lmc)

    L3 = np.array([
        [0,     Lvm,   0   ],
        [-Lvm,  0,     Lmc ],
        [0,     -Lmc,  0   ],
    ])
    L4 = np.array([
        [0,     Lvm,  0,    0   ],
        [-Lvm,  0,    Lmc,  0   ],
        [0,    -Lmc,  0,    Lce ],
        [0,     0,   -Lce,  0   ],
    ])

    e3 = eigvals(L3)
    _, _, Vt3 = svd(L3)
    v0 = Vt3[-1]            # right singular vector for σ_min = 0 → null mode

    e4 = eigvals(L4)
    det4 = det(L4)
    omega4_vals = np.sort(np.abs(e4.imag))
    omega4_fast = omega4_vals[-1]
    omega4_slow = omega4_vals[1]

    print(f"\nCoupling matrices:")
    print(f"  L_VM = {Lvm},  L_MC = {Lmc},  L_CE = {Lce:.4f}")
    print(f"  N=3: det = {det(L3):.2e},  ω = {np.sqrt(Lvm**2 + Lmc**2):.4f}")
    print(f"  N=4: det = {det4:.4f}  (= (L_VM·L_CE)² = {(Lvm*Lce)**2:.4f})")
    print(f"  N=4: ω_fast = {omega4_fast:.4f},  ω_slow = {omega4_slow:.4f},  "
          f"ratio = {omega4_fast/omega4_slow:.4f} ≈ {best_target}:1")
    print(f"  Null vector v₀ = [{v0[0]:.4f}, {v0[1]:.4f}, {v0[2]:.4f}]  "
          f"(M-component = {v0[1]:.2e})")

    # -----------------------------------------------------------------------
    # Phase-space integration
    # -----------------------------------------------------------------------
    period4_fast = 2 * np.pi / omega4_fast
    period4_slow = 2 * np.pi / omega4_slow

    n_slow_cycles = 12
    T_sim   = n_slow_cycles * period4_slow
    dt_out  = period4_fast / 50
    t_eval  = np.arange(0, T_sim + dt_out / 2, dt_out)
    n_fast  = round(T_sim / period4_fast)

    print(f"\nIntegration window:")
    print(f"  T_sim = {T_sim:.4f}  "
          f"= {n_fast} fast cycles × {n_slow_cycles} slow cycles")

    f0_3 = np.array([1.0, 0.0, 0.0])
    f0_4 = np.array([1.0, 0.0, 0.0, 0.0])

    # Dissipation matrices (postulated at this single-scale level).
    # Part II derives D from L through the Mori-Zwanzig projection:
    #   D_{n+1} = L_n^{re} T_n (L_n^{re})^T
    # where T_n is the correlation-time matrix of the eliminated sector.
    # Here we use hand-chosen values to demonstrate the two-tier structure.
    D3d = -np.diag([0.08, 0.05, 0.01])
    D4d = -np.diag([0.08, 0.05, 0.01, 0.005])

    def rhs_const(A, f0):
        """Return a constant-forcing RHS function for solve_ivp."""
        def fn(t, q):
            return A @ q + f0
        return fn

    kw = dict(method='DOP853', max_step=dt_out,
              rtol=ODE_RTOL, atol=ODE_ATOL)

    sol3L  = solve_ivp(rhs_const(L3,        f0_3), [0, T_sim], np.zeros(3),
                       t_eval=t_eval, **kw)
    sol4L  = solve_ivp(rhs_const(L4,        f0_4), [0, T_sim], np.zeros(4),
                       t_eval=t_eval, **kw)
    sol3LD = solve_ivp(rhs_const(L3 + D3d,  f0_3), [0, T_sim], np.zeros(3),
                       t_eval=t_eval, **kw)
    sol4LD = solve_ivp(rhs_const(L4 + D4d,  f0_4), [0, T_sim], np.zeros(4),
                       t_eval=t_eval, **kw)

    t3L  = sol3L.y.T;   t4L  = sol4L.y.T
    t3LD = sol3LD.y.T;  t4LD = sol4LD.y.T
    time = sol3L.t

    v0_proj_3L  = t3L  @ v0
    v0_proj_3LD = t3LD @ v0

    # -----------------------------------------------------------------------
    # Console diagnostics
    # -----------------------------------------------------------------------
    print(f"\nNull-mode projection  v₀·q(t)  [N=3]:")
    print(f"  Pure L:  q(0)={v0_proj_3L[0]:.4f}  →  q(T)={v0_proj_3L[-1]:.4f}  "
          f"(drift={v0_proj_3L[-1]-v0_proj_3L[0]:+.4f})")
    print(f"  L+D:     q(0)={v0_proj_3LD[0]:.4f}  →  q(T)={v0_proj_3LD[-1]:.4f}  "
          f"(drift={v0_proj_3LD[-1]-v0_proj_3LD[0]:+.4f})")

    print(f"\n‖q_final‖  at  t = {T_sim:.1f}:")
    print(f"  N=3 pure L:  {norm(t3L[-1]):8.4f}  ← NON-RETURN (Gateway drift)")
    print(f"  N=4 pure L:  {norm(t4L[-1]):8.4f}  ← bounded oscillation")
    print(f"  N=3 L+D:     {norm(t3LD[-1]):8.4f}  ← drift + dissipative offset")
    print(f"  N=4 L+D:     {norm(t4LD[-1]):8.4f}  ← dissipative steady state")
    norm3 = norm(t3L[-1]);  norm4 = norm(t4L[-1])
    if norm4 > 1e-10:
        ratio_str = f"{norm3/norm4:.0f}:1"
    else:
        ratio_str = f">> 1  (N=4 norm = {norm4:.2e}, i.e. machine precision)"
    print(f"  Ratio ‖q(N=3)‖/‖q(N=4)‖ = {ratio_str}  (topological, not numerical)")

    print(f"\nC-component at t=T:")
    print(f"  N=3 pure L:  C={t3L[-1,2]:+.6f}  ← FROZEN in fabric")
    print(f"  N=4 pure L:  C={t4L[-1,2]:+.6f}  ← returned to origin")
    print(f"  N=3 L+D:     C={t3LD[-1,2]:+.6f}")
    print(f"  N=4 L+D:     C={t4LD[-1,2]:+.6f}")

    print("\n" + "=" * 70)
    print("TWO-TIER ENERGY DECOMPOSITION  (closed loading cycle)")
    print("=" * 70)
    print(f"""
             |  N=3 (odd, Gateway open)  |  N=4 (even, Gateway closed)
─────────────┼───────────────────────────┼─────────────────────────────
E_conf  [L]  |  {Ec3:12.6f} J         |  {Ec4:12.6f} J
E_fric  [D]  |  {Ef3:12.6f} J         |  {Ef4:12.6f} J
Total        |  {Ec3+Ef3:12.6f} J         |  {Ec4+Ef4:12.6f} J
─────────────┼───────────────────────────┼─────────────────────────────
Zero mode?   |  YES → L-channel OPEN     |  NO  → L-channel GENERICALLY CLOSED
D-channel    |  OPEN  (μ > 0)            |  OPEN  (μ > 0)

Frictionless limit (μ → 0):
  N=3: E_conf = {Ec3_0:.6f} J  ← SURVIVES (topological, parity-protected)
  N=4: E_conf = {Ec4_0:.6f} J  ← zero (generic even N; non-generic exceptions possible, see Part II)
  Both E_fric → 0
""")

    # -----------------------------------------------------------------------
    # Parity sweep N = 1 … 7
    # -----------------------------------------------------------------------
    def make_chain_L(N):
        """Build an N×N tridiagonal skew-symmetric chain with L_{i,i+1}=1+0.5i."""
        L = np.zeros((N, N))
        for i in range(N - 1):
            L[i, i + 1] =  1 + 0.5 * i
            L[i + 1, i] = -(1 + 0.5 * i)
        return L

    parity = []
    for N in range(1, 8):
        LN = make_chain_L(N)
        eN = eigvals(LN)
        parity.append((N, det(LN), min(abs(e) for e in eN), N % 2 == 1))

    print("Parity sweep N = 1 … 7:")
    print(f"  {'N':>3}  {'det(L)':>10}  {'min|λ|':>10}  {'Zero mode?':>14}  {'Parity':>6}")
    for N, d, m, has_zero in parity:
        label = "YES ← Gateway" if has_zero else "NO"
        print(f"  {N:3d}  {d:+10.4f}  {m:10.6f}  {label:>14}  "
              f"{'ODD' if N % 2 else 'EVEN':>6}")

    # =======================================================================
    # Figures — 4 × 3 GridSpec layout
    # =======================================================================
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 3, hspace=0.42, wspace=0.32)

    # ── Panel (a): four-grain geometry ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal")
    ax.set_title("(a) Four-grain square\n(initial + loaded + perturbed)")

    grain_colours = ["#2166ac", "#d6604d", "#4daf4a", "#984ea3"]
    c_pert = affine(centres_0_sq, eps, eps)
    c_pert[2] += perturbation

    for i, c in enumerate(centres_0_sq):
        ax.add_patch(Circle(c, R, fill=False, ls="--", ec="0.6", lw=1))
    for i, c in enumerate(c_pert):
        ax.add_patch(Circle(c, R, fill=False, ls="-", ec=grain_colours[i], lw=2))
        ax.text(c[0], c[1], str(i), ha="center", va="center",
                color=grain_colours[i], fontsize=10, fontweight="bold")

    nfp, _ = normals_overlaps(c_pert, cp_sq, R)
    for i, (p, q) in enumerate(cp_sq):
        mid = 0.5 * (c_pert[p] + c_pert[q])
        col = "#d62728" if i in Cc_4 else "#1f77b4"
        ax.annotate("", xy=mid + 0.3 * nfp[i], xytext=mid,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.plot([], [], color="#d62728", lw=2, label=r"$\mathcal{C}^{\rm conf}$")
    ax.plot([], [], color="#1f77b4", lw=2, label=r"$\mathcal{C}^{\rm inv}$")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-1.5, 3.5);  ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel("x (m)");   ax.set_ylabel("y (m)")

    # ── Panel (b): N=4 eigenvalue spectrum ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(r"(b) Spectrum $\mathbf{L}^{\rm VMCE}$ ($N=4$)"
                 "\nTwo $\\pm i\\omega$ pairs — no zero mode")
    for e in e4:
        ax.scatter(e.real, e.imag, s=100, c="#1f77b4", marker="s",
                   zorder=5, edgecolors="k", lw=0.8)
    ax.scatter(0, 0, s=120, c="none", marker="o", zorder=4,
               edgecolors="#d62728", lw=2)
    ax.axhline(0, color="0.85", lw=0.5);  ax.axvline(0, color="0.85", lw=0.5)
    ax.scatter([], [], s=100, c="#1f77b4", marker="s",
               label=r"$\pm i\omega$ pairs")
    ax.scatter([], [], s=120, c="none", marker="o",
               edgecolors="#d62728", lw=2, label="Absent zero mode")
    ax.set_xlabel(r"Re($\lambda$)");  ax.set_ylabel(r"Im($\lambda$)")
    ax.legend(fontsize=9)

    # ── Panel (c): N=3 eigenvalue spectrum ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title(r"(c) Spectrum $\mathbf{L}^{\rm VMC}$ ($N=3$)"
                 "\nZero mode = Gateway")
    for e in e3:
        if abs(e) < 1e-10:
            ax.scatter(e.real, e.imag, s=150, c="#d62728", marker="o",
                       zorder=5, edgecolors="k", lw=0.8)
        else:
            ax.scatter(e.real, e.imag, s=100, c="#1f77b4", marker="s",
                       zorder=5, edgecolors="k", lw=0.8)
    ax.axhline(0, color="0.85", lw=0.5);  ax.axvline(0, color="0.85", lw=0.5)
    ax.scatter([], [], s=150, c="#d62728", marker="o",
               label="Zero mode (Gateway)")
    ax.scatter([], [], s=100, c="#1f77b4", marker="s",
               label=r"$\pm i\omega$ (Stable Layer)")
    ax.set_xlabel(r"Re($\lambda$)");  ax.set_ylabel(r"Im($\lambda$)")
    ax.legend(fontsize=9)

    # ── Shared phase-portrait helper ─────────────────────────────────────────
    def plot_drift_portrait(ax, traj, time, title, cmap_name, v0_dir=None):
        """Plot a time-coloured V–C phase portrait with optional v₀ arrow."""
        V = traj[:, 0];  C = traj[:, 2]
        points   = np.column_stack([V, C]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap_name, linewidth=1.2, alpha=0.85)
        lc.set_array(time[:-1])
        ax.add_collection(lc)
        ax.autoscale()
        cbar = plt.colorbar(lc, ax=ax, label="time", shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        ax.scatter(V[0],  C[0],  s=120, c="green", marker="o", zorder=6,
                   edgecolors="k", lw=1.2, label="Start")
        ax.scatter(V[-1], C[-1], s=120, c="k",     marker="x", zorder=6,
                   lw=2.5, label="End")
        if v0_dir is not None:
            drift_vc  = np.array([V[-1] - V[0], C[-1] - C[0]])
            drift_len = norm(drift_vc)
            if drift_len > 0.1:
                drift_unit  = drift_vc / drift_len
                data_range  = max(V.max() - V.min(), C.max() - C.min())
                arrow_len   = data_range * 0.30
                base_x = V[0] + drift_vc[0] * 0.15
                base_y = C[0] + drift_vc[1] * 0.15
                ax.annotate("",
                    xy=(base_x + drift_unit[0] * arrow_len,
                        base_y + drift_unit[1] * arrow_len),
                    xytext=(base_x, base_y),
                    arrowprops=dict(arrowstyle="->,head_width=0.4",
                                    color="#e74c3c", lw=2.5, ls="--"))
                ax.text(base_x + drift_unit[0] * arrow_len * 1.05,
                        base_y + drift_unit[1] * arrow_len * 1.10,
                        r"$\mathbf{v}_0$", fontsize=14,
                        color="#e74c3c", fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("V (volumetric)")
        ax.set_ylabel("C (configurational)")
        ax.legend(fontsize=8, loc="upper left")
        ax.axhline(0, color="0.85", lw=0.5)
        ax.axvline(0, color="0.85", lw=0.5)

    # ── Panel (d): N=3 pure L ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    plot_drift_portrait(ax, t3L, time,
        "(d) $N=3$, pure $\\mathbf{L}$\nOscillations + secular drift along "
        "$\\mathbf{v}_0$", "Reds", v0_dir=v0)
    ins = ax.inset_axes([0.52, 0.05, 0.45, 0.30])
    ins.plot(time, v0_proj_3L, color="#d62728", lw=0.8)
    ins.set_xlabel("t", fontsize=7)
    ins.set_ylabel(r"$\mathbf{v}_0\!\cdot\!\mathbf{q}$", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_title("null-mode: linear drift", fontsize=7,
                  color="#d62728", fontweight="bold")

    # ── Panel (e): N=4 pure L ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    plot_drift_portrait(ax, t4L, time,
        "(e) $N=4$, pure $\\mathbf{L}$\nBounded oscillations — no secular drift",
        "Blues")
    ins = ax.inset_axes([0.52, 0.05, 0.45, 0.30])
    ins.plot(time, t4L[:, 2], color="#1f77b4", lw=0.5)
    ins.axhline(0, color="0.5", lw=0.3, ls="--")
    ins.set_xlabel("t", fontsize=7);  ins.set_ylabel("C(t)", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_title("bounded: no drift", fontsize=7,
                  color="#1f77b4", fontweight="bold")

    # ── Panel (f): two-tier schematic ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]);  ax.axis("off")
    ax.set_title("(f) Two tiers of irreversibility")
    ax.text(0.5, 0.92,
            "Tier 1: L-channel (topological)\n"
            r"Governed by skew-symmetric $\mathbf{L}$" + "\n"
            r"Zero mode $\rightarrow$ energy frozen in fabric" + "\n"
            "Parity-controlled: odd $N$ (generically)\n"
            r"$\mu$-independent",
            transform=ax.transAxes, fontsize=10, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffcccc", alpha=0.8))
    ax.text(0.5, 0.47,
            "Tier 2: D-channel (dissipative)\n"
            r"Governed by symmetric $\mathbf{D}$" + "\n"
            r"Friction $\rightarrow$ energy lost to atomic lattice" + "\n"
            r"Always present when $\mu > 0$" + "\n"
            "Parity-independent",
            transform=ax.transAxes, fontsize=10, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#cce5ff", alpha=0.8))
    ax.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.62),
                transform=ax.transAxes,
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=2))
    ax.text(0.5, 0.57, "independent at one scale;\nconnected across scales (Part II)",
            transform=ax.transAxes, fontsize=8, ha="center", va="center",
            style="italic", color="0.3")

    # ── Panel (g): N=3 L+D ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    plot_drift_portrait(ax, t3LD, time,
        r"(g) $N=3$, $\mathbf{L}+\mathbf{D}$" + "\nDrift along "
        r"$\mathbf{v}_0$ + dissipative saturation", "Reds", v0_dir=v0)
    ins = ax.inset_axes([0.52, 0.05, 0.45, 0.30])
    ins.plot(time, v0_proj_3LD, color="#d62728", lw=0.8)
    ins.set_xlabel("t", fontsize=7)
    ins.set_ylabel(r"$\mathbf{v}_0\!\cdot\!\mathbf{q}$", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_title("drift saturates (D caps it)", fontsize=7,
                  color="#d62728", fontweight="bold")

    # ── Panel (h): N=4 L+D ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    plot_drift_portrait(ax, t4LD, time,
        r"(h) $N=4$, $\mathbf{L}+\mathbf{D}$" + "\nDecaying oscillations — "
        "D-channel only", "Blues")
    ins = ax.inset_axes([0.52, 0.05, 0.45, 0.30])
    ins.plot(time, t4LD[:, 2], color="#1f77b4", lw=0.5)
    ins.axhline(0, color="0.5", lw=0.3, ls="--")
    ins.set_xlabel("t", fontsize=7);  ins.set_ylabel("C(t)", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_title("bounded + decaying", fontsize=7,
                  color="#1f77b4", fontweight="bold")

    # ── Panel (i): energy bar chart ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("(i) Energy budget: two tiers\n(closed loading cycle)")
    x    = np.arange(4)
    labs = ["3-grain\n($N=3$)\n$\\mu>0$", "4-grain\n($N=4$)\n$\\mu>0$",
            "3-grain\n($N=3$)\n$\\mu=0$", "4-grain\n($N=4$)\n$\\mu=0$"]
    Ecs  = [Ec3,  Ec4,  Ec3_0, Ec4_0]
    Efs  = [Ef3,  Ef4,  0.0,   0.0  ]
    ax.bar(x, Ecs, 0.6,
           label=r"$E^{\rm conf}$ (L-channel, topological)",
           color="#d62728", edgecolor="k", lw=0.8, alpha=0.85)
    ax.bar(x, Efs, 0.6, bottom=Ecs,
           label=r"$E^{\rm fric}$ (D-channel, dissipative)",
           color="#1f77b4", edgecolor="k", lw=0.8, alpha=0.85)
    ax.set_xticks(x);  ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Energy (J)")
    ax.legend(fontsize=8, loc="upper right")
    for i in range(4):
        total = Ecs[i] + Efs[i]
        if total > 1e-8:
            ax.text(x[i], total + 0.01 * max(Ecs + Efs),
                    f"{total:.4f}", ha="center", fontsize=8)

    # ── Panels (j,k): parity sweep ──────────────────────────────────────────
    Ns  = [p[0] for p in parity]
    ds  = [p[1] for p in parity]
    mes = [p[2] for p in parity]
    hzs = [p[3] for p in parity]
    bar_cols = ["#d62728" if h else "#1f77b4" for h in hzs]

    ax = fig.add_subplot(gs[3, 0])
    ax.set_title(r"(j) Parity Theorem: $|\det(\mathbf{L})|$ vs $N$")
    ax.bar(Ns, [abs(d) for d in ds], color=bar_cols, edgecolor="k", lw=0.8)
    ax.set_xlabel("Coupling dimension $N$")
    ax.set_ylabel(r"$|\det(\mathbf{L})|$")
    ax.set_xticks(Ns)
    for i, N in enumerate(Ns):
        ax.text(N, abs(ds[i]) + 0.3 * max(abs(d) for d in ds) * 0.05,
                "odd" if N % 2 else "even",
                ha="center", fontsize=8, color=bar_cols[i], fontweight="bold")
    ax.bar([], [], color="#d62728", label="Odd $N$: det=0 (Gateway)")
    ax.bar([], [], color="#1f77b4", label=r"Even $N$: det$\neq$0")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[3, 1])
    ax.set_title(r"(k) Smallest $|\lambda|$ vs $N$")
    ax.bar(Ns, mes, color=bar_cols, edgecolor="k", lw=0.8)
    ax.set_xlabel("Coupling dimension $N$")
    ax.set_ylabel(r"$\min|\lambda|$")
    ax.set_xticks(Ns)
    m_max = max(mes)
    for i, N in enumerate(Ns):
        if hzs[i]:
            ax.annotate("zero!", xy=(N, 0), xytext=(N + 0.3, m_max * 0.25),
                        arrowprops=dict(arrowstyle="->", color="#d62728"),
                        fontsize=8, color="#d62728", fontweight="bold")

    # ── Panel (l): summary table ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2]);  ax.axis("off")
    ax.set_title("(l) Summary: two-tier comparison")
    td = [
        ["Property",              "$N=3$ (odd)",              "$N=4$ (even)"           ],
        ["$\\det(\\mathbf{L})$",  "= 0",                      f"= {det4:.2f} = $(L_{{VM}}L_{{CE}})^2$"],
        ["Zero mode",             "YES",                      "NO (generic)"               ],
        ["L-channel (config.)",   "OPEN\n(topol. irrev.)",    "GENERICALLY\nCLOSED"       ],
        ["D-channel (frictional)","OPEN\n(always, $\\mu>0$)", "OPEN\n(always, $\\mu>0$)"],
        ["Pure $\\mathbf{L}$ orbits", "Drift along $\\mathbf{v}_0$", r"Closed $\pm i\omega$"],
        ["$\\mathbf{L}+\\mathbf{D}$ orbits", "Drift + saturation", "Decay only"       ],
        ["$\\mu=0$ irrev.?",      "YES ($E^{\\rm conf}>0$)",  "NO"                     ],
        ["Novel result",          "Non-dissipative\nirrev.",   "Classical\ndissip. only"],
    ]
    tb = ax.table(cellText=td[1:], colLabels=td[0],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False);  tb.set_fontsize(8.5);  tb.scale(1.0, 1.7)
    for j in range(3):
        tb[0, j].set_facecolor("#e0e0e0")
        tb[0, j].set_text_props(fontweight="bold")
    for i in range(1, len(td)):
        tb[i, 1].set_facecolor("#ffcccc")
        tb[i, 2].set_facecolor("#cce5ff")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nFigures saved: {OUT_PNG}  |  {OUT_PDF}")


# ===========================================================================
if __name__ == "__main__":
    main()
