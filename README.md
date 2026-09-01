Two Tiers of Irreversibility in Minimal Granular Assemblies

Companion code for:

Nicot F. & Regenauer-Lieb K. (2026) "The Principle of Ordered Action: Contact Topology, Phase-Space Expansion, and the Arrow of Time in Granular Matter" Proc. R. Soc. A

What this demonstrates

Two distinct sources of irreversibility operate in loaded granular assemblies. At a single scale they act independently; across scales they are connected through the Mori–Zwanzig projection (see Part II).

	Tier 1: L-channel (configurational)	Tier 2: D-channel (frictional)
Governed by	Skew-symmetric Onsager block L	Symmetric relaxation block D
Mechanism	Energy frozen in contact fabric by irrecoverable normal rotations	Energy dissipated to the atomic lattice by grain sliding
Parity dependence	Odd N: Gateway open. Even N: generically closed, but non-generic configurations can harbour Gateway channels through odd-parity sub-structure (Part II)	Always present when μ > 0
Friction dependence	None (survives μ → 0)	Vanishes with μ → 0
Cross-scale connection	The L-coupling at a finer level manufactures D at the coarser level through the memory kernel (Part II, transfer relation)	Manufactured from L below; postulated at a single scale in this script

The Parity Theorem discriminates Tier 1 at a given level: a skew-symmetric matrix of odd dimension necessarily has a zero eigenvalue (the Gateway mode), while even dimension generically does not. The word "generically" matters: the grokking experiments of the companion series measure rank(L) = 2 at N = 6 (even), a non-generic configuration with a four-dimensional null space, demonstrating that even levels can carry Gateway channels.

Quick start
bash
python four_grain_poa.py

Outputs:

four_grain_two_tier.pdf — publication-quality 12-panel figure (Figure 6 of the paper)
four_grain_two_tier.png — rasterised copy at 150 dpi
Console diagnostics: eigenvalues, energy budgets, parity sweep table
Requirements
Package	Minimum version
Python	3.8
NumPy	1.20
SciPy	1.7
Matplotlib	3.4

Install with:

bash
pip install numpy scipy matplotlib
What the script computes

Contact mechanics on 3-grain (triangle, N = 3) and 4-grain (square, N = 4) assemblies: forward affine compression, symmetry-breaking perturbation, contact classification into invariant and configurational sets, exact inverse affine reversal, non-return residual measurement. The reverse step uses an exact inverse transform (dividing by 1 − ε) rather than the approximate inverse (multiplying by 1 + ε), eliminating the O(ε²) kinematic artefact and ensuring the measured residual is purely topological.

Two-channel energy budget over a closed loading cycle: configurational energy (L-channel, topological) and frictional energy (D-channel, dissipative), with and without friction. The frictional channel caps the tangential force at the Coulomb limit.

Coupling matrices for the VMC (N = 3) and VMCE (N = 4) thermodynamic coupling chains, with a frequency commensurability sweep that selects an integer fast-to-slow ratio for clean phase portraits.

Phase-space integration using DOP853 (8th-order Dormand–Prince) at tolerances 10⁻¹¹, for pure L and L + D dynamics over 12 slow cycles. The dissipation matrices D are postulated at this single-scale level; Part II derives them from L through the Mori–Zwanzig projection.

Parity sweep from N = 1 to 7, verifying det(L) = 0 at every odd N and det(L) ≠ 0 at every even N for the generic tridiagonal chain.

The 12-panel figure
Row	Left	Centre	Right
1	(a) Grain geometry	(b) N=4 eigenvalue spectrum	(c) N=3 spectrum with Gateway
2	(d) N=3 pure L: drift along v₀	(e) N=4 pure L: bounded	(f) Two-tier schematic
3	(g) N=3 L+D: drift + saturation	(h) N=4 L+D: decay	(i) Energy bar chart
4	(j) |det(L)| vs N	(k) min|λ| vs N	(l) Summary table
Relation to Part II

This script demonstrates the two-tier structure at a single scale. Part II of the PoA series extends the framework to multiple scales using the Mori–Zwanzig projection formalism. Three points of contact:

The dissipation matrices D used here are postulated. Part II shows they are manufactured from L at the finer level through the memory kernel: Dₙ₊₁ = Lₙʳᵉ Tₙ (Lₙʳᵉ)ᵀ, where Tₙ is the correlation-time matrix of the eliminated sector.
The even-N Gateway closure demonstrated here is the generic case. Part II introduces the random-forest construction on the Hasse lattice, showing how non-generic even levels can harbour active inverse-cascade channels through their odd-parity sub-structure.
The null-mode drift along v₀ measured in panel (d) is the single-scale manifestation of Gateway inheritance: Proposition 2 of Part II proves that the null direction of L at level n is annihilated by the induced D at level n+1, so it survives every subsequent coarse-graining.
Repository structure
poa-granular-irreversibility/
├── LICENSE                    BSD-3-Clause
├── README.md                  This file
├── four_grain_poa.py          Main script (single file, ~840 lines)
├── CITATION.cff               Machine-readable citation metadata
├── setup_github.sh            Git init / GitHub push helper
└── .gitignore
Citing this work

If you use this code, please cite:

bibtex
@article{Nicot_Regenauer-Lieb_2026,
  author  = {Nicot, Fran\c{c}ois and Regenauer-Lieb, Klaus},
  title   = {The Principle of Ordered Action: Contact Topology,
             Phase-Space Expansion, and the Arrow of Time
             in Granular Matter},
  journal = {Proc. R. Soc. A},
  year    = {2026},
  note    = {Submitted 3 July 2026}
}
License

BSD-3-Clause. See LICENSE.
