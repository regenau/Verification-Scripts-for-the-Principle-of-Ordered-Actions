# Two Tiers of Irreversibility in Minimal Granular Assemblies

Companion code for:

> Nicot F. & Regenauer-Lieb K. (2026)
> "The Principle of Ordered Action: Contact Topology, Phase-Space Expansion, and the Arrow of Time in Granular Matter"
> *Phil. Trans. R. Soc. A*

## What this demonstrates

Two distinct and independent mechanisms produce irreversibility in loaded granular assemblies:

| | Tier 1: L-channel (configurational) | Tier 2: D-channel (frictional) |
|---|---|---|
| **Governed by** | Skew-symmetric Onsager block **L** | Symmetric relaxation block **D** |
| **Mechanism** | Energy frozen in contact fabric by irrecoverable normal rotations | Energy dissipated to the atomic lattice by grain sliding |
| **Parity dependence** | Odd *N* only (Gateway open) | Always present when μ > 0 |
| **Friction dependence** | None (survives μ → 0) | Vanishes with μ → 0 |

The **Parity Theorem** discriminates Tier 1: a skew-symmetric matrix of odd dimension necessarily has a zero eigenvalue (the Gateway mode), while even dimension generically does not.

## Quick start

```bash
python four_grain_poa.py
```

**Outputs:**
- `four_grain_two_tier.pdf` — publication-quality 12-panel figure (Figure 6 of the paper)
- `four_grain_two_tier.png` — rasterised copy at 150 dpi
- Console diagnostics: eigenvalues, energy budgets, parity sweep table

## Requirements

| Package | Minimum version |
|---|---|
| Python | 3.8 |
| NumPy | 1.20 |
| SciPy | 1.7 |
| Matplotlib | 3.4 |

Install with:
```bash
pip install numpy scipy matplotlib
```

## What the script computes

1. **Contact mechanics** on 3-grain (triangle, *N* = 3) and 4-grain (square, *N* = 4) assemblies: forward affine compression, symmetry-breaking perturbation, contact classification into invariant and configurational sets, exact inverse affine reversal, non-return residual measurement.

2. **Two-channel energy budget** over a closed loading cycle: configurational energy (L-channel, topological) and frictional energy (D-channel, dissipative), with and without friction.

3. **Coupling matrices** for the VMC (*N* = 3) and VMCE (*N* = 4) thermodynamic coupling chains, with frequency commensurability sweep.

4. **Phase-space integration** using DOP853 (8th-order Dormand–Prince) at tolerances 10⁻¹¹, for pure **L** and **L** + **D** dynamics over 12 slow cycles.

5. **Parity sweep** from *N* = 1 to 7, verifying det(**L**) = 0 at every odd *N* and det(**L**) ≠ 0 at every even *N*.

## Repository structure

```
poa-granular-irreversibility/
├── LICENSE
├── README.md
├── four_grain_poa.py          # Main script (single file, no dependencies beyond NumPy/SciPy/Matplotlib)
├── CITATION.cff               # Machine-readable citation metadata
└── .gitignore
```

## Citing this work

If you use this code, please cite:

```bibtex
@article{Nicot_Regenauer-Lieb_2026,
  author  = {Nicot, Fran\c{c}ois and Regenauer-Lieb, Klaus},
  title   = {The Principle of Ordered Action: Contact Topology,
             Phase-Space Expansion, and the Arrow of Time in Granular Matter},
  journal = {Phil. Trans. R. Soc. A},
  year    = {2026},
  note    = {Submitted}
}
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
