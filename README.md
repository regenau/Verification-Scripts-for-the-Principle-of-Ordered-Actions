# gateway-number

[![tests](https://github.com/OWNER/gateway-number/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/gateway-number/actions/workflows/tests.yml)

Reference implementation of the catalysis Gateway number $G$ and its verification report, companion code for

> *Quantum spin regulates the catalytic cycle beyond binding energy* (submitted to *The Journal of Chemical Physics*).

Archive: https://doi.org/10.5281/zenodo.21440491

The Gateway number is the basis-invariant weight of non-dissipative circulation $\mathcal{L}$ (antisymmetric) relative to dissipation $\mathcal{D}$ (symmetric), formed from the relaxation operator $(\mathcal{D}+\mathcal{L})g$ with $g$ the positive-definite local metric:

$$G=\sqrt{-\operatorname{Tr}\big((\mathcal{L}g)^2\big)\,/\,\operatorname{Tr}\big((\mathcal{D}g)^2\big)}.$$

## What the code reproduces

Running the module prints a deterministic report that reproduces every number quoted in the Supplemental Material:

| Report block | Supplemental Material | Content |
|---|---|---|
| S1X | Sec. S1 | two-coordinate spectrum, two-coordinate critical Gateway number, response-slope extraction, reversed-magnetisation pairing, symmetric-triad threshold |
| S6A | Sec. S6 | closed form against the trace invariant, spectrum of $\mathcal{L}g$, basis invariance |
| S6B | Sec. S6 | the illustrative normalised operator and its $G$ |
| S6C | Sec. S6 | dimensional scaling illustration $\gamma\sim\zeta_{\mathrm{SOC}}/\Delta_{\mathrm{cf}}$ |

`expected_output.txt` is the reference report. All checks use closed-form linear algebra; no finite differences. Random checks use fixed seeds.

## Requirements and installation

Python 3.9 or later and NumPy 1.17 or later. Continuous integration runs the test suite on Python 3.9 to 3.13.

The module is a single file and runs without installation. To install from a clone, with the command-line entry point:

```
git clone https://github.com/OWNER/gateway-number.git
cd gateway-number
pip install .
```

## Usage

```
python gateway_number.py           # verification report; exit status 0 when every check passes
python gateway_number.py --json    # machine-readable results
gateway-number                     # same, after installation
```

As a library:

```python
import numpy as np
import gateway_number as gn

D = gn.assemble_D(D_CC=1.0, D_EE=1.0, D_SS=1.0, D_CE=0.3)   # channel order C, E, S
L = gn.assemble_L(L_SC=0.2, L_SE=0.15)
G = gn.gateway_number(D, L, g=np.eye(3))                      # 0.19826...
```

Main functions: `gateway_number`, `circulation_rate`, `closed_form_G`, `relaxation_eigenvalues`, `transform`, `entropy_normed`, `susceptibility`, `gateway_from_response_slope`, `two_coordinate_spectrum`, `two_coordinate_Gcrit`, `triad_Gcrit` and `pairing_estimate`. `gateway_number` and `circulation_rate` accept operators of any dimension, and every function checks its inputs: square, finite, matching shapes, symmetric $\mathcal{D}$, antisymmetric $\mathcal{L}$, positive-definite $g$ and non-vanishing dissipation. A violation raises `ValueError`.

## Conventions

* Triad channel order: C (chemical), E (electric), S (spin). `assemble_L(L_SC, L_SE)` places $L_{SC}$, $L_{SE}$ in the spin row and their negatives in the spin column.
* Coordinate change $x\mapsto Px$: $\mathcal{D}\mapsto P\mathcal{D}P^{\top}$, $\mathcal{L}\mapsto P\mathcal{L}P^{\top}$, $g\mapsto P^{-\top}gP^{-1}$.
* Response: perturbations vary as $e^{-i\omega t}$ and $\delta x=\chi h$, so $\chi^{-1}=g-i\omega\mathcal{A}^{-1}$. `gateway_from_response_slope` inverts the normalised slope before forming $G$, which is valid in any dimension; for two coordinates the slope itself gives $G$.
* `pairing_estimate(A_plus, A_minus, i, j)` returns the antisymmetric coefficient, even in the reference magnetisation, and the paired symmetric average, which vanishes for an exact calculation and measures the numerical error.

## Tests

```
python -m pytest            # or: python -m unittest
```

The suite pins the Sec. S6 numbers, checks basis invariance in dimensions 2 to 6, the two-coordinate theory, the triad threshold limits, the pairing, input validation and the command line.

## Scope

The S6 operator entries are illustrative normalised values, and the Fe line of S6C is a dimensional scaling illustration, not a computed operator and not a prediction. The value of $G$ for a real site requires the reactive prefactor, the dissipative coefficients and the entropy-metric normalisation, which the companion open-system electronic-response calculation (in preparation) is designed to supply.

## Citation

Cite the article above and this archive.
