# Changelog

## 1.1.0

* Added the S1X verification block for the two-coordinate theory of Supplemental Material Sec. S1: relaxation spectrum, two-coordinate critical Gateway number, response-slope extraction, reversed-magnetisation pairing, and the symmetric-triad threshold.
* Restructured the script as an importable module with a command-line report; the report exits with status 1 when a check fails, and `--json` gives machine-readable results.
* Added input validation: shape, finiteness, symmetry of D, antisymmetry of L, positive-definite g and non-vanishing dissipation.
* `gateway_number` and `circulation_rate` clip round-off in the trace, so G = 0 is returned as +0.0.
* Added `gateway_from_response_slope`, valid in any dimension.
* Relabelled S6C as a dimensional scaling illustration and set the 3d Fe marker to zeta_SOC/Delta_cf ~ 1e-1, matching the Supplemental Material.
* Added the unit-test suite, packaging metadata, continuous integration and the reference report `expected_output.txt`.
* All numbers of Supplemental Material Sec. S6 are unchanged.

## 1.0.0

* Initial archived version: S6A mathematical verification, S6B illustrative implementation, S6C physical scaling.
