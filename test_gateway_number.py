"""Tests for gateway_number. Run with `python -m pytest` or `python -m unittest`."""
import json
import math
import io
import contextlib
import unittest

import numpy as np

import gateway_number as gn


def _random_operators(rng, n):
    M = rng.normal(size=(n, n))
    D = M @ M.T + 0.1 * np.eye(n)
    K = rng.normal(size=(n, n))
    L = K - K.T
    N = rng.normal(size=(n, n))
    g = N @ N.T + 0.1 * np.eye(n)
    return D, L, g


class TestS6Regression(unittest.TestCase):
    """Numbers quoted in Supplemental Material Sec. S6."""

    def setUp(self):
        self.s6 = gn.verify_s6()

    def test_values(self):
        self.assertAlmostEqual(self.s6["G"], 0.19826, places=5)
        self.assertAlmostEqual(self.s6["kappa"], 0.25000, places=5)
        self.assertAlmostEqual(self.s6["frobenius"], 0.19826, places=5)
        self.assertAlmostEqual(self.s6["frobenius_transformed"], 0.07582, places=5)
        self.assertAlmostEqual(self.s6["G_transformed"], self.s6["G"], places=12)
        self.assertAlmostEqual(self.s6["G_closed_form"], self.s6["G"], places=12)

    def test_scaling_table(self):
        expected = {0.00: 0.0, 0.02: 0.01586, 0.05: 0.03965, 0.10: 0.07931, 0.20: 0.15861, 0.25: 0.19826}
        for gm, G in self.s6["scaling"]:
            self.assertAlmostEqual(G, expected[gm], places=5)
            self.assertGreaterEqual(math.copysign(1.0, G), 0.0)   # no negative zero

    def test_linear_in_gamma(self):
        g0 = dict(self.s6["scaling"])[0.25] / 0.25
        for gm, G in self.s6["scaling"]:
            self.assertAlmostEqual(G, gm * g0, places=12)


class TestInvariants(unittest.TestCase):
    def test_basis_invariance_any_dimension(self):
        rng = np.random.default_rng(11)
        for n in range(2, 7):
            for _ in range(50):
                D, L, g = _random_operators(rng, n)
                P = rng.normal(size=(n, n))
                if abs(np.linalg.det(P)) < 1e-2:
                    continue
                G0 = gn.gateway_number(D, L, g)
                G1 = gn.gateway_number(*gn.transform(D, L, g, P))
                self.assertAlmostEqual(G0, G1, delta=1e-9 * max(1.0, G0))

    def test_closed_form_matches_trace_invariant(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            a = rng.uniform(0.1, 2.0, size=4)
            b = rng.normal(size=2)
            D = gn.assemble_D(a[0], a[1], a[2], a[3] * 0.3)
            L = gn.assemble_L(*b)
            self.assertAlmostEqual(gn.gateway_number(D, L),
                                   gn.closed_form_G(a[0], a[1], a[2], a[3] * 0.3, *b), places=12)

    def test_pure_dissipation_is_exact_zero(self):
        G = gn.gateway_number(np.eye(3), np.zeros((3, 3)))
        self.assertEqual(G, 0.0)
        self.assertEqual(math.copysign(1.0, G), 1.0)

    def test_lg_spectrum_is_imaginary(self):
        rng = np.random.default_rng(5)
        D, L, g = _random_operators(rng, 4)
        self.assertLess(np.max(np.abs(np.linalg.eigvals(L @ g).real)), 1e-10)


class TestTwoCoordinateTheory(unittest.TestCase):
    def test_gcrit_forms(self):
        for mu1, mu2 in [(2.0, 1.0), (1.0, 1.0), (3.0, 0.0), (0.5, 0.2)]:
            d = (mu1 - mu2) / (mu1 + mu2)
            self.assertAlmostEqual(gn.two_coordinate_Gcrit(mu1, mu2), abs(d) / math.sqrt(1 + d * d), places=14)
        self.assertAlmostEqual(gn.two_coordinate_Gcrit(1.0, 0.0), 1 / math.sqrt(2), places=14)

    def test_spectrum_and_threshold(self):
        mu1, mu2 = 2.0, 1.0
        kc = abs(mu1 - mu2) / 2
        for kappa, complex_expected in [(0.9 * kc, False), (1.1 * kc, True)]:
            lam = gn.two_coordinate_spectrum(mu1, mu2, kappa)
            A = np.diag([mu1, mu2]) + kappa * np.array([[0.0, -1.0], [1.0, 0.0]])
            np.testing.assert_allclose(np.sort_complex(np.array(lam)),
                                       np.sort_complex(np.linalg.eigvals(A)), atol=1e-12)
            G = gn.gateway_number(np.diag([mu1, mu2]), kappa * np.array([[0.0, -1.0], [1.0, 0.0]]))
            self.assertEqual(G > gn.two_coordinate_Gcrit(mu1, mu2), complex_expected)

    def test_response_slope_two_and_three_coordinates(self):
        rng = np.random.default_rng(9)
        for n in (2, 3):
            D, L, g = _random_operators(rng, n)
            At = gn.entropy_normed(D + L, g)
            slope = np.linalg.inv(At)
            G_true = gn.gateway_number(D, L, g)
            self.assertAlmostEqual(gn.gateway_from_response_slope(slope), G_true, places=10)
            naive = gn._G_of_matrix(slope)
            if n == 2:
                self.assertAlmostEqual(naive, G_true, places=10)
            else:
                self.assertGreater(abs(naive - G_true), 1e-6)

    def test_susceptibility_inverse(self):
        rng = np.random.default_rng(13)
        D, L, g = _random_operators(rng, 2)
        A, w = D + L, 0.7
        np.testing.assert_allclose(np.linalg.inv(gn.susceptibility(A, g, w)),
                                   g - 1j * w * np.linalg.inv(A), atol=1e-12)

    def test_triad_gcrit_limits(self):
        self.assertEqual(gn.triad_Gcrit(1.0), 0.0)
        self.assertAlmostEqual(gn.triad_Gcrit(0.0), 0.5, places=14)
        self.assertAlmostEqual(gn.triad_Gcrit(1e8), 1 / math.sqrt(2), places=7)

    def test_pairing(self):
        a, s = 0.37, -1.2
        Ap = np.array([[1.0, s + a], [s - a, 2.0]])
        Am = np.array([[1.0, -s + a], [-s - a, 2.0]])
        L_ij, eps = gn.pairing_estimate(Ap, Am, 0, 1)
        self.assertAlmostEqual(L_ij, a, places=15)
        self.assertAlmostEqual(eps, 0.0, places=15)


class TestInputValidation(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        D, L = np.eye(3), gn.assemble_L(0.2, 0.1)
        with self.assertRaises(ValueError):
            gn.gateway_number(D + np.triu(np.ones((3, 3)), 1), L)          # D not symmetric
        with self.assertRaises(ValueError):
            gn.gateway_number(D, L + np.eye(3))                            # L not antisymmetric
        with self.assertRaises(ValueError):
            gn.gateway_number(D, L, -np.eye(3))                            # g not positive definite
        with self.assertRaises(ValueError):
            gn.gateway_number(D, np.zeros((2, 2)))                         # shape mismatch
        with self.assertRaises(ValueError):
            gn.gateway_number(np.zeros((3, 3)), L)                         # no dissipation
        with self.assertRaises(ValueError):
            gn.gateway_number(D, np.full((3, 3), np.nan))                  # non-finite
        with self.assertRaises(ValueError):
            gn.triad_Gcrit(-1.0)


class TestCommandLine(unittest.TestCase):
    def test_report_exit_status(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(gn.main([]), 0)
        self.assertIn("Gateway number  G = 0.19826", buf.getvalue())

    def test_json(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(gn.main(["--json"]), 0)
        data = json.loads(buf.getvalue())
        self.assertTrue(data["all_checks_pass"])
        self.assertAlmostEqual(data["s6"]["G"], 0.19826, places=5)


if __name__ == "__main__":
    unittest.main()
