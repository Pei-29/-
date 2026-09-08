"""Numerical and economic regression checks; run with unittest discovery."""

import unittest

import numpy as np

from src.portfolio_optimization import (
    analytical_gmv,
    example_data,
    generate_efficient_frontier,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    solve_gmv_cvxpy,
    solve_risk_aversion_portfolio,
    solve_target_return_portfolio,
    validate_inputs,
)


class PortfolioOptimizationTests(unittest.TestCase):
    """Check identities, constraints, failure handling and comparative statics."""

    def setUp(self) -> None:
        self.mu, self.Sigma, self.names = example_data()

    def assert_feasible(self, weights: np.ndarray, target: float | None = None) -> None:
        self.assertTrue(np.isclose(weights.sum(), 1, atol=1e-8, rtol=0))
        self.assertGreaterEqual(weights.min(), -1e-8)
        if target is not None:
            self.assertGreaterEqual(self.mu @ weights, target - 1e-8)

    def test_input_validation(self) -> None:
        validate_inputs(self.mu.tolist(), self.Sigma.tolist())
        self.assertTrue(np.allclose(self.Sigma, self.Sigma.T))
        self.assertGreater(np.linalg.eigvalsh(self.Sigma).min(), 0)
        for mu, covariance in [
            ([0.1], np.eye(2)),
            ([[0.1, 0.2]], np.eye(2)),
            ([0.1, np.nan], np.eye(2)),
            ([0.1, 0.2], [[1, 0.2], [0.1, 1]]),
            ([0.1, 0.2], [[1, 2], [2, 1]]),
            ([0.1, 0.2], [[1, 0], [0, np.inf]]),
            ([0.1, 0.2], [[1 + 1j, 0], [0, 1]]),
        ]:
            with self.subTest(mu=mu, covariance=covariance):
                with self.assertRaises(ValueError):
                    validate_inputs(mu, covariance)

    def test_metrics_include_cross_covariances(self) -> None:
        weights = np.array([0.2, 0.3, 0.5])
        direct = sum(weights[i] * weights[j] * self.Sigma[i, j] for i in range(3) for j in range(3))
        self.assertAlmostEqual(portfolio_return(weights, self.mu), 0.091)
        self.assertAlmostEqual(portfolio_variance(weights, self.Sigma), direct)
        self.assertAlmostEqual(portfolio_volatility(weights, self.Sigma) ** 2, direct)
        with self.assertRaises(ValueError):
            portfolio_return([1, 0], self.mu)

    def test_analytical_matches_numerical_gmv(self) -> None:
        exact = analytical_gmv(self.Sigma)
        numerical = solve_gmv_cvxpy(self.mu, self.Sigma)
        np.testing.assert_allclose(exact, numerical.weights, atol=1e-8, rtol=1e-7)
        self.assertTrue(np.isclose(exact.sum(), 1))
        # Equality-constrained stationarity: Sigma @ w is constant across assets.
        np.testing.assert_allclose(self.Sigma @ exact, np.repeat(exact @ self.Sigma @ exact, 3), atol=1e-12)

    def test_long_only_is_inactive_for_study_inputs(self) -> None:
        result = solve_gmv_cvxpy(self.mu, self.Sigma, long_only=True)
        self.assert_feasible(result.weights)
        self.assertGreater(result.weights.min(), 1e-3)
        np.testing.assert_allclose(result.weights, analytical_gmv(self.Sigma), atol=2e-7, rtol=0)

    def test_binding_short_sale_constraint_on_separate_fixture(self) -> None:
        # A separate test fixture, never substituted for the study's data.
        covariance = np.array([[0.04, 0.015], [0.015, 0.01]])
        mu = np.array([0.08, 0.05])
        exact = analytical_gmv(covariance)
        np.testing.assert_allclose(exact, [-0.25, 1.25], atol=1e-10)
        constrained = solve_gmv_cvxpy(mu, covariance, long_only=True)
        np.testing.assert_allclose(constrained.weights, [0, 1], atol=1e-7)
        self.assertGreater(constrained.variance, portfolio_variance(exact, covariance))

    def test_target_return_and_kkt_residuals(self) -> None:
        result = solve_target_return_portfolio(self.mu, self.Sigma, 0.08)
        self.assert_feasible(result.weights, 0.08)
        self.assertTrue(result.return_constraint_binding)
        self.assertGreater(result.return_dual, 0)
        self.assertGreaterEqual(result.nonneg_duals.min(), -1e-8)
        stationarity = 2 * self.Sigma @ result.weights + result.budget_dual - result.nonneg_duals - result.return_dual * self.mu
        np.testing.assert_allclose(stationarity, 0, atol=1e-8)
        np.testing.assert_allclose(result.nonneg_duals * result.weights, 0, atol=1e-8)
        self.assertLess(abs(result.return_dual * (0.08 - result.expected_return)), 1e-8)

    def test_low_target_returns_gmv(self) -> None:
        result = solve_target_return_portfolio(self.mu, self.Sigma, 0.05)
        self.assertFalse(result.return_constraint_binding)
        np.testing.assert_allclose(result.weights, analytical_gmv(self.Sigma), atol=2e-7, rtol=0)

    def test_hand_solved_two_asset_problems(self) -> None:
        mu, covariance = np.array([0.03, 0.11]), 0.01 * np.eye(2)
        target = solve_target_return_portfolio(mu, covariance, 0.09)
        np.testing.assert_allclose(target.weights, [0.25, 0.75], atol=1e-7)
        preference = solve_risk_aversion_portfolio(mu, covariance, 10)
        np.testing.assert_allclose(preference.weights, [0.3, 0.7], atol=1e-7)
        # This checks the convention lambda * variance, without a hidden 1/2.
        shorting = solve_risk_aversion_portfolio(mu, covariance, 1, long_only=False)
        np.testing.assert_allclose(shorting.weights, [-1.5, 2.5], atol=1e-7)

    def test_frontier_feasibility_endpoints_and_monotonicity(self) -> None:
        frontier = generate_efficient_frontier(self.mu, self.Sigma, asset_names=self.names)
        self.assertEqual(len(frontier), 80)
        self.assertTrue((frontier["status"] == "optimal").all())
        for _, row in frontier.iterrows():
            self.assert_feasible(row[["weight_Asset_A", "weight_Asset_B", "weight_Asset_C"]].to_numpy(dtype=float), row["target_return"])
        self.assertTrue((np.diff(frontier["variance"]) >= -1e-8).all())
        self.assertTrue((np.diff(frontier["expected_return"]) >= -1e-8).all())
        self.assertAlmostEqual(frontier.iloc[0]["expected_return"], self.mu @ analytical_gmv(self.Sigma), places=7)
        self.assertAlmostEqual(frontier.iloc[-1]["expected_return"], 0.12, places=7)

    def test_lambda_comparative_statics_and_frontier_consistency(self) -> None:
        results = [solve_risk_aversion_portfolio(self.mu, self.Sigma, value) for value in [0.1, 0.5, 1, 2, 5, 10, 20]]
        for result in results:
            self.assert_feasible(result.weights)
            # Independently solve a target-return problem at the selected return.
            target = min(result.expected_return, float(self.mu.max()))
            equivalent = solve_target_return_portfolio(self.mu, self.Sigma, target)
            self.assertAlmostEqual(result.variance, equivalent.variance, places=7)
        self.assertTrue((np.diff([r.variance for r in results]) <= 1e-8).all())
        self.assertTrue((np.diff([r.expected_return for r in results]) <= 1e-8).all())

    def test_invalid_targets_lambda_and_solver_status(self) -> None:
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(risk_aversion=value), self.assertRaises(ValueError):
                solve_risk_aversion_portfolio(self.mu, self.Sigma, value)
        with self.assertRaises(ValueError):
            solve_target_return_portfolio(self.mu, self.Sigma, 0.121)
        # Exercise an infeasible solver status, beyond the long-only precheck.
        with self.assertRaisesRegex(ValueError, "infeasible"):
            solve_target_return_portfolio([0.05, 0.05], np.eye(2), 0.06, long_only=False)
        with self.assertRaisesRegex(RuntimeError, "unbounded"):
            solve_risk_aversion_portfolio([0.03, 0.11], np.zeros((2, 2)), 1, long_only=False)
        with self.assertRaises(ValueError):
            generate_efficient_frontier(self.mu, self.Sigma, long_only=False)
        with self.assertRaises(ValueError):
            generate_efficient_frontier(self.mu, self.Sigma, n_points=1)

    def test_singular_covariance_and_constant_returns(self) -> None:
        covariance = np.ones((2, 2)) * 0.01
        validate_inputs([0.05, 0.05], covariance)
        with self.assertRaises(ValueError):
            analytical_gmv(covariance)
        result = solve_gmv_cvxpy([0.05, 0.05], covariance, long_only=True)
        self.assertAlmostEqual(result.weights.sum(), 1, places=8)
        # Arbitrary equal-risk solutions could be dominated; the sampler rejects
        # singular covariance rather than mislabelling them as an efficient set.
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            generate_efficient_frontier([0.04, 0.08], covariance)
        frontier = generate_efficient_frontier([0.05, 0.05], np.eye(2))
        self.assertEqual(len(frontier), 1)


if __name__ == "__main__":
    unittest.main()
