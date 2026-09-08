"""Mean-variance optimization with fixed, annualized illustrative inputs.

Covariance is central to portfolio risk: w.T @ Sigma @ w includes both
individual variances and cross-asset covariance terms. Optimizing only the
individual asset volatilities would miss the benefits of diversification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
FEASIBILITY_TOL = 1e-8
BINDING_TOL = 1e-7


def example_data() -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Return fresh copies of synthetic annualized mu, Sigma and asset names."""
    mu = np.array([0.08, 0.05, 0.12])
    Sigma = np.array([
        [0.04, 0.006, 0.012],
        [0.006, 0.01, 0.004],
        [0.012, 0.004, 0.0225],
    ])
    return mu, Sigma, ("Asset A", "Asset B", "Asset C")


def _vector(values: ArrayLike, name: str) -> FloatArray:
    """Convert a nonempty, finite real vector to float64."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _covariance(Sigma: ArrayLike) -> FloatArray:
    """Validate covariance; tolerate only scale-relative floating-point noise."""
    if np.iscomplexobj(Sigma):
        raise ValueError("Sigma must be real-valued.")
    matrix = np.asarray(Sigma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Sigma must be a nonempty square matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Sigma must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Sigma must contain only finite values.")
    scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("Sigma must be symmetric.")
    matrix = (matrix + matrix.T) / 2.0
    if np.linalg.eigvalsh(matrix)[0] < -1e-12 * scale:
        raise ValueError("Sigma must be positive semidefinite.")
    return matrix


def validate_inputs(mu: ArrayLike, Sigma: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Check dimensions, finiteness, symmetry and PSD; return validated arrays.

    PSD is sufficient for the convex numerical problems. The inverse-based
    analytical GMV formula additionally requires positive definiteness.
    Symmetry and PSD tests admit only tiny floating-point discrepancies.
    """
    returns, covariance = _vector(mu, "mu"), _covariance(Sigma)
    if covariance.shape != (returns.size, returns.size):
        raise ValueError("Sigma dimensions must match the length of mu.")
    return returns, covariance


def _scalar(value: float, name: str) -> float:
    """Require a finite real scalar, rather than an array or NaN."""
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def portfolio_return(w: ArrayLike, mu: ArrayLike) -> float:
    """Compute expected portfolio return in decimal units (0.08 means 8%)."""
    weights, returns = _vector(w, "w"), _vector(mu, "mu")
    if weights.shape != returns.shape:
        raise ValueError("w and mu must have the same length.")
    return float(returns @ weights)


def portfolio_variance(w: ArrayLike, Sigma: ArrayLike) -> float:
    """Compute w.T @ Sigma @ w, including cross-asset covariance terms."""
    weights, covariance = _vector(w, "w"), _covariance(Sigma)
    if covariance.shape != (weights.size, weights.size):
        raise ValueError("Sigma dimensions must match the length of w.")
    # The PSD check above permits roundoff at zero; variance cannot be negative.
    return max(float(weights @ covariance @ weights), 0.0)


def portfolio_volatility(w: ArrayLike, Sigma: ArrayLike) -> float:
    """Return the square root of portfolio variance, in decimal units."""
    return float(np.sqrt(portfolio_variance(w, Sigma)))


def _require_positive_definite(Sigma: FloatArray, operation: str) -> None:
    """Guard operations that rely on an invertible covariance and unique GMV."""
    eigenvalues = np.linalg.eigvalsh(Sigma)
    scale = max(eigenvalues[-1], np.finfo(float).tiny)
    if eigenvalues[0] <= 1e-12 * scale:
        raise ValueError(
            f"{operation} requires numerically positive-definite Sigma; "
            "individual CVXPY solvers support singular PSD inputs."
        )


def analytical_gmv(Sigma: ArrayLike) -> FloatArray:
    """Return budget-only GMV weights; short positions are permitted.

    Solve Sigma x = 1, then normalize x. This implements
    Sigma^{-1}1 / (1.T Sigma^{-1}1) without explicitly forming an inverse.
    Singular or numerically ill-conditioned PSD matrices should use CVXPY.
    """
    covariance = _covariance(Sigma)
    _require_positive_definite(covariance, "Analytical GMV")
    direction = np.linalg.solve(covariance, np.ones(covariance.shape[0]))
    return direction / direction.sum()


@dataclass
class PortfolioResult:
    """Optimal weights, comparable risk/return metrics and constraint duals.

    Weights are the solver's output: they are never clipped or renormalized.
    The return multiplier uses the convention target_return - mu.T @ w <= 0.
    """

    weights: FloatArray
    expected_return: float
    variance: float
    volatility: float
    status: str
    budget_dual: float | None = None
    nonneg_duals: FloatArray | None = None
    return_dual: float | None = None
    target_return: float | None = None

    @property
    def return_constraint_binding(self) -> bool | None:
        """Whether achieved return equals its lower bound within BINDING_TOL."""
        if self.target_return is None:
            return None
        return abs(self.expected_return - self.target_return) <= BINDING_TOL

    def to_record(self, asset_names: Sequence[str] | None = None) -> dict:
        """Return a CSV-friendly record; weights and returns remain decimals."""
        names = (
            list(asset_names) if asset_names is not None
            else [str(i + 1) for i in range(self.weights.size)]
        )
        if len(names) != self.weights.size or any(
            not isinstance(name, str) for name in names
        ):
            raise ValueError("asset_names must contain one string per asset.")
        columns = ["weight_" + name.replace(" ", "_") for name in names]
        if len(set(columns)) != len(columns):
            raise ValueError("asset_names must produce unique weight column names.")
        return {
            **dict(zip(columns, self.weights)),
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "status": self.status,
        }


def _solve_portfolio(
    mu: ArrayLike,
    Sigma: ArrayLike,
    *,
    long_only: bool,
    target_return: float | None = None,
    risk_aversion: float | None = None,
) -> PortfolioResult:
    """Build and solve a convex QP, checking status before accessing weights."""
    mu, Sigma = validate_inputs(mu, Sigma)
    if not isinstance(long_only, (bool, np.bool_)):
        raise ValueError("long_only must be a boolean.")
    if target_return is not None:
        target_return = _scalar(target_return, "target_return")
        if long_only and target_return > float(mu.max()):
            raise ValueError("Infeasible target: a long-only return cannot exceed max(mu).")
    if risk_aversion is not None:
        risk_aversion = _scalar(risk_aversion, "risk_aversion")
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be strictly positive.")
    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("CLARABEL is required; install the project's CVXPY dependency.")

    w = cp.Variable(mu.size)
    budget = cp.sum(w) == 1
    nonnegative = w >= 0 if long_only else None
    minimum_return = (
        mu @ w >= target_return if target_return is not None else None
    )
    constraints = [budget]
    constraints.extend(c for c in (nonnegative, minimum_return) if c is not None)
    # psd_wrap is used only AFTER independently checking Sigma's eigenvalues.
    variance = cp.quad_form(w, cp.psd_wrap(Sigma))
    objective = (
        cp.Minimize(variance) if risk_aversion is None
        else cp.Maximize(mu @ w - risk_aversion * variance)
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
        max_iter=200,
    )
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        raise ValueError(f"Portfolio constraints are infeasible (status={problem.status}).")
    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"Solver did not establish an optimum (status={problem.status}).")
    weights = np.asarray(w.value, dtype=float).reshape(-1).copy()
    if (
        not np.all(np.isfinite(weights))
        or abs(weights.sum() - 1.0) > FEASIBILITY_TOL
    ):
        raise RuntimeError("Solver output failed the finite-weight / budget check.")
    if long_only and weights.min() < -FEASIBILITY_TOL:
        raise RuntimeError("Solver output violates the long-only constraint.")
    achieved_return = portfolio_return(weights, mu)
    if (
        target_return is not None
        and achieved_return < target_return - FEASIBILITY_TOL
    ):
        raise RuntimeError("Solver output violates the minimum-return constraint.")
    risk = portfolio_variance(weights, Sigma)
    return PortfolioResult(
        weights=weights,
        expected_return=achieved_return,
        variance=risk,
        volatility=float(np.sqrt(risk)),
        status=problem.status,
        budget_dual=float(budget.dual_value),
        nonneg_duals=(
            None if nonnegative is None
            else np.asarray(nonnegative.dual_value).copy()
        ),
        return_dual=None if minimum_return is None else float(minimum_return.dual_value),
        target_return=target_return,
    )


def solve_gmv_cvxpy(
    mu: ArrayLike, Sigma: ArrayLike, long_only: bool = False,
) -> PortfolioResult:
    """Minimize variance with full investment and optional nonnegative weights."""
    return _solve_portfolio(mu, Sigma, long_only=long_only)


def solve_target_return_portfolio(
    mu: ArrayLike,
    Sigma: ArrayLike,
    target_return: float,
    long_only: bool = True,
) -> PortfolioResult:
    """Minimize variance subject to full investment and a minimum return.

    A target below the corresponding GMV return can be slack. Under long-only
    constraints max(mu) is the highest feasible return; larger targets raise
    ValueError instead of exposing a missing optimizer solution.
    """
    return _solve_portfolio(
        mu, Sigma, long_only=long_only, target_return=target_return,
    )


def generate_efficient_frontier(
    mu: ArrayLike,
    Sigma: ArrayLike,
    n_points: int = 80,
    *,
    long_only: bool = True,
    max_return: float | None = None,
    asset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sample minimum-variance portfolios from GMV toward higher returns.

    Default bounds are the long-only GMV return and max(mu), including both
    endpoints. Shorting requires an explicit finite max_return because the
    feasible return need not have an upper bound. Equal endpoint returns yield
    one GMV point. Failed solves raise; no invalid row is silently saved.

    This sampler requires positive definiteness. With singular covariance,
    nonunique minimum-variance solutions need an additional return tie-break;
    sampling an arbitrary GMV could otherwise include dominated portfolios.
    """
    mu, Sigma = validate_inputs(mu, Sigma)
    _require_positive_definite(Sigma, "Efficient frontier sampling")
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, (int, np.integer))
        or n_points < 2
    ):
        raise ValueError("n_points must be an integer of at least 2.")
    gmv = solve_gmv_cvxpy(mu, Sigma, long_only=long_only)
    lower = min(gmv.expected_return, float(mu.max())) if long_only else gmv.expected_return
    if max_return is None:
        if not long_only:
            raise ValueError("Set max_return explicitly when short selling is allowed.")
        upper = float(mu.max())
    else:
        upper = _scalar(max_return, "max_return")
    if upper < lower - FEASIBILITY_TOL:
        raise ValueError("max_return must be at least the GMV expected return.")
    if long_only and upper > float(mu.max()):
        raise ValueError("max_return cannot exceed max(mu) for a long-only frontier.")
    if abs(upper - lower) <= FEASIBILITY_TOL:
        return pd.DataFrame([{"target_return": lower, **gmv.to_record(asset_names)}])
    records = []
    for index, target in enumerate(np.linspace(lower, upper, n_points)):
        # Reuse the GMV endpoint to avoid an unnecessary active return bound.
        result = (
            gmv if index == 0 else solve_target_return_portfolio(
                mu, Sigma, float(target), long_only,
            )
        )
        records.append({"target_return": float(target), **result.to_record(asset_names)})
    return pd.DataFrame(records)


def solve_risk_aversion_portfolio(
    mu: ArrayLike,
    Sigma: ArrayLike,
    risk_aversion: float,
    long_only: bool = True,
) -> PortfolioResult:
    """Maximize mu.T @ w - risk_aversion * w.T @ Sigma @ w, lambda > 0.

    There is no factor of 1/2 and no minimum-return constraint in this model.
    Lambda depends on the return units and horizon; here both are annualized
    and returns are decimals. Larger lambda penalizes variance more strongly.
    """
    return _solve_portfolio(
        mu, Sigma, long_only=long_only, risk_aversion=risk_aversion,
    )
