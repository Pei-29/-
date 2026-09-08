"""Run the complete illustrative study and regenerate committed outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Also render on machines without a display server.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.portfolio_optimization import (
    PortfolioResult,
    analytical_gmv,
    example_data,
    generate_efficient_frontier,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    solve_gmv_cvxpy,
    solve_risk_aversion_portfolio,
    solve_target_return_portfolio,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_RETURN = 0.08
RISK_AVERSIONS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _frontier_axes(frontier: pd.DataFrame, gmv: PortfolioResult, title: str) -> tuple:
    """Build a restrained, consistently labelled annualized risk-return chart."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")
    ax.plot(frontier["volatility"], frontier["expected_return"], color="#305d7b", linewidth=2.3, label="Efficient frontier (long-only)")
    ax.scatter(gmv.volatility, gmv.expected_return, marker="*", s=155, color="#243c4c", edgecolor="white", linewidth=0.6, zorder=5, label="Global minimum variance (long-only)")
    ax.set(title=title, xlabel="Annualized portfolio volatility", ylabel="Annualized expected return")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12, y=0.18)
    return fig, ax


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    gmv: PortfolioResult,
    output_path: Path,
) -> Path:
    """Save the efficient frontier and its long-only GMV endpoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10.5, "axes.titlesize": 14, "legend.fontsize": 9}):
        fig, ax = _frontier_axes(frontier, gmv, "Long-only efficient frontier | Synthetic annualized inputs")
        ax.annotate(f"GMV: {gmv.expected_return:.2%} return, {gmv.volatility:.2%} volatility", (gmv.volatility, gmv.expected_return), xytext=(12, -28), textcoords="offset points", fontsize=9)
        ax.legend(loc="upper left", frameon=False)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return output_path


def plot_risk_aversion(
    frontier: pd.DataFrame,
    gmv: PortfolioResult,
    lambda_results: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save lambda choices on the frontier, grouping coincident point labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10.5, "axes.titlesize": 14, "legend.fontsize": 9}):
        fig, ax = _frontier_axes(frontier, gmv, "Risk aversion on the frontier | Synthetic annualized inputs")
        ax.scatter(lambda_results["volatility"], lambda_results["expected_return"], s=44, color="#bd622b", edgecolor="white", linewidth=0.7, zorder=4, label=r"Optimal portfolios for $\lambda > 0$")
        groups: list[dict] = []
        for _, row in lambda_results.iterrows():
            point = np.array([row["volatility"], row["expected_return"]])
            matching = next((group for group in groups if np.allclose(point, group["point"], atol=1e-7, rtol=0)), None)
            if matching is None:
                groups.append({"point": point, "lambdas": [row["lambda"]]})
            else:
                matching["lambdas"].append(row["lambda"])
        for group in groups:
            label = r"$\lambda$ = " + ", ".join(f"{value:g}" for value in group["lambdas"])
            offset = (-12, 13) if len(group["lambdas"]) > 1 else (12, -3)
            align = "right" if len(group["lambdas"]) > 1 else "left"
            ax.annotate(label, group["point"], xytext=offset, textcoords="offset points", ha=align, fontsize=9, color="#8c481e")
        ax.legend(loc="upper left", frameon=False)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return output_path


def main() -> None:
    """Solve all five experiments, check numerical identities and save results."""
    mu, Sigma, asset_names = example_data()
    analytical_weights = analytical_gmv(Sigma)
    analytical = PortfolioResult(
        weights=analytical_weights,
        expected_return=portfolio_return(analytical_weights, mu),
        variance=portfolio_variance(analytical_weights, Sigma),
        volatility=portfolio_volatility(analytical_weights, Sigma),
        status="analytical",
    )
    unconstrained = solve_gmv_cvxpy(mu, Sigma)
    long_only = solve_gmv_cvxpy(mu, Sigma, long_only=True)
    target = solve_target_return_portfolio(mu, Sigma, TARGET_RETURN)
    np.testing.assert_allclose(analytical_weights, unconstrained.weights, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(unconstrained.weights, long_only.weights, atol=1e-7, rtol=1e-6)
    rows = []
    for label, result in [
        ("GMV (analytical)", analytical),
        ("GMV (CVXPY, shorting allowed)", unconstrained),
        ("GMV (CVXPY, long-only)", long_only),
        ("Target return >= 8% (long-only)", target),
    ]:
        rows.append({
            "portfolio": label,
            **result.to_record(asset_names),
            "target_return": result.target_return,
            "return_constraint_binding": result.return_constraint_binding,
        })
    summary = pd.DataFrame(rows)
    frontier = generate_efficient_frontier(mu, Sigma, n_points=80, asset_names=asset_names)
    lambda_results = pd.DataFrame([
        {"lambda": value, **solve_risk_aversion_portfolio(mu, Sigma, value).to_record(asset_names)}
        for value in RISK_AVERSIONS
    ])

    # These are economic properties, in addition to each solver's feasibility checks.
    if np.any(np.diff(frontier["expected_return"]) < -1e-8) or np.any(np.diff(frontier["variance"]) < -1e-8):
        raise RuntimeError("Frontier risk and return should be nondecreasing above GMV.")
    if np.any(np.diff(lambda_results["variance"]) > 1e-8) or np.any(np.diff(lambda_results["expected_return"]) > 1e-8):
        raise RuntimeError("Risk and return should be nonincreasing as lambda increases.")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "portfolio_summary.csv", index=False, float_format="%.12g")
    lambda_results.to_csv(results_dir / "lambda_sensitivity.csv", index=False, float_format="%.12g")
    plot_efficient_frontier(frontier, long_only, PROJECT_ROOT / "figures" / "efficient_frontier.png")
    plot_risk_aversion(frontier, long_only, lambda_results, PROJECT_ROOT / "figures" / "risk_aversion_on_frontier.png")

    columns = ["portfolio", "weight_Asset_A", "weight_Asset_B", "weight_Asset_C", "expected_return", "variance", "volatility"]
    print("Synthetic annualized inputs; weights and returns are decimals.\n")
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nAnalytical / CVXPY maximum weight difference: {np.max(np.abs(analytical_weights - unconstrained.weights)):.2e}")
    print(f"Long-only constraints inactive at GMV: {bool(np.all(long_only.weights > 1e-7))}")
    print(f"8% minimum-return constraint binding: {target.return_constraint_binding}")
    print(f"\nLambda sensitivity:\n{lambda_results.drop(columns='status').to_string(index=False, float_format=lambda value: f'{value:.6f}')}")
    print(f"\nSaved 2 CSV tables and 2 PNG figures; {len(frontier)} feasible frontier points.")
    print("All numerical checks passed.")


if __name__ == "__main__":
    main()
