# Markowitz Portfolio Optimization

A Python implementation of Markowitz mean-variance portfolio optimization, including constrained quadratic programming, efficient frontier construction, and risk-aversion analysis.

## Overview

This compact project studies the trade-off between expected return and risk using three illustrative synthetic assets. It implements the global minimum-variance portfolio both analytically and numerically, then introduces long-only and minimum-return constraints before tracing the long-only efficient frontier.

The inputs are annualized synthetic values, chosen to isolate the optimization mechanics. They are **not** historical estimates or market forecasts.

## Mathematical Formulation

For portfolio weights \(w\), expected return vector \(\mu\), and covariance matrix \(\Sigma\):

\[
\mathbb{E}[R_p] = \mu^\top w, \qquad
\sigma_p^2 = w^\top \Sigma w, \qquad
\sigma_p = \sqrt{w^\top \Sigma w}.
\]

Covariance is essential: its diagonal entries quantify standalone variances, while off-diagonal entries quantify co-movement and therefore diversification effects.

### Global minimum variance

\[
\begin{aligned}
\min_w \quad & w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^\top w = 1.
\end{aligned}
\]

For the unconstrained case, the analytical solution is

\[
w_{\mathrm{GMV}} =
\frac{\Sigma^{-1}\mathbf{1}}
{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}.
\]

### Long-only target-return optimization

\[
\begin{aligned}
\min_w \quad & w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^\top w = 1, \\
& w \geq 0, \\
& \mu^\top w \geq R_{\mathrm{target}}.
\end{aligned}
\]

### Risk-aversion formulation

\[
\begin{aligned}
\max_w \quad & \mu^\top w - \lambda w^\top \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^\top w = 1,\quad w \geq 0,
\end{aligned}
\]

where a larger \(\lambda\) expresses stronger aversion to variance.

## Project Structure

~~~text
markowitz-portfolio-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── run_analysis.py
├── notebooks/
│   └── markowitz_portfolio_optimization.ipynb
├── src/
│   ├── __init__.py
│   └── portfolio_optimization.py
├── figures/
│   ├── efficient_frontier.png
│   ├── risk_aversion_on_frontier.png
│   └── lambda_weights.png
└── results/
    ├── portfolio_summary.csv
    └── lambda_sensitivity.csv
~~~

## Experiments

1. **Unconstrained GMV** — compares the closed-form GMV weights with a CVXPY quadratic-programming solution.
2. **Long-only GMV** — adds \(w_i \geq 0\) to examine the effect of banning short positions.
3. **Target-return optimization** — minimizes risk while requiring an 8% expected return.
4. **Efficient frontier** — solves 80 feasible long-only target-return problems from the GMV point to the highest asset return.
5. **Risk aversion** — evaluates \(\lambda \in \{0.1, 0.5, 1, 2, 5, 10, 20\}\).

## Key Results

These figures come from the committed output files generated with the command *python run_analysis.py*.

| Portfolio | Expected return | Variance | Volatility | Asset A | Asset B | Asset C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytical GMV | 6.7001% | 0.008496 | 9.2171% | 3.3037% | 73.8247% | 22.8717% |
| CVXPY GMV (unconstrained) | 6.7001% | 0.008496 | 9.2171% | 3.3037% | 73.8247% | 22.8717% |
| Long-only GMV | 6.7001% | 0.008496 | 9.2171% | 3.3037% | 73.8247% | 22.8717% |
| Long-only target return (8%) | 8.0000% | 0.009338 | 9.6633% | 2.4390% | 55.7491% | 41.8118% |

The analytical and CVXPY GMV weights match within the configured numerical tolerance. The long-only constraint is inactive for these inputs because the unconstrained GMV solution is already non-negative. The 8% return constraint is binding, which raises volatility relative to the GMV portfolio.

Detailed, machine-readable outputs are in [results/portfolio_summary.csv](results/portfolio_summary.csv) and [results/lambda_sensitivity.csv](results/lambda_sensitivity.csv).

## Efficient Frontier

![Long-only efficient frontier](figures/efficient_frontier.png)

The efficient branch begins at the long-only GMV portfolio. Targets below its 6.7001% expected return would reproduce the GMV solution and are therefore excluded.

## Risk Aversion

![Risk-aversion portfolios on the frontier](figures/risk_aversion_on_frontier.png)

![Weights across risk-aversion levels](figures/lambda_weights.png)

For small \(\lambda\), the objective favors return and selects Asset C. As \(\lambda\) increases, allocations move toward the low-variance diversified region near GMV.

## Installation

~~~bash
git clone https://github.com/YOUR_USERNAME/markowitz-portfolio-optimization.git
cd markowitz-portfolio-optimization
python -m pip install -r requirements.txt
~~~

## Usage

Run the full analysis from the project root:

~~~bash
python run_analysis.py
~~~

To explore the accompanying research note:

~~~bash
jupyter notebook
~~~

Then open **notebooks/markowitz_portfolio_optimization.ipynb**.

## Tech Stack

- Python
- NumPy
- Pandas
- CVXPY
- Matplotlib
- Jupyter

## Limitations

- Expected returns and the covariance matrix are assumed known.
- The data are synthetic rather than estimated from market observations.
- The model omits estimation error, transaction costs, and dynamic rebalancing.

## Future Work

- Real ETF data
- Rolling out-of-sample backtesting
- Transaction costs
- Conditional Value at Risk (CVaR)
- Robust optimization
