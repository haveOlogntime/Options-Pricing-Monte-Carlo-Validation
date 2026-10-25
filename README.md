# Options Pricing and Hedging Lab

A Python project for studying European option prices and Monte Carlo estimation under the Black–Scholes model. The current implementation compares simulation estimators with the analytical price and records how estimates and uncertainty change with the number of simulated paths.

## Implemented

- Black–Scholes prices for European calls and puts, assuming no dividends.
- Delta, Gamma, Vega and Theta.
- Monte Carlo pricing with plain sampling, antithetic variates and a control variate.
- Estimated standard errors and approximate 95% confidence intervals.
- Reproducible experiments, CSV output, convergence plots and unit tests.

The repository currently focuses on pricing and estimator diagnostics. A discrete-time delta-hedging simulation, implied-volatility solver and volatility-surface analysis are not implemented yet.

## Model

The underlying follows risk-neutral geometric Brownian motion,

\[
dS_t = rS_t\,dt + \sigma S_t\,dW_t,
\]

with constant interest rate \(r\), volatility \(\sigma\), and no dividends. The option is European, so its value is the discounted risk-neutral expectation of its terminal payoff.

The implementation uses the analytical Black–Scholes price as a benchmark. The Monte Carlo estimators are:

- **Plain sampling:** averages discounted terminal payoffs.
- **Antithetic variates:** pairs draws \(Z\) and \(-Z\) and averages each pair's payoff.
- **Control variate:** adjusts the discounted payoff using the discounted terminal stock price, whose expectation is \(S_0\) under this model.

For antithetic sampling, the reported standard error is calculated from the pair averages. Confidence intervals use a normal approximation. The control-variate coefficient is estimated from the simulated sample.

## Baseline experiment

The runner uses an at-the-money European call with these inputs:

| Parameter | Value |
| --- | ---: |
| Initial stock price | 100 |
| Strike | 100 |
| Time to expiry | 1 year |
| Risk-free rate | 3% |
| Volatility | 25% |
| Monte Carlo paths | 100,000 |

The analytical Black–Scholes price is approximately **11.348477**. With the fixed seed in the script, the Monte Carlo comparison is:

| Estimator | Price estimate | Estimated standard error | Absolute error |
| --- | ---: | ---: | ---: |
| Plain | 11.375566 | 0.056717 | 0.027089 |
| Antithetic variates | 11.338172 | 0.043592 | 0.010305 |
| Control variate | 11.373839 | 0.022965 | 0.025362 |

These figures describe one run, not a general ranking of the estimators. The convergence experiment also uses one seed per path count, so absolute error need not decrease monotonically. Repeated independent runs are needed for a more reliable comparison of estimator variance.

## Setup

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Generate the CSV files and SVG plots:

```bash
python run_experiments.py
```

Run the test suite:

```bash
python -m pytest -q
```

## Results

- [Monte Carlo estimator comparison](results/monte_carlo_comparison.csv)
- [Convergence experiment data](results/mc_convergence.csv)
- [Absolute pricing error by path count](results/mc_absolute_error.svg)
- [Estimated standard error by path count](results/mc_standard_error.svg)

Running the experiment script overwrites these result files.

## Assumptions and limitations

The model assumes constant volatility and interest rates, risk-neutral geometric Brownian motion, European exercise and no dividends. It does not use market data or include calibration, early exercise, stochastic volatility, transaction costs or a hedging simulation. The results are for numerical study and are not a trading strategy.
