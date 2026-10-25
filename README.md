# Options Pricing and Hedging Lab

A reproducible Python research project for European option pricing, Monte Carlo estimation, implied-volatility analysis and discrete delta hedging under geometric Brownian motion. The experiments compare numerical estimates with analytical prices, recover volatility from option premiums, and measure how discrete rebalancing and transaction costs affect hedge outcomes.

## What's implemented

- Black–Scholes prices for European calls and puts, including continuous dividend yield and deterministic limiting cases.
- Analytic Delta, Gamma, Vega, Theta and Rho.
- Plain Monte Carlo, antithetic variates and a discounted-stock control variate, with standard errors and approximate 95% confidence intervals.
- Bracketed implied-volatility inversion and a surface builder for quotes across strikes and expiries.
- Discrete self-financing delta-hedging simulation with configurable rebalancing frequency, path drift, random seed and proportional transaction costs.
- CSV result tables, SVG figures and automated tests.

## Model and conventions

The underlying follows geometric Brownian motion,

    dS_t = (mu - q) S_t dt + sigma S_t dW_t,

where q is a continuous dividend yield and mu = r for risk-neutral simulations. Black–Scholes pricing discounts risk-neutral European payoffs at the continuously compounded rate r. Rates, volatility and time are annualized; volatility is entered as a decimal.

The Monte Carlo control variate is the discounted terminal stock price, with expectation S0 exp(-qT). For antithetic sampling, uncertainty is estimated from independent pair averages, not individual correlated payoffs. Confidence intervals use a normal approximation.

The hedge simulator starts with the Black–Scholes premium and trades the underlying to match option delta. Cash accrues at r, dividend income on the stock position is credited to cash, and trading costs are charged on traded notional. Terminal hedging error is defined as **replicating portfolio value minus option payoff**. Positive error is a surplus; negative error is a shortfall. By default, paths are generated with risk-neutral drift. A different stock drift can be supplied to examine hedging away from that measure.

## Install

Python 3.10 or newer is required.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"

Alternatively, install the dependencies from requirements.txt with pip.

## Run the experiments

    python -m options_lab

To use a different output directory, random seed or hedge simulation size:

    python -m options_lab --output-dir results --seed 20261025 --hedge-paths 5000

Run the tests with:

    python -m pytest -q

## Baseline pricing example

For an at-the-money European call with S0 = 100, K = 100, T = 1, r = 3%, sigma = 25%, and no dividends, the analytical Black–Scholes value is approximately **11.348477**. The experiment runner compares three estimators using a fixed seed and 100,000 simulated terminal stock values.

## Results

- [Monte Carlo estimator comparison](results/monte_carlo_comparison.csv)
- [Monte Carlo convergence experiment](results/mc_convergence.csv)
- [Implied-volatility surface data](results/implied_volatility_surface.csv)
- [Delta-hedging summary](results/delta_hedging_summary.csv)
- [Monte Carlo absolute pricing error](results/mc_absolute_error.svg)
- [Monte Carlo estimated standard error](results/mc_standard_error.svg)
- [Implied-volatility surface](results/implied_volatility_surface.svg)
- [Hedging RMSE vs rebalancing frequency](results/delta_hedging_rmse.svg)
- [Terminal hedging-error ranges](results/delta_hedging_error_distribution.svg)

The volatility-surface experiment uses a deterministic synthetic option chain to test the inversion workflow. It is not real market data. The Monte Carlo convergence plot uses one seed per path budget, so absolute error can fluctuate rather than decrease monotonically. These experiments are for studying model behaviour, not evidence of a trading strategy's profitability.

## Assumptions and limitations

The models assume European exercise and constant volatility and rates. The lab does not include market-data ingestion, calibration to a live option chain, early-exercise pricing, stochastic volatility, jumps, market microstructure, or a production execution model. The implied-volatility surface builder accepts quotes from a DataFrame or iterable of mappings; external quotes should be checked for stale prices, bid/ask quality and arbitrage consistency before interpretation.
