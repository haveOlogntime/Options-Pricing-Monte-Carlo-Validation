# Options Pricing and Hedging Lab

A Python research project for European option pricing, Monte Carlo estimation, implied-volatility analysis and discrete delta hedging under geometric Brownian motion. The experiments compare estimators against an analytical benchmark, recover volatility from option premiums, and quantify how rebalancing frequency and transaction costs affect hedge outcomes.

## Implemented

- Black–Scholes call and put prices, continuous dividend yield, and deterministic expiry/zero-volatility limits.
- Analytic Delta, Gamma, Vega, Theta and Rho.
- Plain Monte Carlo, antithetic variates and a discounted-stock control variate, with standard errors and approximate 95% confidence intervals.
- Convergence diagnostics and a repeated-seed study reporting bias, empirical variance, RMSE, reported standard error, interval coverage and variance reduction.
- Brent-solver implied volatility and a surface builder for quotes across strikes and expiries.
- Discrete delta-hedging simulation with transaction costs, dividend cash flows, rebalancing frequency and terminal error statistics.
- Reproducible CSV tables, SVG figures, command-line execution, package metadata and automated tests.

## Model conventions

The stock price follows geometric Brownian motion. Under risk-neutral pricing, its total expected return is the continuously compounded interest rate r; q is the continuous dividend yield. Volatility and time are annualized, with volatility entered as a decimal. Option premiums, spot and strike must use the same currency and per-unit convention.

The Monte Carlo control variate is the discounted terminal stock price, whose expectation is S0 × exp(-qT). Antithetic standard errors use independent pair averages. Confidence intervals are normal approximations.

The hedger receives the Black–Scholes premium, buys the option delta, finances the remaining amount in a cash account accruing at r, and credits estimated dividend cash flows on the held stock position. Proportional costs are charged on traded notional at each rebalance. Terminal hedge error is **replicating portfolio value minus option payoff**: a negative value represents a shortfall. Dividend cash flows are approximated discretely over each interval, including the final interval.

## Install

Python 3.10 or newer is required.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"

To install dependencies without editable installation, use:

    python -m pip install -r requirements.txt

## Run

Generate the checked-in tables and plots with the default settings:

    python -m options_lab
    python -m pytest -q

Control the experiment sizes and seed with:

    python -m options_lab --output-dir results --seed 20261025 --hedge-paths 3000 --comparison-paths 20000 --comparison-repeats 30

The repeated-seed study uses independent seeds at a fixed path budget. Empirical variance and confidence-interval coverage are estimates based on the requested number of repetitions; they will vary with the repetition count and seed. The single-seed convergence plot is useful for inspecting path-budget behaviour but absolute error need not decrease monotonically.

## Option quote input

By default, the volatility-surface example uses a **synthetic illustrative option chain**. It is not market data. To calculate implied volatility from downloaded quotes, supply a CSV:

    python -m options_lab --quotes-csv path/to/option_quotes.csv

Required columns:

| Column | Meaning |
| --- | --- |
| strike | Strike price |
| maturity | Time to expiry in years, not a date string |
| market_price | Option premium per underlying unit |

Optional columns are option_type (call or put, default call), spot, rate and dividend_yield. Row-specific model inputs override the defaults. Convert vendor fields to these units before running the solver. Check quote timestamps, bid/ask spreads, contract multipliers and no-arbitrage consistency yourself; the code does not certify quote quality or fit an arbitrage-free surface. The resulting premiums and implied volatilities are written to results/implied_volatility_surface.csv. The program does not fetch live market data.

## Baseline

For an at-the-money European call with S0 = 100, K = 100, T = 1, r = 3%, sigma = 25% and no dividends, the analytical Black–Scholes value is approximately **11.348477**.

## Results

- [Monte Carlo estimator comparison](results/monte_carlo_comparison.csv)
- [Single-seed convergence data](results/mc_convergence.csv)
- [Repeated-seed estimator summary](results/monte_carlo_repeated_seed_summary.csv)
- [Individual repeated-seed runs](results/monte_carlo_repeated_seed_runs.csv)
- [Implied-volatility data](results/implied_volatility_surface.csv)
- [Delta-hedging summary](results/delta_hedging_summary.csv)
- [Monte Carlo absolute pricing error](results/mc_absolute_error.svg)
- [Estimated standard error by path budget](results/mc_standard_error.svg)
- [Estimator variability across seeds](results/mc_repeated_seed_variability.svg)
- [Implied volatility by strike and expiry](results/implied_volatility_surface.svg)
- [Hedging RMSE vs rebalancing frequency](results/delta_hedging_rmse.svg)
- [Terminal hedging-error ranges](results/delta_hedging_error_distribution.svg)

## Limitations

The project assumes European exercise and constant rates and volatility within each Black–Scholes calculation. It does not include a live market-data feed, calibration to a risk-neutral surface, American exercise, stochastic volatility, jumps, market microstructure, or a production execution model. It is for numerical study, not a trading strategy.
