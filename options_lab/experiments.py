"""Reproducible pricing, volatility and delta-hedging experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .black_scholes import bs_greeks, bs_price
from .hedging import delta_hedge_simulation
from .implied_volatility import implied_volatility_surface, synthetic_option_chain
from .monte_carlo import mc_price
from .svg_plots import boxplot_chart, line_chart, range_chart, scatter_chart


def repeated_seed_comparison(S0: float, K: float, T: float, r: float, sigma: float,
                             q: float = 0.0, n_paths: int = 20_000,
                             n_repeats: int = 30, seed: int = 20261025,
                             option_type: str = "call") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare estimator errors over independent random seeds at a fixed budget."""
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 2:
        raise ValueError("n_repeats must be an integer of at least 2")
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 4:
        raise ValueError("n_paths must be an integer of at least 4")
    if n_paths % 2:
        raise ValueError("n_paths must be even for the antithetic estimator")
    benchmark = bs_price(S0, K, T, r, sigma, option_type, q)
    run_rows = []
    for repetition in range(n_repeats):
        run_seed = seed + repetition
        for method in ("plain", "antithetic", "control_variate"):
            result = mc_price(S0, K, T, r, sigma, n_paths=n_paths,
                              option_type=option_type, method=method, seed=run_seed, q=q)
            run_rows.append({
                "repetition": repetition + 1, "seed": run_seed, "method": method,
                "n_paths": n_paths, "price": float(result["price"]),
                "standard_error": float(result["standard_error"]),
                "ci95_low": float(result["ci95_low"]), "ci95_high": float(result["ci95_high"]),
                "absolute_error": abs(float(result["price"]) - benchmark),
                "covered_analytic_price": bool(float(result["ci95_low"]) <= benchmark <= float(result["ci95_high"])),
            })
    runs = pd.DataFrame(run_rows)
    plain_variance = float(runs.loc[runs["method"] == "plain", "price"].var(ddof=1))
    summaries = []
    for method, group in runs.groupby("method", sort=False):
        empirical_variance = float(group["price"].var(ddof=1))
        summaries.append({
            "method": method, "n_paths_per_run": n_paths, "n_repeats": n_repeats,
            "analytic_price": benchmark, "mean_estimate": float(group["price"].mean()),
            "bias": float(group["price"].mean() - benchmark),
            "empirical_standard_deviation": math_sqrt_nonnegative(empirical_variance),
            "empirical_variance": empirical_variance,
            "rmse": float(np.sqrt(np.mean((group["price"].to_numpy() - benchmark) ** 2))),
            "mean_reported_standard_error": float(group["standard_error"].mean()),
            "mean_absolute_error": float(group["absolute_error"].mean()),
            "nominal_95pct_interval_coverage": float(group["covered_analytic_price"].mean()),
            "mean_ci_width": float((group["ci95_high"] - group["ci95_low"]).mean()),
            "variance_reduction_vs_plain": (float(1.0 - empirical_variance / plain_variance)
                                              if plain_variance > 0 else float("nan")),
        })
    return pd.DataFrame(summaries), runs


def math_sqrt_nonnegative(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))


def run_experiments(output_dir: str | Path = "results", seed: int = 20261025,
                    hedge_paths: int = 3_000, comparison_paths: int = 20_000,
                    comparison_repeats: int = 30, quotes_csv: str | Path | None = None
                    ) -> dict[str, pd.DataFrame]:
    """Run all experiments and save results and figures in the output directory.

    If quotes_csv is supplied, it must have strike, maturity (years), and
    market_price columns. Optional columns are option_type, spot, rate and
    dividend_yield. Without it, a labelled synthetic option chain is used.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.25, 0.0
    analytic_price = bs_price(S0, K, T, r, sigma, "call", q)
    greeks = bs_greeks(S0, K, T, r, sigma, "call", q)
    print("European call | Black–Scholes model | no dividends")
    print(f"S0={S0:g}, K={K:g}, T={T:g}, r={r:.2%}, sigma={sigma:.2%}")
    print(f"Analytical price: {analytic_price:.6f}")
    print("Greeks (vega per 1.00 volatility change; theta per year):")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {name:>6}: {greeks[name]:.6f}")

    comparison_rows = []
    for method in ("plain", "antithetic", "control_variate"):
        result = mc_price(S0, K, T, r, sigma, n_paths=100_000,
                          option_type="call", method=method, seed=seed, q=q)
        result["bs_price"] = analytic_price
        result["absolute_error"] = abs(float(result["price"]) - analytic_price)
        comparison_rows.append(result)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "monte_carlo_comparison.csv", index=False)

    convergence_rows = []
    for n_paths in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000):
        for method in ("plain", "antithetic", "control_variate"):
            result = mc_price(S0, K, T, r, sigma, n_paths=n_paths,
                              option_type="call", method=method, seed=seed + n_paths, q=q)
            convergence_rows.append({
                "method": method, "n_paths": n_paths, "price": float(result["price"]),
                "standard_error": float(result["standard_error"]),
                "absolute_error": abs(float(result["price"]) - analytic_price),
                "ci95_low": float(result["ci95_low"]), "ci95_high": float(result["ci95_high"]),
            })
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(output / "mc_convergence.csv", index=False)

    repeated_summary, repeated_runs = repeated_seed_comparison(
        S0, K, T, r, sigma, q=q, n_paths=comparison_paths,
        n_repeats=comparison_repeats, seed=seed + 500_000,
    )
    repeated_summary.to_csv(output / "monte_carlo_repeated_seed_summary.csv", index=False)
    repeated_runs.to_csv(output / "monte_carlo_repeated_seed_runs.csv", index=False)

    methods = ("plain", "antithetic", "control_variate")
    for column, title, filename, ylabel in (
        ("absolute_error", "Monte Carlo pricing error", "mc_absolute_error.svg", "Absolute pricing error"),
        ("standard_error", "Monte Carlo estimated standard error", "mc_standard_error.svg", "Estimated standard error"),
    ):
        line_chart(
            output / filename,
            [{"label": method, "x": group["n_paths"].to_numpy(), "y": group[column].to_numpy()}
             for method, group in ((m, convergence[convergence["method"] == m]) for m in methods)],
            title, "Simulated terminal stock values", ylabel, x_log=True, y_log=True,
        )

    order = ["plain", "antithetic", "control_variate"]
    boxplot_chart(
        output / "mc_repeated_seed_variability.svg",
        [{"label": method, "values": repeated_runs.loc[repeated_runs["method"] == method, "price"].to_numpy()}
         for method in order],
        f"Estimator variability across {comparison_repeats} independent seeds",
        "Monte Carlo price estimate", benchmark=analytic_price,
    )

    if quotes_csv is None:
        quotes = synthetic_option_chain(S0=S0, r=r, q=q)
        quote_source = "synthetic illustrative quotes"
    else:
        quotes = pd.read_csv(quotes_csv)
        quote_source = f"user-supplied CSV: {Path(quotes_csv).name}"
    surface = implied_volatility_surface(quotes, S0=S0, r=r, q=q)
    surface.to_csv(output / "implied_volatility_surface.csv", index=False)
    if "synthetic_volatility" in surface.columns:
        recovery_error = float(np.max(np.abs(surface["synthetic_volatility"] - surface["implied_volatility"])))
        print(f"Maximum synthetic-volatility recovery error: {recovery_error:.3e}")
    scatter_chart(
        output / "implied_volatility_surface.svg",
        surface["strike"].to_numpy(), surface["maturity"].to_numpy(),
        surface["implied_volatility"].to_numpy(), "Implied volatility by strike and expiry",
        "Strike", "Time to expiry (years)", "Implied volatility",
    )

    hedge_rows = []
    for n_steps in (12, 52, 252):
        for cost_rate in (0.0, 0.001):
            result = delta_hedge_simulation(S0, K, T, r, sigma, n_paths=hedge_paths,
                                            n_steps=n_steps, option_type="call", q=q,
                                            transaction_cost_rate=cost_rate, seed=seed + n_steps)
            hedge_rows.append({key: result[key] for key in (
                "n_paths", "n_steps", "initial_option_price", "transaction_cost_rate",
                "mean_hedging_error", "std_hedging_error", "rmse_hedging_error",
                "mean_absolute_error", "error_q05", "error_median", "error_q95", "mean_transaction_cost",
            )})
    hedging = pd.DataFrame(hedge_rows)
    hedging.to_csv(output / "delta_hedging_summary.csv", index=False)
    cost_series = []
    for cost_rate, group in hedging.groupby("transaction_cost_rate", sort=True):
        cost_series.append({
            "label": f"transaction costs {cost_rate * 10_000:.0f} bp",
            "x": group["n_steps"].to_numpy(), "y": group["rmse_hedging_error"].to_numpy(),
            "color": "#2563eb" if cost_rate == 0.0 else "#dc2626",
        })
    line_chart(output / "delta_hedging_rmse.svg", cost_series,
               "Discrete delta hedging: error vs rebalancing frequency",
               "Rehedging intervals over the option's life", "Root-mean-square hedging error")
    labels = [f"{int(row.n_steps)} steps, {row.transaction_cost_rate * 10_000:.0f} bp"
              for row in hedging.itertuples()]
    range_chart(output / "delta_hedging_error_distribution.svg", labels,
                hedging["error_q05"].to_numpy(), hedging["error_median"].to_numpy(),
                hedging["error_q95"].to_numpy(), "Terminal hedging-error ranges",
                "Portfolio value minus option payoff")

    print("\nMonte Carlo repeated-seed summary:")
    print(repeated_summary[["method", "bias", "empirical_standard_deviation", "rmse",
                            "nominal_95pct_interval_coverage", "variance_reduction_vs_plain"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nDelta-hedging summary:")
    print(hedging[["n_steps", "transaction_cost_rate", "mean_hedging_error",
                   "rmse_hedging_error", "mean_transaction_cost"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nVolatility inputs: {quote_source}")
    print(f"Results written to {output.resolve()}")
    return {"monte_carlo_comparison": comparison, "monte_carlo_convergence": convergence,
            "monte_carlo_repeated_seed_summary": repeated_summary, "monte_carlo_repeated_seed_runs": repeated_runs,
            "implied_volatility_surface": surface, "delta_hedging_summary": hedging}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run option pricing, volatility and hedging experiments.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV tables and SVG figures")
    parser.add_argument("--seed", type=int, default=20261025, help="Base random seed")
    parser.add_argument("--hedge-paths", type=int, default=3000, help="Number of paths per hedging experiment")
    parser.add_argument("--comparison-paths", type=int, default=20000, help="Path budget for each repeated-seed run (must be even)")
    parser.add_argument("--comparison-repeats", type=int, default=30, help="Number of independent Monte Carlo repetitions")
    parser.add_argument("--quotes-csv", default=None, help="Optional option-quote CSV with strike, maturity (years), market_price and optional option_type/spot/rate/dividend_yield")
    args = parser.parse_args()
    if args.hedge_paths < 2:
        parser.error("--hedge-paths must be at least 2")
    if args.comparison_paths < 4 or args.comparison_paths % 2:
        parser.error("--comparison-paths must be an even integer of at least 4")
    if args.comparison_repeats < 2:
        parser.error("--comparison-repeats must be at least 2")
    run_experiments(args.output_dir, args.seed, args.hedge_paths,
                    args.comparison_paths, args.comparison_repeats, args.quotes_csv)


if __name__ == "__main__":
    main()
