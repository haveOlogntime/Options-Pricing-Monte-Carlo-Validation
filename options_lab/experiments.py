"""Reproducible experiments and plots for the options research lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .black_scholes import bs_greeks, bs_price
from .hedging import delta_hedge_simulation
from .implied_volatility import implied_volatility_surface, synthetic_option_chain
from .monte_carlo import mc_price


def run_experiments(output_dir: str | Path = "results", seed: int = 20261025, hedge_paths: int = 5_000) -> dict[str, pd.DataFrame]:
    """Run pricing, implied-volatility and delta-hedging experiments."""
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
        result = mc_price(
            S0, K, T, r, sigma, n_paths=100_000,
            option_type="call", method=method, seed=seed, q=q,
        )
        result["bs_price"] = analytic_price
        result["absolute_error"] = abs(float(result["price"]) - analytic_price)
        comparison_rows.append(result)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "monte_carlo_comparison.csv", index=False)
    print("\nMonte Carlo comparison (100,000 simulated paths):")
    print(comparison[
        ["method", "price", "standard_error", "ci95_low", "ci95_high", "absolute_error"]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    convergence_rows = []
    for n_paths in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000):
        for method in ("plain", "antithetic", "control_variate"):
            result = mc_price(
                S0, K, T, r, sigma, n_paths=n_paths,
                option_type="call", method=method, seed=seed + n_paths, q=q,
            )
            convergence_rows.append({
                "method": method,
                "n_paths": n_paths,
                "price": float(result["price"]),
                "standard_error": float(result["standard_error"]),
                "absolute_error": abs(float(result["price"]) - analytic_price),
                "ci95_low": float(result["ci95_low"]),
                "ci95_high": float(result["ci95_high"]),
            })
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(output / "mc_convergence.csv", index=False)
    for column, title, filename in (
        ("absolute_error", "Monte Carlo pricing error (one seed per budget)", "mc_absolute_error.svg"),
        ("standard_error", "Estimated standard error by path count", "mc_standard_error.svg"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for method, group in convergence.groupby("method", sort=False):
            ax.plot(group["n_paths"], group[column], marker="o", linewidth=1.8, label=method.replace("_", " "))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Simulated terminal stock values")
        ax.set_ylabel(column.replace("_", " ").capitalize())
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)

    quotes = synthetic_option_chain(S0=S0, r=r, q=q)
    surface = implied_volatility_surface(quotes, S0=S0, r=r, q=q)
    surface.to_csv(output / "implied_volatility_surface.csv", index=False)
    recovery_error = float(np.max(np.abs(surface["synthetic_volatility"] - surface["implied_volatility"])))
    print(f"\nMaximum synthetic-volatility recovery error: {recovery_error:.3e}")
    matrix = surface.pivot(index="maturity", columns="strike", values="implied_volatility")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    image = ax.imshow(
        matrix.to_numpy() * 100.0, aspect="auto", origin="lower", interpolation="nearest",
        extent=(
            matrix.columns.min() - (matrix.columns.max() - matrix.columns.min()) / (2 * max(len(matrix.columns) - 1, 1)),
            matrix.columns.max() + (matrix.columns.max() - matrix.columns.min()) / (2 * max(len(matrix.columns) - 1, 1)),
            matrix.index.min() - (matrix.index.max() - matrix.index.min()) / (2 * max(len(matrix.index) - 1, 1)),
            matrix.index.max() + (matrix.index.max() - matrix.index.min()) / (2 * max(len(matrix.index) - 1, 1)),
        ),
        cmap="viridis",
    )
    ax.set_xticks(matrix.columns.to_numpy())
    ax.set_yticks(matrix.index.to_numpy())
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (years)")
    ax.set_title("Illustrative implied-volatility surface (synthetic quotes)")
    fig.colorbar(image, ax=ax).set_label("Implied volatility (%)")
    fig.tight_layout()
    fig.savefig(output / "implied_volatility_surface.svg")
    plt.close(fig)

    hedge_rows = []
    for n_steps in (12, 52, 252):
        for cost_rate in (0.0, 0.001):
            result = delta_hedge_simulation(
                S0, K, T, r, sigma, n_paths=hedge_paths, n_steps=n_steps,
                option_type="call", q=q, transaction_cost_rate=cost_rate, seed=seed + n_steps,
            )
            hedge_rows.append({key: result[key] for key in (
                "n_paths", "n_steps", "initial_option_price", "transaction_cost_rate",
                "mean_hedging_error", "std_hedging_error", "rmse_hedging_error",
                "mean_absolute_error", "error_q05", "error_median", "error_q95",
                "mean_transaction_cost",
            )})
    hedging = pd.DataFrame(hedge_rows)
    hedging.to_csv(output / "delta_hedging_summary.csv", index=False)
    print("\nDelta-hedging summary:")
    print(hedging[
        ["n_steps", "transaction_cost_rate", "mean_hedging_error", "rmse_hedging_error", "mean_transaction_cost"]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for cost_rate, group in hedging.groupby("transaction_cost_rate", sort=True):
        ax.plot(group["n_steps"], group["rmse_hedging_error"], marker="o", linewidth=1.8,
                label=f"Transaction costs: {cost_rate * 10_000:.0f} bp")
    ax.set_xlabel("Rehedging intervals over the option's life")
    ax.set_ylabel("Root-mean-square hedging error")
    ax.set_title("Discrete delta hedging: error vs rehedging frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "delta_hedging_rmse.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    labels = [f"{int(row.n_steps)} steps, {row.transaction_cost_rate * 10_000:.0f} bp" for row in hedging.itertuples()]
    ax.errorbar(
        np.arange(len(hedging)), hedging["error_median"],
        yerr=[hedging["error_median"] - hedging["error_q05"], hedging["error_q95"] - hedging["error_median"]],
        fmt="o", capsize=4, linewidth=1.4,
    )
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.65)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Terminal hedge portfolio minus option payoff")
    ax.set_title("Terminal hedging-error range (5th–95th percentile)")
    fig.tight_layout()
    fig.savefig(output / "delta_hedging_error_distribution.svg")
    plt.close(fig)

    print(f"\nResults written to {output.resolve()}")
    print("The option-chain surface is synthetic and demonstrates the inversion workflow.")
    print("The Monte Carlo convergence plot uses one seed per path count; error need not fall monotonically.")
    return {
        "monte_carlo_comparison": comparison,
        "monte_carlo_convergence": convergence,
        "implied_volatility_surface": surface,
        "delta_hedging_summary": hedging,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pricing, volatility and hedging experiments.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV tables and SVG figures")
    parser.add_argument("--seed", type=int, default=20261025, help="Base random seed")
    parser.add_argument("--hedge-paths", type=int, default=5_000, help="Number of paths per hedging experiment")
    args = parser.parse_args()
    if args.hedge_paths < 2:
        parser.error("--hedge-paths must be at least 2")
    run_experiments(args.output_dir, args.seed, args.hedge_paths)


if __name__ == "__main__":
    main()
