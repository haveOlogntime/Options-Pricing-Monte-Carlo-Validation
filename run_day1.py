from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_lab import bs_greeks, bs_price, mc_price

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# start with one at-the-money call so the methods are easy to compare
S0 = 100.0
K = 100.0
T = 1.0
r = 0.03
sigma = 0.25
true_price = bs_price(S0, K, T, r, sigma, "call")

print("baseline: european call, no dividends")
print(f"S0={S0:g}, K={K:g}, T={T:g}, r={r:.2%}, sigma={sigma:.2%}")
print(f"black-scholes price: {true_price:.6f}\n")
print("call greeks (vega per 1.00 volatility, theta per year):")
for name, value in bs_greeks(S0, K, T, r, sigma, "call").items():
    print(f"  {name:>8}: {value:.6f}")

# compare all three methods using the same path budget
rows = []
for method in ("plain", "antithetic", "control_variate"):
    res = mc_price(
        S0, K, T, r, sigma,
        n_paths=100_000,
        option_type="call",
        method=method,
        seed=20261025,
    )
    res["bs_price"] = true_price
    res["absolute_error"] = abs(float(res["price"]) - true_price)
    rows.append(res)

baseline = pd.DataFrame(rows)
baseline.to_csv(RESULTS / "monte_carlo_comparison.csv", index=False)
print("\nmonte carlo comparison at 100,000 paths:")
print(baseline[[
    "method", "n_paths", "price", "standard_error",
    "ci95_low", "ci95_high", "absolute_error",
]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# one seed gives a quick picture, but it isn't enough to rank the methods
path_counts = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
rows = []
for n in path_counts:
    for method in ("plain", "antithetic", "control_variate"):
        res = mc_price(
            S0, K, T, r, sigma,
            n_paths=n,
            option_type="call",
            method=method,
            seed=1000 + n,
        )
        rows.append({
            "method": method,
            "n_paths": n,
            "price": float(res["price"]),
            "standard_error": float(res["standard_error"]),
            "absolute_error": abs(float(res["price"]) - true_price),
            "ci95_low": float(res["ci95_low"]),
            "ci95_high": float(res["ci95_high"]),
        })

results = pd.DataFrame(rows)
results.to_csv(RESULTS / "mc_convergence.csv", index=False)

for col, title, filename in [
    ("absolute_error", "Monte Carlo pricing error (one seed)", "mc_absolute_error.png"),
    ("standard_error", "Estimated standard error vs path count", "mc_standard_error.png"),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in results.groupby("method"):
        ax.plot(group["n_paths"], group[col], marker="o", label=method.replace("_", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total simulated terminal stock values")
    ax.set_ylabel(col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=160)
    plt.close(fig)

print(f"\nresults saved in {RESULTS}")
print("the convergence plot uses one seed per path count; we'll do a repeated-seed test on day 3")
