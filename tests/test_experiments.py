import pandas as pd
import pytest
from options_lab import synthetic_option_chain
from options_lab.experiments import repeated_seed_comparison, run_experiments


def test_synthetic_quotes_cover_strikes_and_expiries():
    quotes = synthetic_option_chain()
    assert "synthetic_volatility" in quotes.columns
    assert quotes["strike"].nunique() >= 5
    assert quotes["maturity"].nunique() >= 4


def test_repeated_seed_comparison_summarizes_estimator_variability():
    summary, runs = repeated_seed_comparison(100, 100, 1, 0.03, 0.25, n_paths=2000, n_repeats=5, seed=100)
    assert len(summary) == 3
    assert len(runs) == 15
    assert summary["empirical_variance"].ge(0).all()
    assert summary["nominal_95pct_interval_coverage"].between(0, 1).all()
    assert summary.loc[summary["method"] == "plain", "variance_reduction_vs_plain"].iloc[0] == pytest.approx(0)


def test_experiment_runner_accepts_quote_csv_and_writes_outputs(tmp_path):
    quote_path = tmp_path / "quotes.csv"
    synthetic_option_chain().drop(columns=["synthetic_volatility"]).to_csv(quote_path, index=False)
    output = tmp_path / "results"
    result = run_experiments(
        output_dir=output, seed=2, hedge_paths=30,
        comparison_paths=100, comparison_repeats=2, quotes_csv=quote_path,
    )
    expected = {
        "monte_carlo_comparison.csv", "mc_convergence.csv",
        "monte_carlo_repeated_seed_summary.csv", "monte_carlo_repeated_seed_runs.csv",
        "implied_volatility_surface.csv", "delta_hedging_summary.csv",
        "mc_absolute_error.svg", "mc_standard_error.svg", "mc_repeated_seed_variability.svg",
        "implied_volatility_surface.svg", "delta_hedging_rmse.svg", "delta_hedging_error_distribution.svg",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    surface = pd.read_csv(output / "implied_volatility_surface.csv")
    assert len(surface) == 20
    assert "implied_volatility" in surface.columns
    assert set(result) == {
        "monte_carlo_comparison", "monte_carlo_convergence", "monte_carlo_repeated_seed_summary",
        "monte_carlo_repeated_seed_runs", "implied_volatility_surface", "delta_hedging_summary",
    }
