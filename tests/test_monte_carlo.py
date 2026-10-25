import pytest

from options_lab import bs_price, mc_price

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25


def test_all_estimators_are_reproducible_and_near_analytic_value():
    target = bs_price(S0, K, T, r, sigma)
    for method in ("plain", "antithetic", "control_variate"):
        first = mc_price(S0, K, T, r, sigma, 100_000, "call", method, seed=7)
        second = mc_price(S0, K, T, r, sigma, 100_000, "call", method, seed=7)
        assert first == second
        assert abs(first["price"] - target) < 4 * first["standard_error"]
        assert first["ci95_low"] < first["price"] < first["ci95_high"]


def test_antithetic_standard_error_uses_independent_pair_averages():
    result = mc_price(S0, K, T, r, sigma, 20_000, method="antithetic", seed=9)
    assert result["n_paths"] == 20_000
    assert result["n_independent_samples"] == 10_000


def test_antithetic_rejects_odd_or_too_small_path_budgets():
    with pytest.raises(ValueError, match="even"):
        mc_price(S0, K, T, r, sigma, 101, method="antithetic")
    with pytest.raises(ValueError, match="at least 4"):
        mc_price(S0, K, T, r, sigma, 2, method="antithetic")


def test_mc_rejects_invalid_path_count_and_method():
    with pytest.raises(ValueError, match="n_paths"):
        mc_price(S0, K, T, r, sigma, 1)
    with pytest.raises(ValueError, match="method"):
        mc_price(S0, K, T, r, sigma, 10, method="importance")  # type: ignore[arg-type]


def test_control_variate_supports_dividend_yield():
    result = mc_price(S0, K, T, r, sigma, 50_000, method="control_variate", q=0.02, seed=4)
    target = bs_price(S0, K, T, r, sigma, q=0.02)
    assert abs(result["price"] - target) < 4 * result["standard_error"]
    assert "control_beta" in result
