import math

import pytest

from options_lab import bs_greeks, bs_price, mc_price

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25


def test_put_call_parity():
    call = bs_price(S0, K, T, r, sigma, "call")
    put = bs_price(S0, K, T, r, sigma, "put")
    assert call - put == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-10)


def test_call_greeks_have_expected_signs():
    # for a regular european call, delta, gamma and vega should be positive
    greeks = bs_greeks(S0, K, T, r, sigma, "call")
    assert greeks["price"] > 0
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


def test_delta_matches_a_small_price_bump():
    h = 1e-3
    bumped = (
        bs_price(S0 + h, K, T, r, sigma, "call")
        - bs_price(S0 - h, K, T, r, sigma, "call")
    ) / (2 * h)
    assert bs_greeks(S0, K, T, r, sigma, "call")["delta"] == pytest.approx(bumped, rel=1e-6)


def test_monte_carlo_result_is_repeatable():
    a = mc_price(S0, K, T, r, sigma, 20_000, "call", "antithetic", seed=7)
    b = mc_price(S0, K, T, r, sigma, 20_000, "call", "antithetic", seed=7)
    assert a == b
    assert a["ci95_low"] < a["price"] < a["ci95_high"]


def test_antithetic_needs_even_number_of_paths():
    with pytest.raises(ValueError):
        mc_price(S0, K, T, r, sigma, 101, "call", "antithetic", seed=1)
