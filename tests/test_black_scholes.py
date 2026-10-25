import math

import pytest

from options_lab import bs_greeks, bs_price

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25


def test_known_at_the_money_call_price():
    assert bs_price(S0, K, T, r, sigma, "call") == pytest.approx(11.3484768, abs=1e-6)


def test_put_call_parity_with_dividend_yield():
    q = 0.015
    call = bs_price(S0, K, T, r, sigma, "call", q)
    put = bs_price(S0, K, T, r, sigma, "put", q)
    assert call - put == pytest.approx(S0 * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_zero_maturity_returns_intrinsic_value():
    assert bs_price(110, 100, 0, r, sigma, "call") == pytest.approx(10)
    assert bs_price(90, 100, 0, r, sigma, "put") == pytest.approx(10)


def test_zero_volatility_uses_deterministic_limit():
    expected = max(S0 - K * math.exp(-r * T), 0)
    assert bs_price(S0, K, T, r, 0, "call") == pytest.approx(expected)


def test_invalid_option_type_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="option_type"):
        bs_price(S0, K, T, r, sigma, "straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        bs_price(0, K, T, r, sigma)
    with pytest.raises(ValueError, match="finite"):
        bs_price(S0, K, T, r, float("nan"))


def test_greeks_match_finite_difference_checks():
    h = 1e-3
    greeks = bs_greeks(S0, K, T, r, sigma, "call")
    delta_fd = (bs_price(S0 + h, K, T, r, sigma) - bs_price(S0 - h, K, T, r, sigma)) / (2 * h)
    gamma_fd = (bs_price(S0 + h, K, T, r, sigma) - 2 * bs_price(S0, K, T, r, sigma) + bs_price(S0 - h, K, T, r, sigma)) / h**2
    assert greeks["delta"] == pytest.approx(delta_fd, rel=1e-6)
    assert greeks["gamma"] == pytest.approx(gamma_fd, rel=1e-4)


def test_put_greeks_have_correct_delta_sign_and_theta_matches_time_bump():
    greeks = bs_greeks(S0, K, T, r, sigma, "put")
    assert -1.0 < greeks["delta"] < 0.0
    dt = 1e-5
    calendar_theta_fd = (bs_price(S0, K, T - dt, r, sigma, "put") - bs_price(S0, K, T + dt, r, sigma, "put")) / (2 * dt)
    assert greeks["theta"] == pytest.approx(calendar_theta_fd, rel=1e-5)


def test_put_theta_with_dividends_matches_time_bump():
    q = 0.025
    greeks = bs_greeks(S0, K, T, r, sigma, "put", q)
    dt = 1e-5
    calendar_theta_fd = (
        bs_price(S0, K, T - dt, r, sigma, "put", q)
        - bs_price(S0, K, T + dt, r, sigma, "put", q)
    ) / (2 * dt)
    assert greeks["theta"] == pytest.approx(calendar_theta_fd, rel=1e-5)
