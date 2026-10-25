"""Black–Scholes prices and sensitivities for European options."""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]


def _option_type(value: str) -> OptionType:
    normalized = value.lower()
    if normalized not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    return normalized  # type: ignore[return-value]


def _validate_inputs(
    spot: float, strike: float, maturity: float, rate: float,
    volatility: float, dividend_yield: float = 0.0,
) -> None:
    values = (spot, strike, maturity, rate, volatility, dividend_yield)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all model inputs must be finite")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0.0:
        raise ValueError("maturity cannot be negative")
    if volatility < 0.0:
        raise ValueError("volatility cannot be negative")


def _d1_d2(
    spot: float, strike: float, maturity: float, rate: float,
    volatility: float, dividend_yield: float = 0.0,
) -> tuple[float, float]:
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if maturity <= 0.0 or volatility <= 0.0:
        raise ValueError("d1 and d2 require positive maturity and volatility")
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def bs_price(
    S0: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "call", q: float = 0.0,
) -> float:
    """Return the Black–Scholes value of a European call or put.

    S0 and K are spot and strike, T is time in years, r is the
    continuously compounded interest rate, sigma is annualized volatility,
    and q is a continuous dividend yield. Deterministic limiting cases are
    handled directly when T or sigma is zero.
    """
    kind = _option_type(option_type)
    _validate_inputs(S0, K, T, r, sigma, q)
    if T == 0.0:
        return float(max(S0 - K, 0.0) if kind == "call" else max(K - S0, 0.0))

    discounted_spot = S0 * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    if sigma == 0.0:
        difference = discounted_spot - discounted_strike
        return float(max(difference, 0.0) if kind == "call" else max(-difference, 0.0))

    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    if kind == "call":
        value = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        value = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return float(value)


def bs_greeks(
    S0: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "call", q: float = 0.0,
) -> dict[str, float]:
    """Return price and analytic Greeks for a European option.

    Vega is per 1.00 absolute change in volatility; theta is per year and
    represents calendar-time decay; rho is per 1.00 absolute rate change.
    Greeks are undefined at expiry or at zero volatility in this implementation.
    """
    kind = _option_type(option_type)
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    root_t = math.sqrt(T)
    discount_r = math.exp(-r * T)
    discount_q = math.exp(-q * T)
    density = norm.pdf(d1)

    gamma = discount_q * density / (S0 * sigma * root_t)
    vega = S0 * discount_q * density * root_t
    common_theta = -S0 * discount_q * density * sigma / (2.0 * root_t)
    if kind == "call":
        delta = discount_q * norm.cdf(d1)
        theta = common_theta - r * K * discount_r * norm.cdf(d2) + q * S0 * discount_q * norm.cdf(d1)
        rho = K * T * discount_r * norm.cdf(d2)
    else:
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta = common_theta + r * K * discount_r * norm.cdf(-d2) - q * S0 * discount_q * norm.cdf(-d1)
        rho = -K * T * discount_r * norm.cdf(-d2)

    return {
        "price": bs_price(S0, K, T, r, sigma, kind, q),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
