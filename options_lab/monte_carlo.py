"""Monte Carlo estimators for European option values."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .black_scholes import OptionType, _option_type, _validate_inputs

MCMethod = Literal["plain", "antithetic", "control_variate"]


def _terminal_values(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    z: np.ndarray,
) -> np.ndarray:
    drift = (rate - dividend_yield - 0.5 * volatility**2) * maturity
    diffusion = volatility * math.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


def _discounted_payoff(
    terminal_spot: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType,
) -> np.ndarray:
    if option_type == "call":
        payoff = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal_spot, 0.0)
    return math.exp(-rate * maturity) * payoff


def _summary(
    samples: np.ndarray,
    method: MCMethod,
    n_paths: int,
    control_beta: float | None = None,
) -> dict[str, float | str | int]:
    if samples.size < 2:
        raise ValueError("at least two independent samples are needed for a standard error")
    estimate = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
    result: dict[str, float | str | int] = {
        "method": method,
        "n_paths": n_paths,
        "n_independent_samples": int(samples.size),
        "price": estimate,
        "standard_error": standard_error,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
    }
    if control_beta is not None:
        result["control_beta"] = control_beta
    return result


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    option_type: OptionType = "call",
    method: MCMethod = "plain",
    seed: int = 12345,
    q: float = 0.0,
) -> dict[str, float | str | int]:
    """Estimate a European option value and return uncertainty diagnostics.

    n_paths counts simulated terminal underlying values. For antithetic
    sampling, the standard error is based on independent pair averages, not
    individual correlated payoffs. The control variate is discounted terminal
    stock price, with known expectation S0 * exp(-q*T).
    """
    kind = _option_type(option_type)
    _validate_inputs(S0, K, T, r, sigma, q)
    if T <= 0.0:
        raise ValueError("T must be positive for Monte Carlo pricing")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive for Monte Carlo pricing")
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least 2")
    if method not in ("plain", "antithetic", "control_variate"):
        raise ValueError("method must be 'plain', 'antithetic', or 'control_variate'")

    rng = np.random.default_rng(seed)
    discount_r = math.exp(-r * T)
    expected_discounted_spot = S0 * math.exp(-q * T)

    if method == "plain":
        z = rng.standard_normal(n_paths)
        terminal = _terminal_values(S0, T, r, sigma, q, z)
        samples = _discounted_payoff(terminal, K, T, r, kind)
        return _summary(samples, method, n_paths)

    if method == "antithetic":
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic variates")
        if n_paths < 4:
            raise ValueError("n_paths must be at least 4 to estimate the standard error from pairs")
        z = rng.standard_normal(n_paths // 2)
        terminal_pos = _terminal_values(S0, T, r, sigma, q, z)
        terminal_neg = _terminal_values(S0, T, r, sigma, q, -z)
        payoff_pos = _discounted_payoff(terminal_pos, K, T, r, kind)
        payoff_neg = _discounted_payoff(terminal_neg, K, T, r, kind)
        pair_averages = 0.5 * (payoff_pos + payoff_neg)
        return _summary(pair_averages, method, n_paths)

    z = rng.standard_normal(n_paths)
    terminal = _terminal_values(S0, T, r, sigma, q, z)
    payoff = _discounted_payoff(terminal, K, T, r, kind)
    discounted_terminal_spot = discount_r * terminal
    variance_control = float(np.var(discounted_terminal_spot, ddof=1))
    if variance_control <= 0.0:
        raise ValueError("control variate has zero sample variance")
    covariance = float(np.cov(payoff, discounted_terminal_spot, ddof=1)[0, 1])
    beta = covariance / variance_control
    adjusted = payoff - beta * (discounted_terminal_spot - expected_discounted_spot)
    return _summary(adjusted, method, n_paths, beta)
