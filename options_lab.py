from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _check_inputs(S0: float, K: float, T: float, r: float, sigma: float) -> None:
    # catch bad inputs here before doing any maths
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not all(math.isfinite(x) for x in (S0, K, T, r, sigma)):
        raise ValueError("all inputs must be finite")


def _d1_d2(S0: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    _check_inputs(S0, K, T, r, sigma)
    d1 = (math.log(S0 / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    # price a european option with the black-scholes formula
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    option_type = option_type.lower()

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    if option_type == "put":
        return float(K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> dict[str, float]:
    # vega is per 1.00 change in volatility, theta is per year
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    discount = math.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = pdf / (S0 * sigma * math.sqrt(T))
    vega = S0 * pdf * math.sqrt(T)
    theta = -S0 * pdf * sigma / (2 * math.sqrt(T))

    if option_type == "call":
        delta = norm.cdf(d1)
        theta -= r * K * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta += r * K * discount * norm.cdf(-d2)

    return {
        "price": bs_price(S0, K, T, r, sigma, option_type),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


def _payoffs(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    z: np.ndarray,
    option_type: OptionType,
) -> tuple[np.ndarray, np.ndarray]:
    # generate terminal stock prices under the risk-neutral model
    ST = S0 * np.exp((r - sigma * sigma / 2) * T + sigma * math.sqrt(T) * z)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    discount = math.exp(-r * T)
    return discount * payoff, discount * ST


def _summary(
    samples: np.ndarray,
    method: str,
    n_paths: int,
    beta: float | None = None,
) -> dict[str, float | str | int]:
    # for antithetic sampling each sample is already the average of a pair
    price = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))

    out: dict[str, float | str | int] = {
        "method": method,
        "n_paths": n_paths,
        "price": price,
        "standard_error": se,
        "ci95_low": price - 1.96 * se,
        "ci95_high": price + 1.96 * se,
    }
    if beta is not None:
        out["control_beta"] = beta
    return out


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    option_type: OptionType = "call",
    method: Literal["plain", "antithetic", "control_variate"] = "plain",
    seed: int = 12345,
) -> dict[str, float | str | int]:
    # the seed keeps runs repeatable
    _check_inputs(S0, K, T, r, sigma)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    rng = np.random.default_rng(seed)

    if method == "plain":
        z = rng.standard_normal(n_paths)
        x, _ = _payoffs(S0, K, T, r, sigma, z, option_type)
        return _summary(x, "plain", n_paths)

    if method == "antithetic":
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic variates")
        if n_paths < 4:
            raise ValueError(
                "n_paths must be at least 4 to estimate the standard error "
                "from antithetic pairs"
            )

        z = rng.standard_normal(n_paths // 2)
        x1, _ = _payoffs(S0, K, T, r, sigma, z, option_type)
        x2, _ = _payoffs(S0, K, T, r, sigma, -z, option_type)
        pair_avg = (x1 + x2) / 2
        return _summary(pair_avg, "antithetic", n_paths)

    if method == "control_variate":
        z = rng.standard_normal(n_paths)
        x, y = _payoffs(S0, K, T, r, sigma, z, option_type)
        var_y = float(np.var(y, ddof=1))
        if var_y == 0:
            raise ValueError("control variate has zero sample variance")

        # the discounted stock price has expected value S0 here
        beta = float(np.cov(x, y, ddof=1)[0, 1] / var_y)
        adjusted = x - beta * (y - S0)
        return _summary(adjusted, "control_variate", n_paths, beta)

    raise ValueError("method must be 'plain', 'antithetic', or 'control_variate'")
