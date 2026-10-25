"""Discrete-time delta hedging under geometric Brownian motion."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import ndtr

from .black_scholes import OptionType, _option_type, _validate_inputs, bs_greeks, bs_price


def _delta_vector(spot, strike, remaining, rate, volatility, option_type, dividend_yield):
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * remaining
    ) / (volatility * math.sqrt(remaining))
    discount_q = math.exp(-dividend_yield * remaining)
    if option_type == "call":
        return discount_q * ndtr(d1)
    return discount_q * (ndtr(d1) - 1.0)


def delta_hedge_simulation(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 10_000, n_steps: int = 52, option_type: OptionType = "call",
    q: float = 0.0, transaction_cost_rate: float = 0.0, seed: int = 12345,
    stock_drift: float | None = None,
) -> dict[str, Any]:
    """Simulate a self-financing delta hedge for a short European option.

    The hedger receives the Black–Scholes premium, buys option delta in the
    underlying, and finances the remaining position in a cash account accruing
    at r. Proportional transaction costs are charged on traded notional.
    Dividend income from the stock position is credited to cash each interval.
    Terminal error is replicating portfolio value minus option payoff; negative
    error indicates a shortfall.
    """
    kind = _option_type(option_type)
    _validate_inputs(S0, K, T, r, sigma, q)
    if T <= 0.0 or sigma <= 0.0:
        raise ValueError("T and sigma must be positive for a hedging simulation")
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least 2")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not math.isfinite(transaction_cost_rate) or transaction_cost_rate < 0.0:
        raise ValueError("transaction_cost_rate must be finite and non-negative")
    drift = r if stock_drift is None else stock_drift
    if not math.isfinite(drift):
        raise ValueError("stock_drift must be finite")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    increments = (drift - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shocks
    stock_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    stock_paths[:, 0] = S0
    stock_paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))

    initial_price = bs_price(S0, K, T, r, sigma, kind, q)
    initial_delta = bs_greeks(S0, K, T, r, sigma, kind, q)["delta"]
    initial_cost = transaction_cost_rate * abs(initial_delta) * S0
    shares = np.full(n_paths, initial_delta, dtype=float)
    cash = np.full(n_paths, initial_price - initial_delta * S0 - initial_cost, dtype=float)
    cumulative_cost = np.full(n_paths, initial_cost, dtype=float)
    cash_growth = math.exp(r * dt)

    for step in range(1, n_steps):
        previous_spot = stock_paths[:, step - 1]
        dividend_cash = shares * previous_spot * math.expm1(q * dt)
        cash = (cash + dividend_cash) * cash_growth
        current_spot = stock_paths[:, step]
        portfolio_before_rebalance = shares * current_spot + cash
        new_delta = _delta_vector(current_spot, K, T - step * dt, r, sigma, kind, q)
        traded_notional = np.abs(new_delta - shares) * current_spot
        cost = transaction_cost_rate * traded_notional
        cash = portfolio_before_rebalance - new_delta * current_spot - cost
        cumulative_cost += cost
        shares = new_delta

    terminal_portfolio = shares * stock_paths[:, -1] + cash * cash_growth
    terminal_stock = stock_paths[:, -1]
    terminal_payoff = (
        np.maximum(terminal_stock - K, 0.0)
        if kind == "call"
        else np.maximum(K - terminal_stock, 0.0)
    )
    errors = terminal_portfolio - terminal_payoff
    return {
        "initial_option_price": float(initial_price),
        "n_paths": n_paths,
        "n_steps": n_steps,
        "transaction_cost_rate": float(transaction_cost_rate),
        "mean_hedging_error": float(np.mean(errors)),
        "std_hedging_error": float(np.std(errors, ddof=1)),
        "rmse_hedging_error": float(np.sqrt(np.mean(errors**2))),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "error_q05": float(np.quantile(errors, 0.05)),
        "error_median": float(np.quantile(errors, 0.50)),
        "error_q95": float(np.quantile(errors, 0.95)),
        "mean_transaction_cost": float(np.mean(cumulative_cost)),
        "hedging_errors": errors,
        "terminal_payoffs": terminal_payoff,
        "terminal_portfolio_values": terminal_portfolio,
        "terminal_stock_prices": terminal_stock,
    }
