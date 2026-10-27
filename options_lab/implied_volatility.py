"""Implied-volatility inversion and volatility-surface data preparation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
from scipy.optimize import brentq

from .black_scholes import OptionType, _option_type, _validate_inputs, bs_price


def implied_volatility(
    market_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    price_tolerance: float = 1e-10,
    max_volatility: float = 10.0,
) -> float:
    """Infer Black–Scholes volatility matching an observed option price.

    Uses a bracketed Brent solver. A premium equal to intrinsic value maps to
    zero volatility; the theoretical upper price bound has no finite implied
    volatility.
    """
    kind = _option_type(option_type)
    if not math.isfinite(market_price) or market_price < 0.0:
        raise ValueError("market_price must be finite and non-negative")
    if not math.isfinite(price_tolerance) or price_tolerance <= 0.0:
        raise ValueError("price_tolerance must be finite and positive")
    if not math.isfinite(max_volatility) or max_volatility <= 1e-8:
        raise ValueError("max_volatility must exceed 1e-8")
    _validate_inputs(S0, K, T, r, 0.0, q)
    if T <= 0.0:
        raise ValueError("T must be positive for implied volatility")

    discounted_spot = S0 * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    if kind == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike

    if market_price < lower - price_tolerance or market_price > upper + price_tolerance:
        raise ValueError(
            f"market_price violates no-arbitrage bounds [{lower:.12g}, {upper:.12g}]"
        )
    if abs(market_price - lower) <= price_tolerance:
        return 0.0
    if market_price >= upper - price_tolerance:
        raise ValueError("market_price is at the upper bound; no finite implied volatility exists")

    objective = lambda volatility: bs_price(S0, K, T, r, volatility, kind, q) - market_price
    high = min(0.5, max_volatility)
    while objective(high) < 0.0 and high < max_volatility:
        high = min(2.0 * high, max_volatility)
    if objective(high) < 0.0:
        raise ValueError("could not bracket implied volatility below max_volatility")
    return float(brentq(objective, 1e-8, high, xtol=1e-12, rtol=1e-12, maxiter=200))


def implied_volatility_surface(
    quotes: pd.DataFrame | Iterable[Mapping[str, object]],
    S0: float,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """Add implied volatilities to quotes with strike, maturity and market price.

    Each quote may optionally specify spot, rate, dividend_yield and option_type;
    otherwise the function-level defaults are used. Invalid rows raise an
    informative exception rather than silently entering the surface.
    """
    frame = quotes.copy() if isinstance(quotes, pd.DataFrame) else pd.DataFrame(list(quotes))
    required = {"strike", "maturity", "market_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"quotes missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("quotes must contain at least one row")

    inferred: list[float] = []
    normalized_types: list[str] = []
    for row_number, row in frame.iterrows():
        try:
            row_spot = float(row.get("spot", S0))
            row_rate = float(row.get("rate", r))
            row_q = float(row.get("dividend_yield", q))
            strike = float(row["strike"])
            maturity = float(row["maturity"])
            market_price = float(row["market_price"])
            kind = _option_type(str(row.get("option_type", "call")))
            iv = implied_volatility(
                market_price, row_spot, strike, maturity, row_rate, kind, row_q
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid option quote at row {row_number}: {exc}") from exc
        inferred.append(iv)
        normalized_types.append(kind)

    output = frame.copy()
    output["option_type"] = normalized_types
    output["implied_volatility"] = inferred
    return output


def synthetic_option_chain(
    S0: float = 100.0,
    r: float = 0.03,
    q: float = 0.0,
    maturities: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    moneyness: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> pd.DataFrame:
    """Create deterministic illustrative quotes; these are not market data."""
    rows: list[dict[str, float | str]] = []
    for maturity in maturities:
        for ratio in moneyness:
            strike = S0 * float(ratio)
            log_moneyness = math.log(ratio)
            volatility = (
                0.22
                - 0.10 * log_moneyness
                + 0.35 * log_moneyness**2
                + 0.015 * math.sqrt(maturity)
            )
            premium = bs_price(S0, strike, float(maturity), r, volatility, "call", q)
            rows.append(
                {
                    "strike": strike,
                    "maturity": float(maturity),
                    "market_price": premium,
                    "option_type": "call",
                    "synthetic_volatility": volatility,
                }
            )
    return pd.DataFrame(rows)
