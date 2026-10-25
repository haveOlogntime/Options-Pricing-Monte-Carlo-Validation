import math

import pandas as pd
import pytest

from options_lab import bs_price, implied_volatility, implied_volatility_surface, synthetic_option_chain


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_implied_volatility_recovers_input_volatility(option_type, q):
    vol = 0.31
    price = bs_price(105, 100, 0.8, 0.04, vol, option_type, q)
    inferred = implied_volatility(price, 105, 100, 0.8, 0.04, option_type, q)
    assert inferred == pytest.approx(vol, abs=1e-8)


def test_intrinsic_value_maps_to_zero_volatility():
    spot, strike, maturity, rate = 110.0, 100.0, 1.0, 0.03
    intrinsic_limit = max(spot - strike * math.exp(-rate * maturity), 0)
    assert implied_volatility(intrinsic_limit, spot, strike, maturity, rate) == 0.0


def test_invalid_prices_and_expiry_are_rejected():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(150, 100, 100, 1, 0.03)
    with pytest.raises(ValueError, match="positive"):
        implied_volatility(2, 100, 100, 0, 0.03)


def test_surface_recovers_synthetic_volatilities():
    quotes = synthetic_option_chain()
    surface = implied_volatility_surface(quotes, S0=100, r=0.03)
    assert len(surface) == len(quotes)
    assert surface["implied_volatility"].to_numpy() == pytest.approx(
        surface["synthetic_volatility"].to_numpy(), abs=1e-7
    )


def test_surface_requires_quote_columns_and_nonempty_input():
    with pytest.raises(ValueError, match="missing required"):
        implied_volatility_surface(pd.DataFrame({"strike": [100]}), S0=100, r=0.03)
    with pytest.raises(ValueError, match="at least one row"):
        implied_volatility_surface(pd.DataFrame(columns=["strike", "maturity", "market_price"]), S0=100, r=0.03)
