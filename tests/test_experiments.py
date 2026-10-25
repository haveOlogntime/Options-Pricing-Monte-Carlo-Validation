from options_lab import synthetic_option_chain


def test_synthetic_quotes_are_labelled_and_cover_strikes_and_expiries():
    quotes = synthetic_option_chain()
    assert "synthetic_volatility" in quotes.columns
    assert quotes["strike"].nunique() >= 5
    assert quotes["maturity"].nunique() >= 4
