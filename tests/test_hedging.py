import numpy as np
import pytest

from options_lab import delta_hedge_simulation

ARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.25)


def test_hedging_simulation_is_reproducible_and_returns_diagnostics():
    first = delta_hedge_simulation(**ARGS, n_paths=200, n_steps=26, seed=11)
    second = delta_hedge_simulation(**ARGS, n_paths=200, n_steps=26, seed=11)
    for key in ("mean_hedging_error", "std_hedging_error", "rmse_hedging_error", "mean_transaction_cost"):
        assert first[key] == second[key]
    assert np.array_equal(first["hedging_errors"], second["hedging_errors"])
    assert first["hedging_errors"].shape == (200,)
    assert first["rmse_hedging_error"] >= 0


def test_transaction_costs_cannot_improve_each_pathwise_replication_value():
    no_cost = delta_hedge_simulation(**ARGS, n_paths=200, n_steps=26, seed=13)
    with_cost = delta_hedge_simulation(
        **ARGS, n_paths=200, n_steps=26, transaction_cost_rate=0.001, seed=13
    )
    assert np.all(with_cost["hedging_errors"] <= no_cost["hedging_errors"] + 1e-12)
    assert with_cost["mean_transaction_cost"] > 0


def test_put_hedging_and_invalid_parameters():
    result = delta_hedge_simulation(**ARGS, n_paths=100, n_steps=12, option_type="put", seed=3)
    assert np.isfinite(result["rmse_hedging_error"])
    with pytest.raises(ValueError, match="n_steps"):
        delta_hedge_simulation(**ARGS, n_steps=0)
    with pytest.raises(ValueError, match="transaction_cost_rate"):
        delta_hedge_simulation(**ARGS, transaction_cost_rate=-0.1)
