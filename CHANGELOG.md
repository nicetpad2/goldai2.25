# ChangeLog

## [v4.9.42+] - 2025-05-21
- Added global pandas import to ensure backtest simulations access `pd` without errors.
- Added explicit pandas import guard inside `run_backtest_simulation_v34` and `_run_backtest_simulation_v34_full` to avoid `UnboundLocalError` in mocked environments.

## [v4.9.41+] - 2025-05-20
- Added robust equity_tracker history update with numeric guards.
