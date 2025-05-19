# ChangeLog

## [v4.9.42+] - 2025-05-21
- Added global pandas import to ensure backtest simulations access `pd` without errors.
- Added explicit pandas import guard inside `run_backtest_simulation_v34` and `_run_backtest_simulation_v34_full` to avoid `UnboundLocalError` in mocked environments.

## [v4.9.43+] - 2025-05-22
- Fixed argument order in `run_backtest_simulation_v34` and ensured side is parsed from kwargs.
- Guarded `equity_tracker['current_equity']` comparison with numeric check and improved history update imports.

## [v4.9.44+] - 2025-05-22
- `run_backtest_simulation_v34` now returns a dictionary by default with keys matching production expectations.
- Added optional `return_tuple=True` argument for legacy compatibility.
- Updated documentation and version constants for enterprise QA.

## [v4.9.41+] - 2025-05-20
- Added robust equity_tracker history update with numeric guards.
