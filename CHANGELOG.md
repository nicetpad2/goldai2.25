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

## [v4.9.45+] - 2025-05-23
- Fixed import errors for optional libraries and deferred GPU setup to main execution block.
- Enhanced logging for library installation and Colab detection.
- Updated version constants and documentation references.

## [v4.9.46+] - 2025-05-23
- Initialized all optional ML library flags inside `import_core_libraries` to avoid `UnboundLocalError` across inference and backtest modules.
- Updated documentation and version constants.

## [v4.9.41+] - 2025-05-20
- Added robust equity_tracker history update with numeric guards.
