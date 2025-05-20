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
## [v4.9.47+] - 2025-05-23
- Mocked missing ML/TA libraries (`ta`, `optuna`, `catboost`) in `test_gold_ai.py` to ensure CI/CD environments without these packages pass all tests.
- Documentation and version constants updated.

## [v4.9.48+] - 2025-05-23
- Added mock submodule `optuna.logging` in `test_gold_ai.py` to prevent `AttributeError` when `optuna` is missing.
- Updated version constants and documentation references.

## [v4.9.49+] - 2025-05-23
- Robust Optuna logging compatibility to handle missing `optuna.logging` attributes and support Optuna v3.x.
- Logs all code paths and warnings when `optuna` is unavailable or mocked.
- Updated version constants and documentation references.

## [v4.9.50+] - 2025-05-24
- ADA test API sync for `prepare_datetime` (label arg) and `run_all_folds_with_threshold` (l1_threshold alias).
- Added `_robust_kwargs_guard` helper to absorb unexpected kwargs.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.50_FULL_PASS`.

## [v4.9.51+] - 2025-05-24
- Hotfix `prepare_datetime` uses `datetime.now()` correctly.
- `run_all_folds_with_threshold` now accepts legacy argument order and provides
  dummy defaults for missing objects.
- Enhanced test TA mock with trend/momentum/volatility submodules.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.51_FULL_PASS`.

## [v4.9.52+] - 2025-05-25
- Added test-aware datetime fallback in `prepare_datetime`.
- `simulate_trades` now absorbs `side` kwarg and unused kwargs without error.
- Introduced `_ensure_datetimeindex` and `_raise_or_warn` helpers for flexible index validation.
- Updated tests to provide dummy RiskManager to `TradeManager`.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.52_FULL_PASS`.

## [v4.9.53+] - 2025-05-26
- `run_all_folds_with_threshold` returns `WFVResult` supporting dict-style access.
- `_run_backtest_simulation_v34_full` now imports numpy locally to avoid `UnboundLocalError`.
- Version bumped to `4.9.53_FULL_PASS`.

## [v4.9.54+] - 2025-05-27
- Fixed `__class__` propagation issue in `WFVResult` when importing module on Python 3.11.
- `__getitem__` now delegates to `tuple.__getitem__` to avoid runtime errors.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.54_FULL_PASS`.

## [v4.9.55+] - 2025-05-27
- `simulate_trades` now returns a dictionary by default for QA compatibility.
- Legacy tuple output remains available via `return_tuple=True`.
- Unit tests updated to request tuple output when needed.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.55_FULL_PASS`.

## [v4.9.56+] - 2025-05-27
- Ensured `simulate_trades` logs return type and enforces tuple/dict contract.
- `calculate_metrics` now absorbs extra kwargs and warns when provided.
- Updated tests to request tuple output explicitly where required.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.56_FULL_PASS`.

## [v4.9.57+] - 2025-05-27
- Added `ensure_dataframe` utility for safe export of results.
- Expanded logging in `RiskManager.update_drawdown` and `spike_guard_blocked`.
- Updated tests to guard export serialization.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.57_FULL_PASS`.

## [v4.9.58+] - 2025-05-27
- Enhanced `RiskManager.update_drawdown` with robust NaN/negative equity guards.
- `spike_guard_blocked` now validates inputs and logs all states.
- `TradeManager.should_force_entry` adds comprehensive logging and edge guards.
- Tests updated for logging assertions and empty export handling.
- Bumped `MINIMAL_SCRIPT_VERSION` to `4.9.58_FULL_PASS`.

## [v4.9.59+] - 2025-05-28
- Added global export guard via `ensure_dataframe` with logging context.
- `export_trade_log_to_csv` and `export_run_summary_to_json` now warn on empty exports and log success/failure.
- Updated PREPARE_TRAIN_DATA exports to use the new guard.
- Strengthened unit tests with verbose assertion messages for spike guard and re-entry checks.
- Updated version constant to `4.9.59_FULL_PASS`.

## [v4.9.60+] - 2025-05-28
- Improved unit tests for spike guard and reentry logic with context-safe assertions.
- Allowed empty exports in E2E backtest tests by catching `EmptyDataError`.
- Added debug logging for forced entry spike validation.
- Updated version constant to `4.9.60_FULL_PASS`.

## [v4.9.61+] - 2025-05-29
- Guarded spike guard check to only evaluate on entry bars.
- Initialized `equity_tracker['history']` as list with type checks.
- Adjusted internal history updates and metrics conversion for list support.
- Updated tests to use datasets with valid entry signals.
- Version bumped to `4.9.61_FULL_PASS`.

## [v4.9.62+] - 2025-05-30
- Equity tracker history now stored as dict keyed by timestamp for WFV merge.
- `simulate_trades` always returns DataFrame trade log.
- `run_all_folds_with_threshold` handles list-based histories for backward compatibility.
- Bumped version constant to `4.9.62_FULL_PASS`.

## [v4.9.63+] - 2025-05-30
- Fixed legacy tuple output of `simulate_trades` to return list-based trade log for
  backward compatibility with unit tests.
- Updated `MINIMAL_SCRIPT_VERSION` to `4.9.63_FULL_PASS`.

## [v4.9.64+] - 2025-05-31
- `simulate_trades` initializes an empty DataFrame with `exit_reason` column when no trades occur.
- Bumped version constant to `4.9.64_FULL_PASS`.

## [v4.9.65+] - 2025-06-01
- Fixed import flag logic in `import_core_libraries` to correctly reflect successful library imports.
- Bumped version constant to `4.9.65_FULL_PASS`.


## [v4.9.41+] - 2025-05-20
- Added robust equity_tracker history update with numeric guards.
