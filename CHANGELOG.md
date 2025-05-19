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


## [v4.9.41+] - 2025-05-20
- Added robust equity_tracker history update with numeric guards.
