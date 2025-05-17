# Agent: gold_ai_test_runner
version: 4.9.25
description: >
  This agent is responsible for validating the Gold AI backtesting and simulation system,
  focusing on robust import handling, mock patching of critical libraries, and unit test execution.
  It is designed for Codex or AI Studio environments where certain libraries may not be available.

goals:
  - Import gold_ai2025.py safely with mocked libraries (torch, shap, matplotlib, etc.)
  - Apply inline patches to avoid ImportError and __version__ AttributeError
  - Run all unit tests in test_gold_ai.py and collect results
  - Ensure 100% test case collection without crashing (even if some tests are skipped)
  - Log fallback import results and mock status
  - Validate that fallback libraries have __version__ attribute set
  - Avoid all external file/module imports (no use of gold_ai_patch_studio.py)

files:
  - gold_ai2025.py
  - test_gold_ai.py

tools:
  - file_system
  - subprocess_runner
  - pytest_runner
  - patch_editor
  - python_ast_inspector

permissions:
  - allow_inline_patch
  - allow_module_mocking
  - allow_safe_sys_modules_override

mock_targets:
  - torch
  - shap
  - catboost
  - matplotlib
  - matplotlib.pyplot
  - matplotlib.font_manager
  - scipy
  - optuna
  - GPUtil
  - psutil
  - cv2
  - IPython
  - google.colab
  - google.colab.drive

critical_tests:
  - TestGoldAIPart1SetupAndEnv
  - test_library_import_fails_install_succeeds
  - test_environment_is_colab_drive_mount_succeeds
  - test_library_already_imported

constraints:
  - No external dependencies beyond listed files
  - All patches must be inline within test_gold_ai.py
  - Coverage tracking is optional but preferred
