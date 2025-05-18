# Agent: gold_ai_test_runner
version: 4.9.29
description: >
  This agent validates the Gold AI backtesting and simulation system,
  focusing on robust import handling, dynamic mocking of critical libraries, and full unit test execution.
  It is designed for Codex or AI Studio environments, where some dependencies may be unavailable or must be mocked inline.

goals:
  - Safely import gold_ai2025.py with dynamic mocking for libraries (torch, shap, matplotlib, etc.)
  - Apply inline patches to avoid ImportError and __version__ AttributeError
  - Run all unit tests in test_gold_ai.py and collect results (including multi-order, BE-SL, kill switch)
  - Ensure all test cases are collected and executed, with no test runner crashes
  - Log all fallback imports, patch results, and mock status
  - Validate that all fallback libraries have a __version__ attribute
  - No external file/module imports (do not use gold_ai_patch_studio.py)
  - [Patch AI Studio v4.9.26+] Simulate_trades must support multi-order reentry, correct BE-SL/SL logic, and run_summary["hard_kill_triggered"] on every kill switch
  - All logic changes must be logged with [Patch AI Studio v4.9.26+] in the logger
  - [Patch AI Studio v4.9.29] All dynamic type guards (expected_type, isinstance) must use _isinstance_safe to prevent TypeError in edge cases

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
  - test_simulate_trades_multi_order_with_reentry
  - test_simulate_trades_multi_order_with_reentry_fixed
  - test_besl_trigger
  - test_simulate_trades_with_kill_switch_activation

constraints:
  - No dependencies beyond the listed files
  - All patches must be inline in test_gold_ai.py or gold_ai2025.py where possible
  - Coverage tracking is optional but recommended
  - [Patch AI Studio v4.9.26+] All trade exits, trade log events, and run_summary flags must be explicitly auditable
  - [Patch AI Studio v4.9.29] All isinstance checks with dynamic expected_type must be guarded by _isinstance_safe

Version: v4.9.29  
Project: Gold AI (Enterprise Refactor)  
Purpose: Defines agent roles, patch protocols, and test/QA standards for Gold AI (v4.9.29+), including type guard safety and auditability of all critical logic.

🧠 Core AI Units
Agent	Main Role	Responsibilities
GPT Dev	Core Algo Dev	Implements/patches core functions (simulate_trades, update_trailing_sl, etc.), ensures real logic, follows SHAP/MetaModel, and applies [Patch AI Studio v4.9.26+]
Instruction_Bridge	AI Studio Liaison	Translates patch instructions into clear AI Studio/Codex prompts and organizes multi-step patching
Code_Runner_QA	Execution Test	Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA
GoldSurvivor_RnD	Strategy Analyst	Analyzes TP1/TP2, SL, spike, pattern, and verifies entry/exit correctness
ML_Innovator	Advanced ML	Researches SHAP, Meta Classifier, feature engineering, reinforcement learning
Model_Inspector	Model Diagnostics	Checks for overfitting, noise, data leakage, fallback correctness, and metrics drift

🛡 Risk & Execution
Agent	Main Role	Responsibilities
OMS_Guardian	OMS Specialist	Validates order management: risk, TP/SL, lot sizing, spike, and forced entry
System_Deployer	Live Trading	(Future) Manages deployment, monitoring, CI/CD, live risk switch
Param_Tuner_AI	Param Tuning	Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session-specific logic

🧪 Test & Mocking
Agent	Main Role	Responsibilities
Execution_Test_Unit	QA Testing	Checks test coverage, adds edge cases, audits completeness before production
Colab_Navigator	Colab Specialist	Handles get_ipython, drive.mount, GPU/Colab mocking and dependency
API_Sentinel	API Guard	Checks API Key handling, permissions, and safe usage

📊 Analytics & Drift
Agent	Main Role	Responsibilities
Pattern_Learning_AI	Pattern Anomaly	Detects pattern errors, repeated SL, and failed reentry
Session_Research_Unit	Session Winrate	Analyzes session behavior: Asia, London, NY
Wave_Marker_Unit	Wave Tagging	Auto-labels Elliott Waves and other price structures
Insight_Visualizer	Visualization	Builds equity curves, SHAP summaries, and fold heatmaps

🔒 Agent Communication & Control
All core logic changes must log [Patch AI Studio v4.9.26+] (and v4.9.29 for type guard/edge)
Agents modifying production code must notify all relevant agents
No system owned by GPT Dev, OMS_Guardian, or ML_Innovator can be deleted without approval
All pull requests/patches must pass through Execution_Test_Unit before merge

🔄 Version Control
Must-do	Description
✅ Use explicit versioning (e.g., v4.9.29)	Every patch/agent must track logic version matching latest patch
✅ Use AI Studio Prompt Patch Workflow	For clear, stepwise, reviewable changes between agents
❌ No direct commits to production	Everything must pass QA (pytest -v, --cov, etc.)

---
**Type Guard Patch (v4.9.29):**  
All dynamic `isinstance(obj, expected_type)` checks in any agent/logic must use:

```python
def _isinstance_safe(obj, expected_type):
    if expected_type is None:
        return True
    if isinstance(expected_type, type):
        return isinstance(obj, expected_type)
    if isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type):
        return isinstance(obj, expected_type)
    logging.error(f"[Patch] expected_type argument for isinstance is not a type or tuple of types. Got: {expected_type!r}")
    return False
and update all guards:
if not _isinstance_safe(config, expected_type):
    raise TypeError(...)
Note:

All logic in simulate_trades, including BE-SL, multi-order reentry, and kill switch, must be compliant with [Patch AI Studio v4.9.26+] and must audit all trade log and run_summary events.

Every test runner, agent, or QA process must log/audit trade exits and critical flags for Studio review.

[Patch AI Studio v4.9.26+] must appear in all logs and core patch instructions, with v4.9.29 for new type-guard logic.
