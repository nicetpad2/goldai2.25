# AGENTS.md

## Gold AI Enterprise - Agent Roles, Patch Protocol, and Test/QA Standards  
**Version:** v4.9.29  
**Project:** Gold AI (Enterprise Refactor)  

---

### 🧠 Core AI Units

| Agent                | Main Role            | Responsibilities                                                                                   |
|----------------------|---------------------|----------------------------------------------------------------------------------------------------|
| **GPT Dev**          | Core Algo Dev       | Implements/patches core functions (`simulate_trades`, `update_trailing_sl`), ensures enterprise logic, follows SHAP/MetaModel, and applies `[Patch AI Studio v4.9.26+]` + `[Patch AI Studio v4.9.29]` |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions into clear AI Studio/Codex prompts and organizes multi-step patching |
| **Code_Runner_QA**   | Execution Test      | Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA      |
| **GoldSurvivor_RnD** | Strategy Analyst    | Analyzes TP1/TP2, SL, spike, pattern, and verifies entry/exit correctness                          |
| **ML_Innovator**     | Advanced ML         | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                      |
| **Model_Inspector**  | Model Diagnostics   | Checks for overfitting, noise, data leakage, fallback correctness, and metrics drift               |

---

### 🛡 Risk & Execution

| Agent              | Main Role          | Responsibilities                                                                              |
|--------------------|-------------------|-----------------------------------------------------------------------------------------------|
| **OMS_Guardian**   | OMS Specialist    | Validates order management: risk, TP/SL, lot sizing, spike, and forced entry                  |
| **System_Deployer**| Live Trading      | (Future) Manages deployment, monitoring, CI/CD, live risk switch                              |
| **Param_Tuner_AI** | Param Tuning      | Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session-specific logic            |

---

### 🧪 Test & Mocking

| Agent                  | Main Role         | Responsibilities                                                                                 |
|------------------------|------------------|--------------------------------------------------------------------------------------------------|
| **Execution_Test_Unit**| QA Testing       | Checks test coverage, adds edge cases, audits completeness before production                     |
| **Colab_Navigator**    | Colab Specialist | Handles `get_ipython`, `drive.mount`, GPU/Colab mocking and dependency                          |
| **API_Sentinel**       | API Guard        | Checks API Key handling, permissions, and safe usage                                             |

---

### 📊 Analytics & Drift

| Agent                | Main Role            | Responsibilities                                                    |
|----------------------|---------------------|---------------------------------------------------------------------|
| **Pattern_Learning_AI** | Pattern Anomaly    | Detects pattern errors, repeated SL, and failed reentry             |
| **Session_Research_Unit** | Session Winrate | Analyzes session behavior: Asia, London, NY                         |
| **Wave_Marker_Unit** | Wave Tagging        | Auto-labels Elliott Waves and other price structures                |
| **Insight_Visualizer** | Visualization      | Builds equity curves, SHAP summaries, and fold heatmaps             |

---

## Patch Protocols & Version Control

- **Explicit versioning (e.g., v4.9.29)**:  
  Every patch/agent must track logic version matching latest patch.

- **Patch Logging**:  
  All logic changes must log `[Patch AI Studio v4.9.26+]` (and v4.9.29 for type guard/edge).  
  Agents modifying production code must notify all relevant agents (esp. GPT Dev, OMS_Guardian, ML_Innovator).

- **Critical Constraints**:
  - No direct commits to production: Everything must pass QA (pytest -v, --cov, etc.).
  - No system owned by GPT Dev, OMS_Guardian, or ML_Innovator can be deleted without approval.
  - All pull requests/patches must pass through **Execution_Test_Unit** before merge.

---

## Agent Test Runner - Key Features (gold_ai_test_runner)

- **Version:** 4.9.29
- **Purpose:** Validates Gold AI backtesting/simulation system with robust import handling, dynamic mocking, and unit test execution for enterprise/AI Studio environments.
- **Capabilities**:
  - Dynamic mocking for critical libraries (torch, shap, matplotlib, etc.)
  - Inline patch protocol to avoid ImportError and __version__ errors
  - Executes all tests in test_gold_ai.py, reports results (multi-order, BE-SL, kill switch, etc.)
  - Logs all fallback imports, patch status, and mock activity
  - Ensures all fallback libraries have a `__version__`
  - **[Patch AI Studio v4.9.26+]**: All trade exits, log events, run_summary flags must be auditable
  - **[Patch AI Studio v4.9.29]**: All dynamic type guards must use `_isinstance_safe`
  - No dependencies beyond listed files (gold_ai2025.py, test_gold_ai.py)

---

## Mock Targets (for test_runner)

- torch, shap, catboost, matplotlib, matplotlib.pyplot, matplotlib.font_manager, scipy, optuna, GPUtil, psutil, cv2, IPython, google.colab, google.colab.drive

---

## Critical Tests (for test_runner)

- TestGoldAIPart1SetupAndEnv
- test_library_import_fails_install_succeeds
- test_environment_is_colab_drive_mount_succeeds
- test_library_already_imported
- test_simulate_trades_multi_order_with_reentry
- test_simulate_trades_multi_order_with_reentry_fixed
- test_besl_trigger
- test_simulate_trades_with_kill_switch_activation

---

## Type Guard Patch (v4.9.29) - All Core Agents

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
