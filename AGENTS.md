# AGENTS.md

**Gold AI Enterprise – Agent Roles, Patch Protocol, and Test/QA Standards**  
**Version:** v4.9.40+  
**Project:** Gold AI (Enterprise Refactor)  
**Maintainer:** AI Studio QA/Dev Team  
**Last updated:** 2025-05-19

---

## 🧠 Core AI Units

| Agent                | Main Role           | Responsibilities                                                                                                                                                           |
|----------------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **GPT Dev**          | Core Algo Dev      | Implements/patches core (simulate_trades, update_trailing_sl), ensures enterprise logic, SHAP/MetaModel, applies `[Patch AI Studio v4.9.26+]`, `[v4.9.29]`, `[v4.9.34+]`  |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions to clear AI Studio/Codex prompts, organizes multi-step patching                                                                              |
| **Code_Runner_QA**   | Execution Test     | Runs scripts, collects pytest results, sys.path, checks logs, prepares zip for Studio/QA                                                                                   |
| **GoldSurvivor_RnD** | Strategy Analyst   | Analyzes TP1/TP2, SL, spike, pattern, verifies entry/exit correctness                                                                                                      |
| **ML_Innovator**     | Advanced ML        | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                                                                                              |
| **Model_Inspector**  | Model Diagnostics  | Checks overfitting, noise, data leakage, fallback correctness, metrics drift                                                                                               |

---

## 🛡 Risk & Execution

| Agent              | Main Role        | Responsibilities                                                                                      |
|--------------------|-----------------|------------------------------------------------------------------------------------------------------|
| **OMS_Guardian**   | OMS Specialist  | Validates order management: risk, TP/SL, lot sizing, spike, forced entry                             |
| **System_Deployer**| Live Trading    | (Future) Manages deployment, monitoring, CI/CD, live risk switch                                     |
| **Param_Tuner_AI** | Param Tuning    | Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session-specific logic                   |

---

## 🧪 Test & Mocking

| Agent                 | Main Role        | Responsibilities                                                               |
|-----------------------|------------------|---------------------------------------------------------------------------------|
| **Execution_Test_Unit** | QA Testing      | Checks test coverage, adds edge cases, audits completeness before production    |
| **Colab_Navigator**     | Colab Specialist| Handles get_ipython, drive.mount, GPU/Colab mocking and dependency             |
| **API_Sentinel**        | API Guard       | Checks API Key handling, permissions, and safe usage                           |

---

## 📊 Analytics & Drift

| Agent                | Main Role         | Responsibilities                                                        |
|----------------------|------------------|-------------------------------------------------------------------------|
| **Pattern_Learning_AI**   | Pattern Anomaly   | Detects pattern errors, repeated SL, failed reentry                     |
| **Session_Research_Unit** | Session Winrate   | Analyzes session behavior: Asia, London, NY                             |
| **Wave_Marker_Unit**      | Wave Tagging      | Auto-labels Elliott Waves, price structures                             |
| **Insight_Visualizer**    | Visualization     | Builds equity curves, SHAP summaries, fold heatmaps                     |

---

## 🔁 Patch Protocols & Version Control

- **Explicit Versioning:**  
  All patches/agent changes must explicitly log version (e.g., `v4.9.40+`) matching latest codebase.

- **Patch Logging:**  
  All logic changes must log `[Patch AI Studio v4.9.26+]`, `[v4.9.29+]`, `[v4.9.34+]` (or latest).  
  Any change to core logic must notify relevant owners (GPT Dev, OMS_Guardian, ML_Innovator).

- **Critical Constraints:**  
    - **No direct commits to production:** Must pass QA (`pytest -v`, `--cov`)
    - **No agent domain may be deleted/bypassed:** (GPT Dev, OMS_Guardian, ML_Innovator)
    - **All PRs/Patches:** Must pass through `Execution_Test_Unit` before merge

---

## 🧩 Agent Test Runner – QA Key Features

**Version:** 4.9.40+  
**Purpose:** Validates Gold AI system: robust import handling, dynamic mocking, complete unit test execution.

**Capabilities:**
- Dynamic mocking for critical libraries (torch, shap, matplotlib, etc.)
- Inline patch protocol for ImportError, __version__ errors
- Runs all tests in `test_gold_ai.py`, logs results (multi-order, BE-SL, kill switch, etc.)
- Ensures all fallback libs have a `__version__`
- `[Patch AI Studio v4.9.26+]`: All trade exits, log events, run_summary flags are auditable
- `[Patch AI Studio v4.9.29+]`: All dynamic type guards use `_isinstance_safe` or robust equivalent
- `[Patch AI Studio v4.9.34+]`: All edge/branch/minimal/failure/DataFrame guards are covered
- No dependencies beyond (`gold_ai2025.py`, `test_gold_ai.py`)

### 🧪 Mock Targets (for test_runner)
`torch`, `shap`, `catboost`, `matplotlib`, `matplotlib.pyplot`, `matplotlib.font_manager`, `scipy`, `optuna`, `GPUtil`, `psutil`, `cv2`, `IPython`, `google.colab`, `google.colab.drive`

### 🔥 Critical Tests (for test_runner)
- `TestGoldAIPart1SetupAndEnv`
- `test_library_import_fails_install_succeeds`
- `test_environment_is_colab_drive_mount_succeeds`
- `test_library_already_imported`
- `test_simulate_trades_multi_order_with_reentry`
- `test_simulate_trades_multi_order_with_reentry_fixed`
- `test_besl_trigger`
- `test_simulate_trades_with_kill_switch_activation`
- Edge/branch/typeguard for `_isinstance_safe`, minimal/None/NaT paths

---

## 🛡 Type Guard Patch – All Core Agents

**All dynamic isinstance(obj, expected_type) checks must use:**
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
 QA Flow & Testing Requirements
Coverage Target:
All patches must bring test coverage >90% for test_gold_ai.py + gold_ai2025.py (excluding placeholders).

Placeholders:
Known placeholders (MT5Connector, FutureAdditions) are skipped from coverage.

Edge/Fail Branches:
All failure/edge/testable paths (DataFrame guards, file not found, NaT, type error) must be exercised by at least one test.

Patch Review:
All merges require:

Full live log

Error summary

Review vs. this AGENTS.md

No merge without Execution_Test_Unit pass and log review
