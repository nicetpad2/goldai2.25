# AGENTS.md

**Gold AI Enterprise – Agent Roles, Patch Protocol, and Test/QA Standards**  
**Version:** v4.9.40+  
**Project:** Gold AI (Enterprise Refactor)  
**Maintainer:** AI Studio QA/Dev Team  
**Last updated:** 2025-05-19

---

## 🧠 Core AI Units

| Agent                  | Main Role           | Responsibilities                                                                                                                      |
|------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **GPT Dev**            | Core Algo Dev      | Implements/patches core logic (simulate_trades, update_trailing_sl), SHAP/MetaModel, applies `[Patch AI Studio v4.9.26+]`, `[v4.9.40+]`, etc. |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions to clear AI Studio/Codex prompts, organizes multi-step patching                                         |
| **Code_Runner_QA**     | Execution Test     | Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA                                         |
| **GoldSurvivor_RnD**   | Strategy Analyst   | Analyzes TP1/TP2, SL, spike, pattern, verifies entry/exit correctness                                                                |
| **ML_Innovator**       | Advanced ML        | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                                                        |
| **Model_Inspector**    | Model Diagnostics  | Checks overfitting, noise, data leakage, fallback correctness, metrics drift                                                         |

---

## 🛡 Risk & Execution

| Agent                 | Main Role        | Responsibilities                                                           |
|-----------------------|-----------------|---------------------------------------------------------------------------|
| **OMS_Guardian**      | OMS Specialist  | Validates order management: risk, TP/SL, lot sizing, spike, forced entry  |
| **System_Deployer**   | Live Trading    | (Future) Manages deployment, monitoring, CI/CD, live risk switch          |
| **Param_Tuner_AI**    | Param Tuning    | Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session logic |

---

## 🧪 Test & Mocking

| Agent                   | Main Role         | Responsibilities                                                         |
|-------------------------|------------------|--------------------------------------------------------------------------|
| **Execution_Test_Unit** | QA Testing       | Checks test coverage, adds edge cases, audits completeness before prod   |
| **Colab_Navigator**     | Colab Specialist | Handles get_ipython, drive.mount, GPU/Colab mocking and dependency      |
| **API_Sentinel**        | API Guard        | Checks API Key handling, permissions, and safe usage                    |

---

## 📊 Analytics & Drift

| Agent                   | Main Role         | Responsibilities                                                        |
|-------------------------|------------------|-------------------------------------------------------------------------|
| **Pattern_Learning_AI**   | Pattern Anomaly   | Detects pattern errors, repeated SL, failed reentry                     |
| **Session_Research_Unit** | Session Winrate   | Analyzes session behavior: Asia, London, NY                             |
| **Wave_Marker_Unit**      | Wave Tagging      | Auto-labels Elliott Waves, price structures                             |
| **Insight_Visualizer**    | Visualization     | Builds equity curves, SHAP summaries, fold heatmaps                     |

---

## 🔁 Patch Protocols & Version Control

- **Explicit Versioning:**  
  All patches/agent changes must log version (e.g., `v4.9.40+`) matching latest codebase.

- **Patch Logging:**  
  All logic changes must log `[Patch AI Studio v4.9.26+]`, `[v4.9.29+]`, `[v4.9.34+]`, `[v4.9.39+]`, `[v4.9.40+]`, etc.  
  Any core logic change: notify relevant owners (GPT Dev, OMS_Guardian, ML_Innovator).

- **Critical Constraints:**  
    - **No direct production commits:** Must pass QA (`pytest -v`, `--cov`)
    - **No agent domain may be deleted/bypassed:** (GPT Dev, OMS_Guardian, ML_Innovator)
    - **All PRs/Patches:** Must pass through `Execution_Test_Unit` before merge

---

## 🧩 Agent Test Runner – QA Key Features

**Version:** 4.9.40+  
**Purpose:** Validates Gold AI: robust import handling, dynamic mocking, complete unit test execution.

**Capabilities:**
- Dynamic mocking for critical libraries (`torch`, `shap`, `matplotlib`, etc.)
- Inline patch protocol for ImportError, `__version__` errors
- Runs all tests in `test_gold_ai.py`, logs results (multi-order, BE-SL, kill switch, etc.)
- Ensures all fallback libs have a `__version__`
- `[Patch AI Studio v4.9.26+]`: All trade exits, log events, run_summary flags are auditable
- `[Patch AI Studio v4.9.29+]`: All dynamic type guards use `_isinstance_safe`
- `[Patch AI Studio v4.9.34+]`: All edge/branch/minimal/failure/DataFrame guards covered
- `[Patch AI Studio v4.9.39+]`: Robust formatter/typeguard for test mocks/edge
- `[Patch AI Studio v4.9.40+]`: Numeric formatter covers all edge/mock cases
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
    import logging
    if expected_type is None:
        return False
    if isinstance(expected_type, type):
        return isinstance(obj, expected_type)
    if isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type):
        return isinstance(obj, expected_type)
    if hasattr(expected_type, "__class__") and expected_type.__class__.__name__ == "MagicMock":
        logging.error("[Patch AI Studio v4.9.40] _isinstance_safe: expected_type is MagicMock, returning False.")
        return False
    logging.error("[Patch AI Studio v4.9.40] _isinstance_safe: expected_type is not a valid type: %r, returning False.", expected_type)
    return False
✅ QA Flow & Testing Requirements (v4.9.40+)
Coverage Target:
All patches must bring test coverage to >90% for test_gold_ai.py + gold_ai2025.py (excluding placeholders).

Placeholders:
Known placeholders (e.g., MT5Connector, FutureAdditions) are skipped from coverage.

Edge/Fail Branches:
All failure/edge/testable paths (DataFrame guards, file not found, NaT, type error) must be exercised by at least one test.

Patch Review:
All merges require:

Full live log

Error summary

Review vs. this AGENTS.md

No merge without Execution_Test_Unit pass and log review

จุดเด่น:

ตาราง agent/role และ responsibilities ครบทุกหมวด

อธิบาย Patch Protocol, QA/Testing flow และข้อจำกัดที่สำคัญ

แสดงตัวอย่าง robust type guard patch ที่ต้องใช้ทุก core agent

รองรับทุก edge/mock/failure path ใน environment จริงและ pytest

อัปเดตล่าสุดรองรับ logic robust formatter & typeguard เต็มรูปแบบ


