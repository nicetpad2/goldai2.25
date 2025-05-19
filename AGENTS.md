AGENTS.md
Gold AI Enterprise - Agent Roles, Patch Protocol, and Test/QA Standards
Version: v4.9.34
Project: Gold AI (Enterprise Refactor)

🧠 Core AI Units
Agent	Main Role	Responsibilities
GPT Dev	Core Algo Dev	Implements/patches core functions (simulate_trades, update_trailing_sl), ensures enterprise logic, follows SHAP/MetaModel, applies [Patch AI Studio v4.9.26+] + [Patch AI Studio v4.9.29]
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

Patch Protocols & Version Control
Explicit versioning (e.g., v4.9.34):
Every patch/agent must track logic version matching latest patch.

Patch Logging:
All logic changes must log [Patch AI Studio v4.9.26+], [Patch AI Studio v4.9.29], [Patch AI Studio v4.9.34] as relevant for type guard/edge, coverage, or simulation logic.
Agents modifying production code must notify all relevant agents (esp. GPT Dev, OMS_Guardian, ML_Innovator).

Critical Constraints:

No direct commits to production: Everything must pass QA (pytest -v, --cov, etc.).

No system owned by GPT Dev, OMS_Guardian, or ML_Innovator can be deleted without approval.

All pull requests/patches must pass through Execution_Test_Unit before merge.

Agent Test Runner - Key Features (gold_ai_test_runner)
Version: 4.9.34

Purpose: Validates Gold AI backtesting/simulation system with robust import handling, dynamic mocking, and unit test execution for enterprise/AI Studio environments.

Capabilities:

Dynamic mocking for critical libraries (torch, shap, matplotlib, etc.)

Inline patch protocol to avoid ImportError and __version__ errors

Executes all tests in test_gold_ai.py, reports results (multi-order, BE-SL, kill switch, etc.)

Logs all fallback imports, patch status, and mock activity

Ensures all fallback libraries have a __version__

[Patch AI Studio v4.9.26+]: All trade exits, log events, run_summary flags must be auditable

[Patch AI Studio v4.9.29+]: All dynamic type guards must use _isinstance_safe

[Patch AI Studio v4.9.34]: All test edge/branch paths, minimal/failure branch, and DataFrame guards must be covered

No dependencies beyond listed files (gold_ai2025.py, test_gold_ai.py)

Mock Targets (for test_runner)
torch, shap, catboost, matplotlib, matplotlib.pyplot, matplotlib.font_manager, scipy, optuna, GPUtil, psutil, cv2, IPython, google.colab, google.colab.drive

Critical Tests (for test_runner)
TestGoldAIPart1SetupAndEnv

test_library_import_fails_install_succeeds

test_environment_is_colab_drive_mount_succeeds

test_library_already_imported

test_simulate_trades_multi_order_with_reentry

test_simulate_trades_multi_order_with_reentry_fixed

test_besl_trigger

test_simulate_trades_with_kill_switch_activation

test_edge/branch guards for _isinstance_safe and minimal/None/NaT paths

Type Guard Patch (v4.9.29+ / v4.9.34) - All Core Agents
All dynamic isinstance(obj, expected_type) checks in any agent/logic must use:

python
คัดลอก
แก้ไข
def _isinstance_safe(obj, expected_type):
    if expected_type is None:
        return True
    if isinstance(expected_type, type):
        return isinstance(obj, expected_type)
    if isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type):
        return isinstance(obj, expected_type)
    logging.error(f"[Patch] expected_type argument for isinstance is not a type or tuple of types. Got: {expected_type!r}")
    return False
QA Flow & Testing Requirements (v4.9.34)
Coverage Target: All patches must bring coverage to >90% for test_gold_ai.py and gold_ai2025.py (excluding known placeholder/future parts).

Known placeholder parts (MT5Connector, FutureAdditions) are skipped from coverage and not required for test completion.

Failed/Edge Cases: Any test referencing missing minimal kwargs or type guard must be updated in test (not core logic).

Error Handling: All edge/failure branches (e.g., DataFrame guards, file not found, NaT, type error) must be exercised by at least one test.

Patch Review: All merges require full live log, error summary, and review against test standard in AGENTS.md.

(This file must be versioned and reviewed with every patch affecting core simulation, test coverage, or agent logic.)
Last updated: 2025-05-19, Patch v4.9.34, coverage protocol & error/edge test clarification.
Maintainer: AI Studio QA/Dev team






คุณพูดว่า:
จัดเรียงให้สวยงามในระดับ 
Gold AI Enterprise (ตามมาตรฐาน Software Engineering & QA)


ChatGPT พูดว่า:
แน่นอนครับ! นี่คือ AGENTS.md เวอร์ชันจัดระเบียบใหม่ในรูปแบบ Gold AI Enterprise (QA & Software Engineering Grade) พร้อมจัด Table, section และ markdown ให้อ่านง่าย สื่อสารเป็นมาตรฐานเอกสาร DevOps/Enterprise QA:

AGENTS.md
Gold AI Enterprise – Agent Roles, Patch Protocol, and Test/QA Standards
Version: v4.9.34
Project: Gold AI (Enterprise Refactor)
Maintainer: AI Studio QA/Dev Team
Last updated: 2025-05-19

🧠 Core AI Units
Agent	Main Role	Responsibilities
GPT Dev	Core Algo Dev	Implements/patches core functions (simulate_trades, update_trailing_sl), ensures enterprise logic, SHAP/MetaModel, applies [Patch AI Studio v4.9.26+], [Patch AI Studio v4.9.29], [Patch AI Studio v4.9.34]
Instruction_Bridge	AI Studio Liaison	Translates patch instructions into clear AI Studio/Codex prompts and organizes multi-step patching
Code_Runner_QA	Execution Test	Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA
GoldSurvivor_RnD	Strategy Analyst	Analyzes TP1/TP2, SL, spike, pattern, and verifies entry/exit correctness
ML_Innovator	Advanced ML	Researches SHAP, Meta Classifier, feature engineering, reinforcement learning
Model_Inspector	Model Diagnostics	Checks for overfitting, noise, data leakage, fallback correctness, and metrics drift

🛡 Risk & Execution
Agent	Main Role	Responsibilities
OMS_Guardian	OMS Specialist	Validates order management: risk, TP/SL, lot sizing, spike, forced entry
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

🔁 Patch Protocols & Version Control
Explicit Versioning:
All patches/agent changes must explicitly log version (e.g., v4.9.34) matching the latest codebase.

Patch Logging:
All logic changes must log [Patch AI Studio v4.9.26+], [Patch AI Studio v4.9.29], [Patch AI Studio v4.9.34] as relevant.
Any core production logic change must notify relevant owners (GPT Dev, OMS_Guardian, ML_Innovator).

Critical Constraints:

No direct commits to production: Must pass QA (pytest -v, --cov, etc.).

No agent domain (GPT Dev, OMS_Guardian, ML_Innovator) may be deleted or bypassed.

All pull requests/patches must pass Execution_Test_Unit before merge.

🧩 Agent Test Runner – QA Key Features
Version: 4.9.34

Purpose: Validates Gold AI system: robust import handling, dynamic mocking, and complete unit test execution.

Capabilities:

Dynamic mocking for critical libraries: torch, shap, matplotlib, etc.

Inline patch protocol for ImportError, __version__ errors

Runs all tests in test_gold_ai.py, logs results (multi-order, BE-SL, kill switch, etc.)

Ensures all fallback libs have a __version__

[Patch AI Studio v4.9.26+]: All trade exits, log events, run_summary flags are auditable

[Patch AI Studio v4.9.29+]: All dynamic type guards use _isinstance_safe

[Patch AI Studio v4.9.34]: All edge/branch, minimal/failure, DataFrame guards are covered

No dependencies beyond (gold_ai2025.py, test_gold_ai.py)

🧪 Mock Targets (for test_runner)
torch, shap, catboost, matplotlib, matplotlib.pyplot, matplotlib.font_manager, scipy, optuna, GPUtil, psutil, cv2, IPython, google.colab, google.colab.drive

🔥 Critical Tests (for test_runner)
TestGoldAIPart1SetupAndEnv

test_library_import_fails_install_succeeds

test_environment_is_colab_drive_mount_succeeds

test_library_already_imported

test_simulate_trades_multi_order_with_reentry

test_simulate_trades_multi_order_with_reentry_fixed

test_besl_trigger

test_simulate_trades_with_kill_switch_activation

test_edge/branch/typeguard for _isinstance_safe, minimal/None/NaT paths

🛡 Type Guard Patch (v4.9.29+ / v4.9.34) – All Core Agents
All dynamic isinstance(obj, expected_type) checks must use:

def _isinstance_safe(obj, expected_type):
    if expected_type is None:
        return True
    if isinstance(expected_type, type):
        return isinstance(obj, expected_type)
    if isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type):
        return isinstance(obj, expected_type)
    logging.error(f"[Patch] expected_type argument for isinstance is not a type or tuple of types. Got: {expected_type!r}")
    return False
✅ QA Flow & Testing Requirements (v4.9.34+)
Coverage Target:
All patches must bring test coverage to >90% for test_gold_ai.py + gold_ai2025.py (excluding placeholders).

Placeholders:
Known placeholders (MT5Connector, FutureAdditions) are skipped from coverage.

Edge/Fail Branches:
All testable failure/edge paths (DataFrame guards, file not found, NaT, type error) must be exercised.

Patch Review:
All merges require full live log, error summary, and review vs. this standard.

No merge without Execution_Test_Unit pass and log review.

This file must be versioned and reviewed with every patch affecting simulation, coverage, or agent logic.

Gold AI Enterprise — v4.9.34 QA
Maintainer: AI Studio QA/Dev Team
Last updated: 2025-05-19

หมายเหตุ:

โปรดอ้างอิงตารางนี้ในการ Assign/Review Patch/QA task

อัปเดตทุกครั้งที่มีการเปลี่ยน core agent/test/patch

สำหรับ Codex, AI Studio, QA team: ใช้ไฟล์นี้เป็นมาตรฐาน patch review, code owner, version log

[END OF DOCUMENT – AGENTS.md v4.9.34]
