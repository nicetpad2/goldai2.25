# AGENTS.md

**Gold AI Enterprise – Agent Roles, Patch Protocol, and Test/QA Standards**  

**Version:** v4.9.164+
=======
**Version:** v4.9.163+

**Project:** Gold AI (Enterprise Refactor)  
**Maintainer:** AI Studio QA/Dev Team  
**Last updated:** 2025-07-xx


Gold AI Enterprise QA/Dev version: v4.9.163+ (ATR feature update, doc update instructions, config fail-safe, patch verbose suppression, class attribute fix, equity history dict fix, equity history QA test)
=======
Gold AI Enterprise QA/Dev version: v4.9.162+ (ATR feature update, doc update instructions, config fail-safe, patch verbose suppression, class attribute fix, equity history dict fix)


---

## 🟦 **สถานการณ์ปัจจุบัน & Task ที่กำลังทดสอบ**

- **[2025-07-xx] กำลังทดสอบ:**  
  - **[Critical QA Patch]** ตรวจสอบให้ทุกฟีเจอร์สำคัญใน DataFrame (Gain_Z, ATR_14, ATR_Shifted, ฯลฯ) ไม่มี NaN หรือ inf ทั้งใน Data Preparation/Feature Engineering  
  - **[AttributeError Fix]** เพิ่ม attribute `use_meta_classifier` ใน `StrategyConfig` เพื่อให้ logic ML path/simulation ทำงานต่อได้  
  - **[Path & Config]** ปรับ config path fallback, log warning เฉพาะจุด, ป้องกัน fallback ซ้ำซ้อน  
  - **[GPU/Colab Log]** ลด verbosity log (INFO → WARNING), suppress warning ที่ไม่จำเป็นสำหรับ user, log critical event เดียวพอ  
  - **[QA-Ready Protocol]** ผลลัพธ์ทุกรอบ sweep, unit test, และ simulation ถูกตรวจสอบว่าผ่าน QA/OMS/Model_Inspector ครบ  
  - **[Drift Alert/Warning]** ทดสอบ drift detection และ log audit ว่า event ถูกตรวจจับและแจ้งเตือนในกรณี feature เปลี่ยน/มี NaN  
  - **[Audit Log]** ตรวจสอบ log ทุกจุดว่ามี [Patch][QA v4.9.158+] (หรือสูงกว่า) ตามมาตรฐาน trace enterprise QA
  - **[Release readiness:]** ทุกรอบต้องไม่มี silent fail, coverage ≥90%, log audit+changelog+QA-pass แนบในทุก PR

---

## 🧠 Core AI Units

| Agent                  | Main Role           | Responsibilities                                                                                                                              |
|------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **GPT Dev**            | Core Algo Dev      | Implements/patches core logic (simulate_trades, update_trailing_sl, run_backtest_simulation_v34), SHAP/MetaModel, applies `[Patch AI Studio v4.9.26+]` – `[v4.9.53+]` |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions to clear AI Studio/Codex prompts, organizes multi-step patching                                                 |
| **Code_Runner_QA**     | Execution Test     | Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA                                                 |
| **GoldSurvivor_RnD**   | Strategy Analyst   | Analyzes TP1/TP2, SL, spike, pattern, verifies entry/exit correctness                                                                         |
| **ML_Innovator**       | Advanced ML        | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                                                                 |
| **Model_Inspector**    | Model Diagnostics  | Checks overfitting, noise, data leakage, fallback correctness, metrics drift                                                                  |

---

## 🛡 Risk & Execution

| Agent                 | Main Role        | Responsibilities                                                            |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| **OMS_Guardian**      | OMS Specialist  | Validates order management: risk, TP/SL, lot sizing, spike, forced entry    |
| **System_Deployer**   | Live Trading    | (Future) Manages deployment, monitoring, CI/CD, live risk switch            |
| **Param_Tuner_AI**    | Param Tuning    | Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session logic   |

---

## 🧪 Test & Mocking

| Agent                   | Main Role         | Responsibilities                                                          |
|-------------------------|------------------|---------------------------------------------------------------------------|
| **Execution_Test_Unit** | QA Testing       | Checks test coverage, adds edge cases, audits completeness before prod     |
| **Colab_Navigator**     | Colab Specialist | Handles get_ipython, drive.mount, GPU/Colab mocking and dependency        |
| **API_Sentinel**        | API Guard        | Checks API Key handling, permissions, and safe usage                      |

---

## 📊 Analytics & Drift

| Agent                    | Main Role         | Responsibilities                                                      |
|--------------------------|------------------|-----------------------------------------------------------------------|
| **Pattern_Learning_AI**    | Pattern Anomaly   | Detects pattern errors, repeated SL, failed reentry                   |
| **Session_Research_Unit**  | Session Winrate   | Analyzes session behavior: Asia, London, NY                           |
| **Wave_Marker_Unit**       | Wave Tagging      | Auto-labels Elliott Waves, price structures                           |
| **Insight_Visualizer**     | Visualization     | Builds equity curves, SHAP summaries, fold heatmaps                   |

---

## 🔁 Patch Protocols & Version Control

- **Explicit Versioning:**  
  All patches/agent changes must log version (e.g., `v4.9.158+`) matching latest codebase.

- **Patch Logging:**  
  All logic changes must log `[Patch AI Studio vX.Y.Z+]` หรือ `[Patch][QA vX.Y.Z+]` ตรงกับเวอร์ชันจริงใน CHANGELOG.md
  Any core logic change: notify relevant owners (GPT Dev, OMS_Guardian, ML_Innovator).
  After each patch/update, append details to **CHANGELOG.md** และปรับเวอร์ชันใน **AGENTS.md** ให้ตรงกัน

- **Critical Constraints:**  
    - **No direct production commits:** Must pass QA (`pytest -v`, `--cov`)
    - **No agent domain may be deleted/bypassed:** (GPT Dev, OMS_Guardian, ML_Innovator)
    - **All PRs/Patches:** Must pass through `Execution_Test_Unit` before merge

---

## 🚦 **Enterprise QA Status (Current): ON**


 - QA Enterprise Status: **ON (patch v4.9.163+)**
=======
 - QA Enterprise Status: **ON (patch v4.9.162+)**

- Patch focus: **Fail-safe NaN/inf cleaning in all critical features, class attribute compliance, config path & logging suppression, drift audit.**
- **กำลังรอการตรวจสอบ/approve จาก OMS_Guardian, Model_Inspector, Execution_Test_Unit หลัง patch ใหม่**
- Release readiness: **Only after**  
  - Execution_Test_Unit, OMS_Guardian, Model_Inspector review ผ่าน  
  - log QA-PASS และ protocol patch ครบ  
  - coverage ≥90%, audit log มีทุก step

---

**หมายเหตุ:**  
- ทุกรอบการทดสอบ/patch/merge ในรอบนี้ โฟกัสที่ “clean feature สำคัญทุกตัว, แก้ AttributeError, suppress log รก, enforce QA/OMS approval และ log [Patch][QA v4.9.158+] ขึ้นไปทุกจุดที่เกี่ยวข้อง”
- หากพบ logic หรือ log ใดไม่ตรงกับ AGENTS.md, CHANGELOG.md หรือไม่ log audit ครบ ให้ rollback/patch ทันที

---
