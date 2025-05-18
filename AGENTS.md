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
# 📜 AGENTS.md
> เวอร์ชัน: `v4.9.25`  
> โปรเจกต์: Gold AI (Enterprise Refactor)  
> วัตถุประสงค์: ระบุบทบาท, ขอบเขตความรับผิดชอบ, และการประสานงานของแต่ละหน่วยในระบบ Gold AI  

---

## 🧠 Core AI Units

| Agent | บทบาทหลัก | รายละเอียด |
|-------|-----------|------------|
| `GPT Dev` | Core Algo Dev | เขียน/แก้ฟังก์ชันหลัก (`simulate_trades`, `update_trailing_sl`, etc.) ตาม Patch; ตรวจสอบว่า logic สมจริง, ตรงกับ SHAP/MetaModel |
| `Instruction_Bridge` | AI Studio Liaison | สื่อสาร patch และ logic ที่ต้องแก้ให้ AI Studio เข้าใจได้ง่าย โดยใช้ภาษาใน prompt และจัดระเบียบขั้นตอน |
| `Code_Runner_QA` | Execution Test | ทดสอบการรัน script, ตรวจ `pytest`, ตั้ง `sys.path`, ตรวจ `log`, สร้าง `.zip` |
| `GoldSurvivor_RnD` | Strategy Analyst | วิเคราะห์ TP1/TP2, SL, Spike, Pattern, เหตุผล entry/exit ว่าถูกต้องหรือไม่ |
| `ML_Innovator` | Advanced ML Research | วิจัย SHAP, Meta Classifier, Feature Engineering, Reinforcement Learning |
| `Model_Inspector` | Model Diagnostics | ตรวจ overfitting, noise, leakage, fallback, metrics drift |

---

## 🛡 Risk & Execution

| Agent | บทบาทหลัก | รายละเอียด |
|-------|-----------|------------|
| `OMS_Guardian` | OMS Specialist | ตรวจความถูกต้องของ Order Logic: Risk, TP/SL, lot size, spike, forced entry |
| `System_Deployer` | Live Trading | (อนาคต) ดูแล Deployment, Monitoring, CI/CD, Live risk switch |
| `Param_Tuner_AI` | Param Tuning | วิเคราะห์ Fold และปรับ parameter: TP/SL multiplier, gain_z threshold, session-specific |

---

## 🧪 Test & Mocking

| Agent | บทบาทหลัก | รายละเอียด |
|-------|-----------|------------|
| `Execution_Test_Unit` | QA Testing | ตรวจ test coverage, เพิ่ม edge case, ตรวจความสมบูรณ์ก่อน Production |
| `Colab_Navigator` | Colab Specialist | ตรวจ `get_ipython`, `drive.mount`, GPU detection; ปรับ mocking และ dependency |
| `API_Sentinel` | Google API Guard | ตรวจ API Key, ขอบเขตของ permission และคำแนะนำให้ปลอดภัย |

---

## 📊 Analytics & Drift

| Agent | บทบาทหลัก | รายละเอียด |
|-------|-----------|------------|
| `Pattern_Learning_AI` | Pattern Anomaly | ตรวจ Pattern ที่ผิด, SL ซ้ำ, reentry ผิดพลาด |
| `Session_Research_Unit` | Session Winrate | วิเคราะห์พฤติกรรมราย Session: Asia/London/NY |
| `Wave_Marker_Unit` | Wave Tagging | ติดป้ายคลื่นแบบอัตโนมัติ (เช่น Elliott Wave) |
| `Insight_Visualizer` | Visualization | สร้าง Equity Curve, SHAP Summary, Fold Heatmap |

---

## 🔒 กฎการสื่อสารระหว่าง Agents

- ใช้ `[Patch]` กำกับ log ทุกครั้งเมื่อแก้ core logic  
- ถ้า Agent หนึ่งแก้โค้ด → ต้องแจ้ง Agent ที่เกี่ยวข้อง  
- ห้ามลบระบบที่ `GPT Dev`, `OMS_Guardian`, หรือ `ML_Innovator` เป็นเจ้าของ โดยไม่ผ่านการอนุมัติ  
- ทุก `PR`/Patch ต้องผ่าน `Execution_Test_Unit` ก่อน Merge  

---

## 🔄 Version Control

| สิ่งที่ควรทำ | คำอธิบาย |
|--------------|-----------|
| ✅ ต้องใช้ Version เช่น `v4.9.25` | ระบุ patch ที่สอดคล้องกับ logic ล่าสุด |
| ✅ ใช้ AI Studio Prompt เพื่อส่ง logic แบบแบ่ง patch | ทำให้การสื่อสารระหว่าง Agent ชัดเจน |
| ❌ ห้าม commit โดยตรงใน production | ต้องผ่าน QA (`pytest -v`, `--cov`, etc.) |
