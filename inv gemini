# STEP 1: ติดตั้งไลบรารีที่ต้องใช้
!pip install -U google-generativeai
!pip install -U pandas numpy matplotlib ta optuna catboost gputil psutil shap tqdm scikit-learn pyyaml
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install catboost

import subprocess
import os
import sys
import getpass
import glob
import time
import json
import hashlib
import pandas as pd
import psutil
import yaml
import random
import google.generativeai as genai

# ---------- CONFIG -------------
GOLD_AI_SCRIPT = "/content/drive/MyDrive/new/gold_ai2025.py"
CFG_BASE = "/content/drive/MyDrive/new/"
OUTPUT_BASE = "/content/drive/MyDrive/new/gold_ai_sweep/"
REPORT_PATH = CFG_BASE + "goldai_sweep_qa_report.txt"
CSV_RESULT_PATH = CFG_BASE + "goldai_sweep_qa_results.csv"
MD_REPORT_PATH = CFG_BASE + "goldai_sweep_qa_report.md"
CHANGELOG_PATH = CFG_BASE + "goldai_changelog.txt"
LOG_PATH = CFG_BASE + "goldai_sweep_stdout.log"
RAM_PATH = CFG_BASE + "goldai_sweep_max_ram.txt"
GEMINI_API_KEY = os.environ.get("AIzaSyAmNvZzYZHPIFmPhmRnnMpVmFG11DIFcm4") or getpass.getpass("🔑 ใส่ Gemini API Key (จะไม่โชว์): ")

# ---------- PARAM GRID (Hyperparameter Sweep) -------------
PARAM_GRID = [
    {"risk_per_trade": 0.01, "base_tp_multiplier": 1.8, "max_holding_bars": 24},
    {"risk_per_trade": 0.02, "base_tp_multiplier": 1.5, "max_holding_bars": 36},
    {"risk_per_trade": 0.015, "base_tp_multiplier": 2.0, "max_holding_bars": 12},
    # เพิ่มชุด param อื่นๆ ได้ตามต้องการ
]

# ---------- RAM Monitoring -------------
def get_ram_usage_gb():
    vmem = psutil.virtual_memory()
    used_gb = vmem.used / (1024**3)
    total_gb = vmem.total / (1024**3)
    percent = vmem.percent
    return used_gb, total_gb, percent

max_ram_used = 0.0
def update_max_ram():
    global max_ram_used
    used_gb, _, _ = get_ram_usage_gb()
    if used_gb > max_ram_used:
        max_ram_used = used_gb

def print_ram_usage(label=""):
    used_gb, total_gb, percent = get_ram_usage_gb()
    print(f"[RAM {label}] ใช้งาน {used_gb:.2f} GB / {total_gb:.2f} GB ({percent:.1f}%)")

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

def config_hash(cfg_dict):
    s = yaml.dump(cfg_dict)
    return hashlib.md5(s.encode()).hexdigest()

def write_config(cfg_dict, path):
    with open(path, "w") as f:
        yaml.dump(cfg_dict, f)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4096)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------- Hyperparameter Sweep Loop -------------
all_sweep_results = []
sweep_changelog = []
max_ram_overall = 0.0

for idx, param in enumerate(PARAM_GRID):
    config_path = CFG_BASE + f"config_sweep_{idx+1}.yaml"
    pipeline_id = hashlib.md5((str(time.time())+str(random.random())).encode()).hexdigest()[:12]
    param["pipeline_id"] = pipeline_id
    write_config(param, config_path)
    print(f"\n\n========== PARAM SWEEP [{idx+1}/{len(PARAM_GRID)}] (hash={config_hash(param)}) ==========")
    print(param)
    run_output_dir = os.path.join(OUTPUT_BASE, f"run_{idx+1}_{pipeline_id}")
    os.makedirs(run_output_dir, exist_ok=True)

    print_ram_usage(f"ก่อนรัน GoldAI (Sweep {idx+1})")
    ram_before = get_ram_usage_gb()[0]
    process = subprocess.Popen(
        [sys.executable, GOLD_AI_SCRIPT, "--config", config_path, "--output_dir", run_output_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    all_log = ""
    for line in process.stdout:
        print(line, end="")
        all_log += line
        update_max_ram()
    process.wait()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n======== Pipeline {pipeline_id} ========\n{all_log}\n")
    print_ram_usage(f"หลังรัน GoldAI (Sweep {idx+1})")
    ram_after = get_ram_usage_gb()[0]
    if max_ram_used > max_ram_overall:
        max_ram_overall = max_ram_used

    # Data Quality, Artifact, Outlier, Duplicate, Schema
    output_files = glob.glob(os.path.join(run_output_dir, "*.csv")) + glob.glob(os.path.join(run_output_dir, "*.json"))
    if not output_files:
        print(f"❌ No output files found in {run_output_dir}")
        continue
    # Outlier detection and QA summary
    qa_summary = ""
    for file in output_files:
        basename = os.path.basename(file)
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(file)
                ndup = df.duplicated().sum()
                if ndup > 0:
                    print(f"⚠️ {basename} has {ndup} duplicated rows")
                for col in ["pnl_usd_net", "drawdown", "slippage"]:
                    if col in df.columns:
                        mean, std = df[col].mean(), df[col].std()
                        z = (df[col] - mean) / std
                        outlier_idx = df.index[z.abs() > 4].tolist()
                        if outlier_idx:
                            print(f"⚠️ Outlier in {col}: idx={outlier_idx[:10]} (total {len(outlier_idx)})")
            except Exception as e:
                print(f"⚠️ DataQuality fail: {basename}: {e}")

        # LLM QA
        if file.endswith(".json"):
            with open(file, encoding='utf-8') as f:
                data = json.load(f)
            sample_text = str(data)[:6000]
            prompt = f"วิเคราะห์ run summary: {sample_text}"
        elif file.endswith(".csv"):
            with open(file, encoding='utf-8') as f:
                lines = f.readlines()
                N = 200
                if len(lines) > N*2:
                    sample_text = "".join(lines[:N]) + "\n... (cut) ...\n" + "".join(lines[-N:])
                else:
                    sample_text = "".join(lines)
            prompt = f"วิเคราะห์ trade log: {sample_text}"
        else:
            continue

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        review_text = model.generate_content(prompt).text
        print(f"\n=== Gemini QA Review ({basename}) ===\n", review_text)
        qa_summary += f"\n--- {basename} ---\n{review_text}"

    result = {
        "pipeline_id": pipeline_id,
        "param": param,
        "config_hash": config_hash(param),
        "output_dir": run_output_dir,
        "qa_summary": qa_summary,
        "max_ram_used": max_ram_used,
        "ram_before": ram_before,
        "ram_after": ram_after,
    }
    all_sweep_results.append(result)
    sweep_changelog.append(f"{time.ctime()} | pipeline_id: {pipeline_id} | param: {param} | max_ram: {max_ram_used:.2f}GB")

    with open(f"{run_output_dir}/qa_review.md", "w", encoding="utf-8") as f:
        f.write(qa_summary)

# ---- Export all sweep results (QA Table/Markdown/Changelog) ----
pd.DataFrame([
    {
        "pipeline_id": r["pipeline_id"],
        **{k: v for k, v in r["param"].items() if k != "pipeline_id"},
        "config_hash": r["config_hash"],
        "max_ram_used": r["max_ram_used"],
        "ram_before": r["ram_before"],
        "ram_after": r["ram_after"],
        "qa_summary": r["qa_summary"][:1000]
    } for r in all_sweep_results
]).to_csv(CSV_RESULT_PATH, index=False)
print(f"\n📊 Hyperparam Sweep QA summary saved: {CSV_RESULT_PATH}")

with open(MD_REPORT_PATH, "w", encoding="utf-8") as f:
    for r in all_sweep_results:
        f.write(f"## Pipeline {r['pipeline_id']} | Param: {r['param']}\n\n```\n{r['qa_summary']}\n```\n\n")
print(f"📄 Markdown report all sweep: {MD_REPORT_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    for r in all_sweep_results:
        f.write(f"\n\n=== {r['pipeline_id']} ({r['param']}) ===\n{r['qa_summary']}\n")
print(f"📄 TXT QA report all sweep: {REPORT_PATH}")

with open(CHANGELOG_PATH, "a") as f:
    for l in sweep_changelog:
        f.write(l + "\n")
print(f"🗒️ Changelog updated: {CHANGELOG_PATH}")

with open(RAM_PATH, "w") as f:
    f.write(f"Max RAM used (any sweep): {max_ram_overall:.2f} GB\n")
print(f"📝 Max RAM used (all sweep) saved to: {RAM_PATH}")

# QA Fail-Pattern
RISK_WORDS = ["max drawdown", "consecutive sl", "fail", "error", "critical", "risk", "drawdown > 30%"]
fail_flag = False
for r in all_sweep_results:
    for risk in RISK_WORDS:
        if risk.lower() in r["qa_summary"].lower():
            print(f"❌ [QA FAIL] {r['pipeline_id']} พบความเสี่ยง: {risk}")
            fail_flag = True
if fail_flag:
    print("❌ Sweep QA พบปัญหาเสี่ยง! (exit code 1 สำหรับ CI/CD)")
    sys.exit(1)
else:
    print("✅ Hyperparameter Sweep QA Passed ทุก run (exit code 0)")
