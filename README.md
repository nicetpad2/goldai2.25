# Gold AI

Enterprise-grade, auditable, and robust AI system for XAUUSD algorithmic trading and research.

This repository includes:
- `gold_ai2025.py` – Main Gold AI algorithm, backtesting engine, risk/OMS logic, robust type/format guards, and enterprise patch protocols.
- `test_gold_ai.py` – Comprehensive unit tests, edge path coverage, dynamic mocking for core dependencies, and full QA validation.

---

## 🚀 Installation

```bash
pip install -r requirements.txt
🧪 Running Tests
# Run the full test suite with branch coverage reporting
pytest -v --cov=gold_ai2025 --cov-report=term-missing
📝 Project Notes
Patch Protocol:
All logic patches and critical changes must log their version (e.g., [Patch AI Studio v4.9.42+]) in code and test logs per AGENTS.md.
The latest patch `[Patch AI Studio v4.9.54+]` adds `TradeSimResult` for `simulate_trades` outputs and further improves walk-forward result handling.

Type/Format Guards:
Use only _isinstance_safe and _float_fmt as enforced by QA for all dynamic type or format operations.

Testing:
The helper extend_safe_import_for_studio (used for dynamic import/mocking during tests) is now defined directly in test_gold_ai.py for full auditability and is no longer imported from a separate module.

Test Requirements:
All merges/patches must pass pytest with >90% coverage and must exercise all edge, fail, and error paths (see AGENTS.md).

File Policy:
Only the latest gold_ai2025.py and test_gold_ai.py are used for QA and patching. No legacy/test artifacts are referenced.

📄 Documentation
AGENTS.md:
Agent roles, patch protocol, QA standards, and testing requirements for enterprise CI/CD.

CHANGELOG.md:
(If present) Version and patch history.

Inline Docstrings:
All functions and patches are documented with version and audit comments as required.

💡 Contact & Contributions
All core logic or patch PRs must follow patch-based diff, be reviewed by QA, and update AGENTS.md if agent/process logic changes.

For issues, QA, or enterprise integration, contact the AI Studio QA/Dev Team.


