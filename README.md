# Gold AI

This project contains `gold_ai2025.py` and accompanying tests.

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
# Run full test suite with coverage
pytest -v --cov=gold_ai2025 --cov-report=term-missing
```

## Notes
- The helper `extend_safe_import_for_studio` used during tests is defined directly in `test_gold_ai.py` and no longer imported from a separate module.
