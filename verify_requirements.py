import importlib
import os
import sys

REQ_FILE = os.path.join(os.path.dirname(__file__), 'requirements.txt')

missing = []
if os.path.exists(REQ_FILE):
    with open(REQ_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pkg = line.split('==')[0]
            pkg = pkg.split('[')[0]
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)
else:
    print(f"Requirements file not found: {REQ_FILE}")
    sys.exit(1)

if missing:
    print('Missing packages:', ', '.join(missing))
    sys.exit(1)
else:
    print('All required packages are installed.')
