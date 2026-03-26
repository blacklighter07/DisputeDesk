from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


from dispute_desk.server.app import app as _app
from dispute_desk.server.app import main as _inner_main

app = _app


def main() -> None:
    _inner_main()


if __name__ == "__main__":
    main()
