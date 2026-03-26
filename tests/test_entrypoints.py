import subprocess
import sys
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parents[2]


def test_wrapper_imports_cleanly_from_parent_directory():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "module = importlib.import_module('dispute_desk_env.server.app'); "
                "print(module.app.title)"
            ),
        ],
        cwd=PROJECT_PARENT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DisputeDesk Environment API" in result.stdout
