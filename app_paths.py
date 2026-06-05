"""
Single source of truth for where the app reads/writes its data files.

WHY THIS EXISTS:
    When packaged with PyInstaller, every module's __file__ points *inside* the
    bundle (a temp dir or the _internal folder), not next to the .exe. If each
    module computed its own Path(__file__).parent they would resolve to
    different, wrong folders. Routing every path through DATA_DIR keeps the whole
    app pointed at ONE canonical folder:
        - normal run  -> this file's directory (the project folder)
        - frozen .exe -> the folder containing the .exe (persistent, user-visible)
"""

import sys
from pathlib import Path


def _data_dir() -> Path:
    if getattr(sys, "frozen", False):          # running inside a PyInstaller bundle
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


DATA_DIR = _data_dir()
