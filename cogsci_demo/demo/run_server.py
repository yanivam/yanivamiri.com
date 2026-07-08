#!/usr/bin/env python3
"""Launch booth demo web server.

Run from anywhere:
  python demo/run_server.py
  cd demo && python run_server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "demo.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "demo"), str(PROJECT_ROOT / "src")],
    )


if __name__ == "__main__":
    main()
