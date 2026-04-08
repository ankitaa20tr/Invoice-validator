"""
server/app.py — Multi-mode deployment entry point required by OpenEnv.

Provides:
  - `app`    : the FastAPI application instance
  - `main()` : callable server launcher (used by pyproject.toml [project.scripts])
"""

import uvicorn
from app.main import app  # re-export FastAPI app


def main():
    """Launch the uvicorn server. Entry point for multi-mode deployment."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
