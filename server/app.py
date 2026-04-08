"""
server/app.py — Multi-mode deployment entry point required by OpenEnv.

This module re-exports the FastAPI `app` instance and provides a `main()`
function so the environment can be launched in multiple modes:
  - Docker:  CMD in Dockerfile starts uvicorn directly
  - pip:     `server` console script calls main() from pyproject.toml
  - import:  other tools can do `from server.app import app`
"""

from app.main import app, main

__all__ = ["app", "main"]
