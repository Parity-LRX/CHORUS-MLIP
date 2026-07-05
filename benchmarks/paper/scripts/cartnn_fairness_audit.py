#!/usr/bin/env python
"""Compatibility wrapper for the operator fairness audit script."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "operator" / "cartnn_fairness_audit.py"
    runpy.run_path(str(target), run_name="__main__")
