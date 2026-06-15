# Project metadata lives in pyproject.toml. This shim exists only so legacy
# `pip install -e .` invocations still work; prefer `uv sync` / `uv pip install -e .`.
from setuptools import setup

setup()
