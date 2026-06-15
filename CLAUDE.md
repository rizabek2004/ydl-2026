# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an early-stage Python learning repository. Work is organized into per-day folders (`day_1/`, and presumably `day_2/`, etc. as the project grows). Each folder holds standalone scripts; there is no package, build system, or test suite yet.

## Environment

- A virtualenv lives in `.venv/`. Activate it before running anything: `source .venv/bin/activate`.
- Note a version mismatch to be aware of: `.venv/pyvenv.cfg` was created with Python 3.13.5, but the active interpreter on this machine reports 3.12.12. If you hit interpreter issues, check which `python` resolves first.
- The venv currently has only `pip` installed — no third-party dependencies, and no `requirements.txt`/`pyproject.toml`/`setup.py`.

## Running

Scripts are run directly, e.g.:

```bash
python day_1/hello.py
```

`day_1/hello.py` reads from stdin (`input(...)`), so it expects interactive input when run.

## Security note

`day_1/creds.txt` contains credential-like key/value lines and is **not** gitignored (the repo is not yet a git repository). Do not commit this file or echo its contents. If git is initialized here, add `creds.txt` (or `*.txt`) and `.venv/` to `.gitignore` first.
