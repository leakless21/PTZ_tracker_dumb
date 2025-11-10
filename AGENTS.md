# PTZ Tracker – Engineering Guide for Agents

This AGENTS.md applies to the entire repository (root scope). It captures how we build and modify the PTZ Camera Object Tracking System. Keep the core principles; tailor decisions to this Python computer‑vision project.

---

## Language, Tools, and Commands

- Python: 3.13+ (per `pyproject.toml` and `pixi.toml`)
- Style: PEP 8, type hints on public functions, concise docstrings
- Package structure: src-layout (setuptools-compliant)
- Env: Pixi recommended
  - Run: `pixi run track` or `pixi run track-config`
  - Dev shell: `pixi shell`
  - Version check: `pixi run version`
  - Tests: `pixi run test`
- Optional pip path is fine for users, but do not commit pip-only scripts.
- Logging: Loguru (console + rotating file, config in `setup_logging()` function in `main.py`)

---

## Core Principles (Project‑Adapted)

### Single Responsibility (SRP)

Each module/class has one reason to change.

- **[main.py](src/ptz_tracker/main.py)**: Application orchestration, state machine, keyboard input, logging, video loop
- **[core/tracker.py](src/ptz_tracker/core/tracker.py)**: Object detection (background subtraction, contours), single-object tracking state machine
- **[core/ptz.py](src/ptz_tracker/core/ptz.py)**: PTZ control math, ROI extraction, pan/tilt/zoom calculations
- **[io/video_io.py](src/ptz_tracker/io/video_io.py)**: Video input/output capture and writing
- **[io/config_manager.py](src/ptz_tracker/io/config_manager.py)**: YAML configuration loading and parameter validation
- **[ui/app_state.py](src/ptz_tracker/ui/app_state.py)**: Application state container and management
- **[ui/input_handler.py](src/ptz_tracker/ui/input_handler.py)**: Keyboard event processing and command mapping
- **[ui/debug_view.py](src/ptz_tracker/ui/debug_view.py)**: Visualization helpers, debug mosaic, overlay drawing
- Validation/parsing lives with config‑loading utilities and at module boundaries, not in processing loops
- Benefit: simpler tests, predictable changes, easier performance tuning

### Open/Closed (OCP)

Open to extension via composition/strategy; closed to risky edits.

- Add new trackers or filters by new functions/classes and selection in config rather than editing existing logic
- Example: new background subtraction preset ⇒ add a creator function; do not rewrite call sites
- Benefit: safer iteration under MVP timelines

### Liskov Substitution (LSP)

Swappable components must honor expectations.

- If swapping `KCF` with `CSRT`, both must expose `init(frame, bbox)` and `update(frame) -> (ok, bbox)` semantics and valid bbox constraints
- Do not introduce behavior that requires caller special‑casing one tracker
- Benefit: predictable state transitions, fewer edge bugs

### Interface Segregation (ISP)

Keep interfaces small and focused.

- Separate responsibilities: detection vs. single‑object tracking vs. PTZ control
- Utilities should not expose unrelated knobs; prefer small helpers (e.g., `clean_mask`, `is_valid_bbox`)
- Benefit: targeted tests and easier refactors

### Dependency Inversion (DIP)

Depend on abstractions/config, not concretes.

- Pass in configuration and creators (factory functions) for trackers/BS models; avoid globally constructed singletons
- Keep `tracker.py` free from direct I/O assumptions; return data needed by `main.py`
- Benefit: easier mocking, reproducible runs

---

## DRY (Don’t Repeat Yourself)

- Extract repeated drawing (boxes, labels, status) into small helpers
- Centralize mask cleaning and bbox validation
- Reuse config accessors with defaults; avoid scattering magic numbers
- Avoid duplicate keyboard handling; centralize key map in `main.py`

---

## KISS (Keep It Simple)

- Keep functions short and intention‑revealing; descriptive names over comments
- Limit parameters exposed to those in `config.yaml`; sane defaults over elaborate tuning
- Avoid premature threading/async; measure first
- Don't create unnecessary documentation layers

---

## YAGNI (You Aren’t Gonna Need It)

- No deep learning detectors, ONVIF, or RTSP in MVP
- Use Loguru for logging; keep setup minimal (console + rotating file)
- No plugin systems or DI frameworks; simple factories/config flags suffice
- Don’t add new states, files, or layers unless required by current success criteria

---

## Coding Standards

- Naming: `snake_case` for functions/vars, `PascalCase` for classes, module‑level constants in `UPPER_SNAKE`
- Types: annotate public functions; return `Tuple[bool, Tuple[int,int,int,int]]` for trackers
- Errors: validate bboxes and bounds; fail fast with clear messages
- Imports: standard lib, third‑party, local — in that order, use relative imports within package
- File size targets (soft): `main.py`~450, `tracker.py`~600, `ptz.py`~200, `debug_view.py`~250
- Line length: 100 characters (per `pyproject.toml` black config)
- Format with black and isort before committing (both configured in `pyproject.toml`)

---

## Testing and Validation

- Use `pytest` (configured in `pyproject.toml`) for all automated tests
- Test structure: `tests/unit/` for isolated unit tests, `tests/integration/` for component integration tests
- Start with targeted tests for helpers (e.g., PTZ math in `core/ptz.py`, bbox validation, contour filtering)
- Test trackers (KCF vs CSRT) and background subtraction algorithms (MOG2 vs KNN) with varied inputs
- Manual validation: verify state transitions (DETECTION → TRACKING → LOST), FPS targets, overlay semantics
- Run `pixi run test` before committing changes

---

## Performance Guardrails

- Targets: 720p ≥30 FPS; 1080p ≥20 FPS
- Avoid unnecessary copies/conversions; prefer in‑place OpenCV ops
- Batch small drawing operations where reasonable
- Use `KCF` when FPS dips; make it a config switch, not hardcoded

---

## Configuration Conventions

- Only parameters described in `TECHNICAL_SPECIFICATIONS.md`/`config.yaml`
- Defaults must be safe and produce visible output without edits
- Validation: clamp values, log warnings, don’t crash on minor config issues

---

## Git and Changes

- Small, focused commits with imperative messages
- Do not rename core files without updating docs and tasks
- Update `README.md` and `TECHNICAL_SPECIFICATIONS.md` when behavior or parameters change

---

## Project Structure

```plaintext
src/ptz_tracker/          # Main package (src-layout)
├── __init__.py           # Package metadata (v0.1.0)
├── main.py               # Application entry point (~463 lines)
├── core/                 # Core tracking logic
│   ├── tracker.py        # Detection & tracking (~607 lines)
│   └── ptz.py            # PTZ calculations (~194 lines)
├── io/                   # Input/Output operations
│   ├── video_io.py       # Video capture/writing (~144 lines)
│   └── config_manager.py # YAML config loading (~127 lines)
└── ui/                   # User interface layer
    ├── app_state.py      # Application state (~97 lines)
    ├── input_handler.py  # Keyboard input (~104 lines)
    └── debug_view.py     # Visualization (~248 lines)

tests/                    # Test directory (pytest)
├── unit/                 # Unit tests
└── integration/          # Integration tests

run.py                    # Entry point script (adds src/ to path)
config.yaml               # Runtime configuration
pyproject.toml           # Package metadata & tool config
pixi.toml                # Pixi task definitions
```

## Quick Reference

- **Run default**: `pixi run track`
- **Run with config**: `pixi run track-config`
- **Enter shell**: `pixi shell`
- **Run tests**: `pixi run test`
- **Check versions**: `pixi run version`
- **Visual indicators**: Detection (cyan), tracking (green), lost (red)
- **Keyboard controls**: 0–9 select, R reset, D debug mosaic, Space pause, Q/ESC quit

---

## Summary

Keep it simple, focused, and measurable. Preserve three states, and core visual semantics. Extend by composition and config, not rewrites. Good code here is simple, clear, and purposeful.
