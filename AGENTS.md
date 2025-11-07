# PTZ Tracker – Engineering Guide for Agents

This AGENTS.md applies to the entire repository (root scope). It captures how we build and modify the PTZ Camera Object Tracking System. Keep the core principles; tailor decisions to this Python computer‑vision project.

---

## Language, Tools, and Commands

- Python: 3.13 (per `pixi.toml`)
- Style: PEP 8, type hints on public functions, concise docstrings
- Env: Pixi recommended
  - Run: `pixi run start`
  - Dev shell: `pixi shell`
  - Version check: `pixi run version`
- Optional pip path is fine for users, but do not commit pip-only scripts.
- Logging: Loguru (console + rotating file)

---

## Core Principles (Project‑Adapted)

### Single Responsibility (SRP)

Each module/class has one reason to change.

- Keep state machine and I/O in `main.py`; keep CV logic in `tracker.py`
- Put PTZ math and ROI in `ptz.py` only
- Keep visualization helpers in `debug_view.py`
- Validation/parsing lives with config‑loading utilities, not in processing loops
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
- Imports: standard lib, third‑party, local — in that order
- File size targets (soft): `main.py`~250, `tracker.py`~300, `ptz.py`~100, `debug_view.py`~100

---

## Testing and Validation

- Start with targeted tests near helpers (e.g., PTZ math, bbox validation)
- If `pytest` exists, use it; otherwise keep simple assertions in ad‑hoc scripts during MVP
- Manual validation: verify state transitions, FPS targets, and overlay semantics match specs

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

## Quick Reference

- Run default: `pixi run track`
- Run with config: `pixi run track-config`
- Enter shell: `pixi shell`
- Colors: detection cyan, tracking green, lost red
- Keys: 0–9 select, R reset, D debug mosaic, Space pause, Q/ESC quit

---

## Summary

Keep it simple, focused, and measurable. Preserve three states, and core visual semantics. Extend by composition and config, not rewrites. Good code here is simple, clear, and purposeful.
