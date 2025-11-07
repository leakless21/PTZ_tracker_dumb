# PTZ Camera Object Tracking System```markdown

# PTZ Camera Object Tracking System

A real-time object tracking system using background subtraction and simulated PTZ (Pan-Tilt-Zoom) camera control.

A real-time object tracking system using background subtraction and simulated PTZ (Pan-Tilt-Zoom) camera control.

## Features

## Features

- **Three-State Tracking**: Detection (multi-object) → Tracking (single-object) → Lost (recovery) → Detection

- **Background Subtraction**: OpenCV MOG2/KNN foreground detection with adaptive morphological filtering- **Three-State Tracking**: Detection (multi-object) → Tracking (single-object) → Lost (recovery) → Detection

- **Multi-Object Detection**: Contour-based detection with ID persistence using Norfair tracker- **Background Subtraction**: OpenCV MOG2/KNN foreground detection with adaptive morphological filtering

- **Single-Object Tracking**: High-accuracy CSRT/KCF tracker for selected objects- **Multi-Object Detection**: Contour-based detection with ID persistence using Norfair tracker

- **Virtual PTZ Control**: Simulated pan, tilt, and zoom to keep object centered in frame- **Single-Object Tracking**: High-accuracy CSRT/KCF tracker for selected objects

- **Visual Feedback**: Cyan boxes (detection), green boxes (tracking), red search area (recovery)- **Virtual PTZ Control**: Simulated pan, tilt, and zoom to keep object centered in frame

- **Debug Mosaic**: 2×4 grid visualization showing 8 pipeline stages- **Visual Feedback**: Cyan boxes (detection), green boxes (tracking), red search area (recovery)

- **Keyboard Control**: Select objects by ID (0-9), reset, toggle debug, pause, quit- **Debug Mosaic**: 2×4 grid visualization showing 8 pipeline stages

- **YAML Configuration**: Fully configurable parameters with sensible defaults- **Keyboard Control**: Select objects by ID (0-9), reset, toggle debug, pause, quit

- **Loguru Logging**: Console + rotating file logs for debugging and field diagnostics- **YAML Configuration**: Fully configurable parameters with sensible defaults

- **Loguru Logging**: Console + rotating file logs for debugging and field diagnostics

## Quick Start with Pixi

## Quick Start with Pixi

### Prerequisites

### Prerequisites

Install Pixi (modern package manager):

Install Pixi (modern package manager):

```bash

# Linux/macOS```bash

curl -fsSL https://pixi.sh/install.sh | bash# Linux/macOS

curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)

iwr -useb https://pixi.sh/install.ps1 | iex# Windows (PowerShell)

```iwr -useb https://pixi.sh/install.ps1 | iex

```

### Installation

### Installation

```bash

# Clone repository```bash

git clone <repository-url># Clone repository

cd PTZ_tracker_dumbgit clone <repository-url>

cd PTZ_tracker_dumb

# Install dependencies with Pixi

pixi install# Install dependencies with Pixi

```pixi install

```

### Running the Application

That's it! Pixi handles all dependencies automatically.

```bash

# Run with default settings### Running the Application

pixi run track

```bash

# Run with custom video# Run with default settings

pixi run python main.py --input video.mp4 --output result.mp4pixi run track



# Enter development shell# Run with custom configuration

pixi shellpixi run track-config

python main.py

```# Run with specific video files

pixi run python main.py --input video.mp4 --output result.mp4

### Verify Installation

# Enter development shell

```bashpixi shell

pixi run version```

```

### Logging

Expected output:

```- Uses Loguru for logging by default.

Python 3.13.x- Outputs to colorized console and a rotating file at `logs/ptz_tracker.log` (10 MB, keep 5 files).

OpenCV: 4.x.x- No configuration required for MVP; defaults are safe.

NumPy: 2.x.x

```### Check Installation



## System Architecture```bash

pixi run version

**~750 lines across 4 modules + configuration**```



| Module | Lines | Purpose |Expected output:

|--------|-------|---------|```

| **main.py** | ~250 | Video I/O, main loop, state machine, keyboard input |Python 3.x.x

| **tracker.py** | ~300 | Background subtraction, Norfair & CSRT/KCF tracking |OpenCV: 4.x.x

| **ptz.py** | ~100 | PTZ calculations and ROI extraction |NumPy: 1.x.x

| **debug_view.py** | ~100 | Debug mosaic visualization (2×4 grid) |```

| **config.yaml** | ~50 | Configuration parameters |

## Configuration

## State Machine

Edit `config.yaml` to customize:

```

        ┌──────────────┐- Background subtraction algorithm (OpenCV or BGSLibrary)

   ┌───▶│  DETECTION   │◀────────┐- Object detection parameters

   │    │ (Multi-obj)  │         │- PTZ control sensitivity

   │    └──────┬───────┘         │- Display options

   │           │                 │

   │    Press 0-9                │See `TECHNICAL_SPECIFICATIONS.md` for detailed parameter descriptions.

   │           │                 │

   │           ▼                 │## Project Structure

   │    ┌──────────────┐         │

   │    │   TRACKING   │         │```

   │    │ (Single-obj) │         │PTZ_tracker_dumb/

   │    └──────┬───────┘         │├── pyproject.toml          # Pixi configuration (recommended)

   │           │                 │├── pixi.toml              # Alternative minimal config

   │    Tracker fails            │ Timeout├── config.yaml            # Application configuration

   │           │                 │ or press R├── main.py                # Main application (to be implemented)

   │           ▼                 │├── TECHNICAL_SPECIFICATIONS.md  # Complete technical specs

   │    ┌──────────────┐         │├── README.md              # This file

   └────│     LOST     │─────────┘└── .gitignore             # Git ignore rules

        │  (Recovery)  │```

        └──────────────┘

```## Development



### States### Install with Development Tools



**DETECTION**: Multi-object mode (Norfair tracker)```bash

- Shows all moving objects with cyan boxes and persistent IDs# Install with visualization and dev tools

- Press 0-9 to select object and enter TRACKING modepixi install --environment dev



**TRACKING**: Single-object mode (CSRT/KCF tracker)# This adds: ipython, pytest, matplotlib, pandas

- Shows locked object with green box```

- PTZ control keeps object centered

- Returns to DETECTION on tracking failure### Development Workflow



**LOST**: Recovery mode (background subtraction)```bash

- Attempts to redetect object near last position# Enter Pixi shell

- Returns to DETECTION on timeout (3 seconds)pixi shell



## Keyboard Controls# Run application

python main.py

| Key | Action |

|-----|--------|# Interactive Python (IPython)

| **0-9** | Select object by ID |ipython

| **R** | Reset to DETECTION mode |

| **D** | Toggle debug mosaic |# Run tests

| **Space** | Pause/Resume |pytest tests/

| **Q / ESC** | Quit |

# Exit shell

## Configurationexit

```

Edit `config.yaml` to customize:

### Managing Dependencies

- Background subtraction algorithm (MOG2 or KNN)

- Detection thresholds (area, aspect ratio)```bash

- Tracking parameters (distance threshold, hit counter)# Add a package from conda-forge

- PTZ control (pan/tilt sensitivity, zoom limits, deadband)pixi add scipy

- Display options (tile size, debug mosaic)

# Add a PyPI package

See `TECHNICAL_SPECIFICATIONS.md` for detailed parameter descriptions.pixi add --pypi scikit-image



## Project Structure# Add Loguru (logging)

pixi add loguru  # or: pixi add --pypi loguru

```

PTZ_tracker_dumb/# Remove a package

├── main.py                        # Main application looppixi remove scipy

├── tracker.py                     # Background subtraction & tracking

├── ptz.py                         # PTZ control calculations# Update all packages

├── debug_view.py                  # Debug mosaic visualizationpixi update

├── config.yaml                    # Configuration parameters

├── requirements.txt               # Python dependencies# List installed packages

├── pixi.toml                      # Pixi environment configpixi list

├── TECHNICAL_SPECIFICATIONS.md    # Complete technical docs```

├── PROJECT_PLAN.md                # Implementation roadmap

├── AGENTS.md                      # Engineering guidelines## Environment Options

├── README.md                      # This file

└── UAV_videos/                    # Test video filesPixi supports multiple environments:

```

- **default**: Core dependencies (opencv, numpy, pybgs)

## Performance Targets- **viz**: Adds matplotlib and pandas for analysis

- **dev**: Adds development tools (ipython, pytest)

- **720p (1280×720)**: ≥30 FPS

- **1080p (1920×1080)**: ≥20 FPS```bash

- **Latency**: <50ms (frame to display)# Use visualization environment

- **Memory**: <500 MBpixi run --environment viz python analyze.py

- **CPU**: <80% of single core

# Use development environment

## Dependenciespixi install --environment dev

```

- **opencv-python** ≥4.5: Video I/O, image processing, tracking

- **numpy** ≥1.19: Numerical operations## Legacy pip Installation

- **norfair** ≥2.0: Multi-object tracking with ID persistence

- **pyyaml** ≥5.0: Configuration file parsingIf Pixi is not available:

- **loguru** ≥0.7: Logging (console + rotating file)

```bash

## Installation Methods# Create virtual environment

python -m venv venv

### With Pixi (Recommended)source venv/bin/activate  # Linux/macOS

# or: venv\Scripts\activate  # Windows

```bash

pixi install# Install dependencies

pixi run trackpip install opencv-python numpy pybgs pyyaml loguru

```

# Run application

### With pippython main.py

```

```bash

python -m venv venv## Documentation

source venv/bin/activate

pip install -r requirements.txt- **TECHNICAL_SPECIFICATIONS.md**: Complete implementation guide

python main.py  - Detailed OpenCV/BGSLibrary API specifications

```  - Algorithm descriptions and parameters

  - Complete implementation workflow

## Debug Mosaic (2×4 Grid)  - Performance optimization tips

  - Logging guidance (Loguru defaults and usage)

Press **D** to toggle during runtime:

## License

```

┌─────────────┬─────────────┬─────────────┬─────────────┐MIT License

│ 1. Original │ 2. FG Mask  │ 3. Cleaned  │ 4. Contours │

│             │    (Raw)    │    Mask     │             │## Contributing

├─────────────┼─────────────┼─────────────┼─────────────┤

│ 5. Norfair  │ 6. CSRT     │ 7. PTZ ROI  │ 8. Final    │See `TECHNICAL_SPECIFICATIONS.md` for implementation details and coding guidelines.

│  Detection  │  Tracking   │   Overlay   │   Output    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## Logging

Logs are written to `logs/ptz_tracker.log`:
- **Rotation**: 10 MB per file
- **Retention**: 5 files
- **Console**: Colorized INFO level
- **File**: Detailed diagnostic messages

No configuration required; sensible defaults work out-of-box.

## Success Criteria

✅ Loads and processes video files  
✅ Detects multiple moving objects  
✅ Assigns persistent IDs to objects  
✅ User can select objects by ID (0-9)  
✅ Tracks selected object with CSRT/KCF  
✅ Virtual PTZ keeps object centered  
✅ Debug mosaic shows pipeline stages  
✅ Recovers from tracking loss  
✅ Achieves target frame rates  
✅ Clean, maintainable codebase  

## Documentation

- **TECHNICAL_SPECIFICATIONS.md**: Complete API reference, algorithms, implementation details
- **PROJECT_PLAN.md**: Implementation timeline and task breakdown
- **AGENTS.md**: Coding guidelines and engineering principles

## Future Enhancements

- Deep learning detection (YOLO)
- Physical PTZ camera support (ONVIF)
- Real-time camera streams (RTSP)
- Multiple simultaneous object tracking
- Trajectory analysis and heatmaps

## License

MIT License

## Contributing

Follow guidelines in `AGENTS.md` for coding standards, architecture principles, and contribution workflow.
