# PTZ Camera Object Tracking System

A real-time object tracking system using background subtraction and simulated PTZ (Pan-Tilt-Zoom) camera control.

## Features

- **Dual-Mode Tracking System**:
  - **Detection Mode (Norfair)**: Multi-object tracking when no object selected
  - **Locked Mode (CSRT)**: High-accuracy single-object tracking when locked
- **Background Subtraction**: OpenCV (MOG2, KNN) or BGSLibrary (43+ algorithms)
- **Object Detection**: Contour-based detection with filtering
- **Virtual PTZ**: Simulated pan, tilt, and zoom on video files
- **Real-time Visualization**: Multi-object boxes (cyan) in detection mode, locked object (green)
- **Debug Mosaic**: 2×4 grid showing all pipeline stages
- **Keyboard Selection**: Press numeric keys (0-9) to select and lock onto tracked objects by ID
- **Configurable**: YAML-based configuration system

## Quick Start with Pixi

### Prerequisites

Install Pixi (modern package manager):

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd PTZ_tracker_dumb

# Install dependencies with Pixi
pixi install
```

That's it! Pixi handles all dependencies automatically.

### Running the Application

```bash
# Run with default settings
pixi run track

# Run with custom configuration
pixi run track-config

# Run with specific video files
pixi run python main.py --input video.mp4 --output result.mp4

# Enter development shell
pixi shell
```

### Check Installation

```bash
pixi run version
```

Expected output:
```
Python 3.x.x
OpenCV: 4.x.x
NumPy: 1.x.x
```

## Configuration

Edit `config.yaml` to customize:

- Background subtraction algorithm (OpenCV or BGSLibrary)
- Object detection parameters
- PTZ control sensitivity
- Display options

See `TECHNICAL_SPECIFICATIONS.md` for detailed parameter descriptions.

## Project Structure

```
PTZ_tracker_dumb/
├── pyproject.toml          # Pixi configuration (recommended)
├── pixi.toml              # Alternative minimal config
├── config.yaml            # Application configuration
├── main.py                # Main application (to be implemented)
├── TECHNICAL_SPECIFICATIONS.md  # Complete technical specs
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## Development

### Install with Development Tools

```bash
# Install with visualization and dev tools
pixi install --environment dev

# This adds: ipython, pytest, matplotlib, pandas
```

### Development Workflow

```bash
# Enter Pixi shell
pixi shell

# Run application
python main.py

# Interactive Python (IPython)
ipython

# Run tests
pytest tests/

# Exit shell
exit
```

### Managing Dependencies

```bash
# Add a package from conda-forge
pixi add scipy

# Add a PyPI package
pixi add --pypi scikit-image

# Remove a package
pixi remove scipy

# Update all packages
pixi update

# List installed packages
pixi list
```

## Environment Options

Pixi supports multiple environments:

- **default**: Core dependencies (opencv, numpy, pybgs)
- **viz**: Adds matplotlib and pandas for analysis
- **dev**: Adds development tools (ipython, pytest)

```bash
# Use visualization environment
pixi run --environment viz python analyze.py

# Use development environment
pixi install --environment dev
```

## Legacy pip Installation

If Pixi is not available:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install opencv-python numpy pybgs pyyaml

# Run application
python main.py
```

## Documentation

- **TECHNICAL_SPECIFICATIONS.md**: Complete implementation guide
  - Detailed OpenCV/BGSLibrary API specifications
  - Algorithm descriptions and parameters
  - Complete implementation workflow
  - Performance optimization tips

## License

MIT License

## Contributing

See `TECHNICAL_SPECIFICATIONS.md` for implementation details and coding guidelines.
