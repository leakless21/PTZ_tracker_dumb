#!/usr/bin/env python3
"""
PTZ Camera Object Tracking System - Entry Point

This is the main entry point for the PTZ Tracker application.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ptz_tracker.main import main

if __name__ == "__main__":
    main()
