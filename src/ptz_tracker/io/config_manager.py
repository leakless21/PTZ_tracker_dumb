"""
Configuration management module for PTZ Camera Object Tracking System.

Handles loading, validation, and management of YAML configuration.
"""

import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class ConfigManager:
    """Manages application configuration loading and validation."""

    def __init__(self):
        """Initialize config manager."""
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self, config_path: str = "config.yaml") -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary, or None if error
        """
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info("Loaded config from {path}", path=config_path)
            return self.config
        except FileNotFoundError:
            logger.error("Config file not found: {path}", path=config_path)
            return None
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML: {err}", err=e)
            return None

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(description="PTZ Object Tracker")
        parser.add_argument(
            "--config", default="config.yaml", help="Path to configuration file"
        )
        parser.add_argument("--input", help="Input video file")
        parser.add_argument("--output", help="Output video file")
        parser.add_argument("--tracker", choices=["KCF", "CSRT"], help="Tracker type")
        parser.add_argument(
            "--no-display", action="store_true", help="Disable display window"
        )
        return parser.parse_args()

    def apply_arguments(self, args: argparse.Namespace) -> None:
        """Apply command-line arguments to configuration.

        Args:
            args: Parsed command-line arguments
        """
        if self.config is None:
            return

        # Override with command-line args
        if args.input:
            self.config["video"]["input_path"] = args.input
        if args.output:
            self.config["video"]["output_path"] = args.output
        if args.tracker:
            self.config["tracking"]["tracker"] = args.tracker
        if args.no_display:
            self.config["display"]["show_window"] = False

    def validate_config(self) -> bool:
        """Validate configuration values.

        Returns:
            True if valid, False otherwise
        """
        if self.config is None:
            logger.error("No configuration loaded")
            return False

        errors = []

        # Check video input
        input_path = self.config["video"]["input_path"]
        if not Path(input_path).exists():
            errors.append(f"Video input file not found: {input_path}")

        # Check algorithm
        algo = self.config["background_subtraction"]["algorithm"]
        if algo not in ["MOG2", "KNN"]:
            errors.append(
                f"background_subtraction.algorithm must be 'MOG2' or 'KNN', got: {algo}"
            )

        # Check tracker
        tracker = self.config["tracking"]["tracker"]
        if tracker not in ["KCF", "CSRT"]:
            errors.append(f"tracking.tracker must be 'KCF' or 'CSRT', got: {tracker}")

        # Check numeric ranges
        if (
            self.config["ptz_control"]["target_object_size"] < 0.1
            or self.config["ptz_control"]["target_object_size"] > 0.9
        ):
            errors.append("ptz_control.target_object_size should be between 0.1 and 0.9")

        if errors:
            for error in errors:
                logger.error(error)
            return False

        return True

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration.

        Returns:
            Configuration dictionary
        """
        return self.config