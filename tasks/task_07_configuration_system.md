# Task 7: Configuration System

**Phase:** 3 - Recovery & Configuration
**Duration:** 1 day
**Priority:** Medium
**Dependencies:** All previous tasks (provides configuration to all modules)

---

## Overview

Implement a YAML-based configuration system that allows users to customize all aspects of the tracking system without modifying code. Includes loading, validation, defaults, and error handling.

---

## Implementation Details

### 7.1 Configuration File Structure

**Objective:** Organize parameters into logical sections

**File: config.yaml**

```yaml
# Video I/O
video:
  input: "input.mp4"
  output: "output.mp4"
  output_codec: "mp4v"
  save_output: true
  loop_playback: false

# Background Subtraction
background_subtraction:
  algorithm: "MOG2"  # MOG2 or KNN
  history: 500
  learning_rate: -1  # -1 for automatic
  var_threshold: 16  # MOG2 only
  detect_shadows: true

# Object Detection
object_detection:
  min_area: 500
  max_area_fraction: 0.5
  kernel_size: 5  # 3, 5, or 7
  threshold_value: 127

# Tracking
tracking:
  tracker: "KCF"  # KCF or CSRT

  # Norfair multi-object tracking
  norfair:
    distance_threshold: 50
    hit_counter_max: 10
    initialization_delay: 3

  # Object selection
  selection:
    min_track_age: 5

  # Validation
  validation:
    min_bbox_area: 100
    max_bbox_area_fraction: 0.8

  # Recovery
  recovery:
    search_radius: 150
    size_similarity_threshold: 0.5
    timeout: 3.0

# PTZ Control
ptz:
  pan_sensitivity: 45.0
  tilt_sensitivity: 30.0
  zoom_min: 1.0
  zoom_max: 5.0
  deadband: 0.05
  target_object_size: 0.3

# Display
display:
  show_window: true
  show_info: true
  show_debug_mosaic: true
  window_name: "PTZ Tracker"

  debug_mosaic:
    tile_width: 320
    tile_height: 240
```

### 7.2 Configuration Loading

**Objective:** Read YAML file and parse into Python dictionary

**Implementation:**
```
import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
```

**Error Handling:**
- File not found: Use default config or exit with error
- Parse error: Show line number and description
- Invalid structure: Validate schema

### 7.3 Default Configuration

**Objective:** Provide fallback values if config missing or incomplete

**Implementation:**
```
def get_default_config():
    """Return default configuration dictionary"""
    return {
        'video': {
            'input': 'input.mp4',
            'output': 'output.mp4',
            'output_codec': 'mp4v',
            'save_output': True,
            'loop_playback': False
        },
        'background_subtraction': {
            'algorithm': 'MOG2',
            'history': 500,
            'learning_rate': -1,
            'var_threshold': 16,
            'detect_shadows': True
        },
        # ... rest of defaults ...
    }
```

**Usage:**
```
config = load_config('config.yaml')
if config is None:
    print("Using default configuration")
    config = get_default_config()
```

### 7.4 Configuration Validation

**Objective:** Ensure all values are within acceptable ranges

**Validation Rules:**
- **video.input**: Must be valid file path (existence check)
- **background_subtraction.algorithm**: Must be "MOG2" or "KNN"
- **background_subtraction.history**: Integer, 1-2000
- **object_detection.min_area**: Integer, > 0
- **object_detection.max_area_fraction**: Float, 0.0-1.0
- **tracking.tracker**: Must be "KCF" or "CSRT"
- **ptz.deadband**: Float, 0.0-0.5
- **ptz.zoom_min**: Float, >= 1.0
- **ptz.zoom_max**: Float, > zoom_min

**Implementation:**
```
def validate_config(config):
    """Validate configuration values"""
    errors = []

    # Check video input
    if not Path(config['video']['input']).exists():
        errors.append(f"Video input file not found: {config['video']['input']}")

    # Check algorithm
    if config['background_subtraction']['algorithm'] not in ['MOG2', 'KNN']:
        errors.append("background_subtraction.algorithm must be 'MOG2' or 'KNN'")

    # Check numeric ranges
    if config['ptz']['deadband'] < 0 or config['ptz']['deadband'] > 0.5:
        errors.append("ptz.deadband must be between 0.0 and 0.5")

    # ... more validations ...

    return errors
```

**Action on Validation Failure:**
- Print all errors
- Either exit or use defaults for invalid values
- Log warnings for non-critical issues

### 7.5 Configuration Access

**Objective:** Provide convenient access to nested configuration values

**Option 1: Direct Dictionary Access**
```
min_area = config['object_detection']['min_area']
```

**Option 2: Config Class (Optional)**
```
class Config:
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, path, default=None):
        """Get nested value with dot notation"""
        keys = path.split('.')
        value = self._config
        for key in keys:
            if key not in value:
                return default
            value = value[key]
        return value

# Usage
config = Config(config_dict)
min_area = config.get('object_detection.min_area', 500)
```

### 7.6 Command-Line Arguments

**Objective:** Allow overriding config values from command line

**Implementation:**
```
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='PTZ Object Tracker')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', help='Input video file')
    parser.add_argument('--output', help='Output video file')
    parser.add_argument('--tracker', choices=['KCF', 'CSRT'],
                       help='Tracker type')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    return parser.parse_args()

# Usage
args = parse_arguments()
config = load_config(args.config)

# Override with command-line args
if args.input:
    config['video']['input'] = args.input
if args.output:
    config['video']['output'] = args.output
if args.tracker:
    config['tracking']['tracker'] = args.tracker
if args.no_display:
    config['display']['show_window'] = False
```

### 7.7 Configuration Presets

**Objective:** Provide preset configurations for common scenarios

**Preset Files:**
- `config_fast.yaml`: Optimized for speed (KCF, low resolution)
- `config_accurate.yaml`: Optimized for accuracy (CSRT, high thresholds)
- `config_indoor.yaml`: Indoor settings (KNN, tight filtering)
- `config_outdoor.yaml`: Outdoor settings (MOG2, shadow detection)

**Usage:**
```
python main.py --config config_fast.yaml
```

### 7.8 Configuration Documentation

**Objective:** Help users understand what each parameter does

**In-file Comments:**
```yaml
ptz:
  # Sensitivity controls how quickly PTZ responds to object position
  # Higher = faster response but more jitter
  # Typical range: 20-60
  pan_sensitivity: 45.0

  # Deadband prevents micro-adjustments when object near center
  # Value is fraction of frame dimension (0.05 = 5%)
  # Higher = less jitter but less precision
  deadband: 0.05
```

**Separate Documentation:**
- README section explaining parameters
- TECHNICAL_SPECIFICATIONS.md with detailed descriptions
- Example configs with comments

---

## Test Scenarios

### Test 7.1: Load Valid Config
- **Scenario:** Load config.yaml with all valid values
- **Expected Result:** Config loaded successfully
- **Validation:** All parameters accessible

### Test 7.2: Missing Config File
- **Scenario:** Config file doesn't exist
- **Expected Result:** Error message, use defaults or exit
- **Validation:** Graceful handling, clear error message

### Test 7.3: Invalid YAML Syntax
- **Scenario:** Config has syntax error (e.g., bad indentation)
- **Expected Result:** Parse error with line number
- **Validation:** User can locate and fix error

### Test 7.4: Missing Parameters
- **Scenario:** Config missing some sections (e.g., no ptz section)
- **Expected Result:** Use defaults for missing values
- **Validation:** System runs with partial config

### Test 7.5: Invalid Parameter Values
- **Scenario:** deadband = 2.0 (out of range 0.0-0.5)
- **Expected Result:** Validation error, use default or exit
- **Validation:** Clear error message explaining valid range

### Test 7.6: Invalid Video Path
- **Scenario:** video.input points to non-existent file
- **Expected Result:** Validation error on load
- **Validation:** Error before attempting video capture

### Test 7.7: Command-Line Override
- **Scenario:** python main.py --input test.mp4
- **Expected Result:** Uses test.mp4 instead of config value
- **Validation:** Command-line takes precedence

### Test 7.8: Preset Configuration
- **Scenario:** Load config_fast.yaml preset
- **Expected Result:** Fast tracking (KCF, optimized settings)
- **Validation:** Parameters match preset intent

### Test 7.9: Type Mismatch
- **Scenario:** min_area: "five hundred" (string instead of int)
- **Expected Result:** Type validation error or conversion
- **Validation:** Clear error or automatic type conversion

### Test 7.10: Nested Access
- **Scenario:** Access config['ptz']['deadband']
- **Expected Result:** Returns 0.05
- **Validation:** Nested dictionaries work correctly

### Test 7.11: Default Fallback
- **Scenario:** Access non-existent parameter
- **Expected Result:** Returns None or default value
- **Validation:** No KeyError exceptions

### Test 7.12: Configuration Reload
- **Scenario:** Modify config.yaml and reload
- **Expected Result:** New values loaded
- **Validation:** Changes take effect (for hot-reload feature, post-MVP)

---

## Caveats

### YAML Limitations
- **Type Inference**: YAML may infer types incorrectly (e.g., 01 becomes int 1, not string "01")
- **Special Characters**: Certain characters in strings may require quoting
- **Indentation Sensitive**: Spaces vs tabs can cause silent errors

### Validation Challenges
- **Interdependent Parameters**: Some params depend on others (e.g., zoom_max > zoom_min)
- **Context-Dependent Validity**: Valid ranges may depend on video resolution
- **Exhaustive Validation**: Hard to validate every possible combination

### Default Configuration
- **Staleness**: Defaults in code may drift from config.yaml examples
- **Discovery**: Users may not know all available parameters
- **Override Precedence**: Complex precedence (defaults < config < command-line)

### Performance Considerations
- **Loading Overhead**: YAML parsing adds startup time (negligible for small configs)
- **Hot Reload**: Reloading during runtime requires careful state management
- **Validation Cost**: Extensive validation adds startup delay

### User Experience
- **Error Messages**: Need to be clear and actionable
- **Parameter Discovery**: Users may not know what parameters exist
- **Valid Ranges**: Need documentation for acceptable values
- **Preset Confusion**: Too many presets overwhelm users

### Edge Cases
- **Empty Config**: Empty file or all values commented out
- **Unicode**: File paths with non-ASCII characters
- **Platform Differences**: Windows vs Linux path separators
- **Permissions**: Config file not readable

---

## Success Criteria

✅ Config file loads successfully with valid YAML
✅ All parameters accessible via dictionary or class
✅ Missing config file handled gracefully (defaults or error)
✅ Invalid values caught by validation
✅ Clear error messages for common issues
✅ Command-line arguments override config values
✅ Default configuration provided for all parameters
✅ Video input path validated before processing
✅ Numeric ranges validated (e.g., 0.0-1.0 for fractions)
✅ Enum values validated (e.g., "KCF" or "CSRT" only)
✅ Configuration documented with comments
✅ Example presets provided for common scenarios

---

## Dependencies

**Python Libraries:**
- pyyaml >= 5.0
- argparse (standard library)
- pathlib (standard library)

**Previous Tasks:**
- All tasks use configuration values

**Configuration File:**
- config.yaml (this task creates it)

---

## Integration Notes

**Inputs:**
- config.yaml file
- Command-line arguments (optional)

**Outputs:**
- Configuration dictionary
- Validation errors (if any)

**Used By:**
- All modules: Every task uses config values
- main.py: Loads config at startup, passes to modules

**Best Practices:**
- Load once at startup, pass to all modules
- Don't reload during runtime (MVP), add hot-reload post-MVP
- Validate early (before video processing starts)

---

## Estimated Effort

- YAML Loading: 2-3 hours
- Default Configuration: 2-3 hours
- Validation: 3-4 hours
- Command-Line Arguments: 2-3 hours
- Documentation: 2-3 hours
- Preset Files: 2-3 hours
- Testing: 2-3 hours
- **Total: 1-1.5 days**
