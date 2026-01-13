"""
Example configuration file for tessera tests.

Copy this file to config.py and update the paths to match your system:
    cp config.example.py config.py

Then edit config.py with your local paths.
"""

from pathlib import Path

# Base path to GADGET-4 simulation output
SNAPSHOT_BASE = Path("/path/to/your/simulation/gadget4/output")

# Base path to gotetra reference outputs (for validation tests)
GOTETRA_BASE = Path("/path/to/gotetra/outputs")

# Path to original ORIGAMI binary (for ORIGAMI validation tests)
# Build from: https://github.com/BettyMcFly/origami
ORIGAMI_BIN = Path("/path/to/origami/code/origamitag")

# Halo centers for validation tests (comoving coordinates in Mpc/h)
HALO_CENTERS = {
    'snapshot_034': (128.0, 128.0, 128.0),
    'snapshot_074': (128.0, 128.0, 128.0),
}
