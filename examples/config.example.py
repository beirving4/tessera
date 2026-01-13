"""
Example configuration file for tessera examples.

Copy this file to config.py and update the paths to match your system:
    cp config.example.py config.py

Then edit config.py with your local paths.
"""

from pathlib import Path

# Base path to GADGET-4 simulation output
# Update this to point to your simulation data
SNAPSHOT_BASE = Path("/path/to/your/simulation/gadget4/output")

# Example halo centers (comoving coordinates in Mpc/h)
# These are specific to each simulation - update for your data
HALO_CENTERS = {
    'snapshot_034': (128.0, 128.0, 128.0),  # Example center
    'snapshot_074': (128.0, 128.0, 128.0),  # Example center
}
