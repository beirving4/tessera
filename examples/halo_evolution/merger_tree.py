"""
GADGET-4 Merger Tree Reader

Efficient, lazy-loading reader for GADGET-4 merger tree HDF5 files.
Supports both single-file and distributed (split) tree formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import h5py
import numpy as np


def discover_tree_files(directory: Path) -> List[Path]:
    """
    Discover split tree files in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing trees.*.hdf5 files

    Returns
    -------
    files : List[Path]
        List of tree files sorted by file index
    """
    pattern = re.compile(r'^trees\.(\d+)\.hdf5$')
    indexed_files = []

    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            idx = int(match.group(1))
            indexed_files.append((idx, path))

    # Sort by file index
    indexed_files.sort(key=lambda x: x[0])

    return [path for _, path in indexed_files]


class MergerTree:
    """
    Reader for GADGET-4 merger tree files.

    Supports lazy loading - only reads arrays when accessed.
    Handles both single-file (trees.hdf5) and distributed (trees.*.hdf5) formats.

    Parameters
    ----------
    tree_path : str or Path
        Path to either:
        - A single trees.hdf5 file
        - A directory containing trees.*.hdf5 files

    Attributes
    ----------
    n_trees : int
        Number of trees (z=0 halos)
    n_halos : int
        Total number of halos across all trees
    n_snapshots : int
        Number of snapshots in the simulation
    snap_times : np.ndarray
        Scale factor at each snapshot
    snap_redshifts : np.ndarray
        Redshift at each snapshot
    box_size : float
        Simulation box size in Mpc/h
    hubble : float
        Hubble parameter h
    omega_matter : float
        Matter density parameter
    omega_lambda : float
        Dark energy density parameter

    Examples
    --------
    >>> # Single file
    >>> tree = MergerTree('trees.hdf5')
    >>> print(f"Found {tree.n_trees} trees")
    >>>
    >>> # Distributed files
    >>> tree = MergerTree('/path/to/treedata/')
    >>> print(f"Found {tree.n_trees} trees from {tree.n_files} files")
    >>>
    >>> # Get tree table info
    >>> tree_lengths = tree.tree_table['Length']
    >>>
    >>> # Access halo properties (lazy loaded)
    >>> snap_nums = tree.tree_halos['SnapNum']
    >>> positions = tree.tree_halos['SubhaloPos']
    >>> main_prog = tree.tree_halos['TreeMainProgenitor']
    """

    def __init__(self, tree_path: Union[str, Path]):
        tree_path = Path(tree_path)

        # Determine if single file or directory
        if tree_path.is_file():
            # Single file mode
            self._files = [tree_path]
            self._distributed = False
        elif tree_path.is_dir():
            # Distributed mode - find all tree files
            self._files = discover_tree_files(tree_path)
            if not self._files:
                raise FileNotFoundError(
                    f"No tree files (trees.*.hdf5) found in directory: {tree_path}"
                )
            self._distributed = True
        else:
            raise FileNotFoundError(f"Tree path not found: {tree_path}")

        # Store the primary path for display
        self.tree_file = self._files[0] if self._distributed else tree_path

        # Cache for lazy-loaded arrays
        self._tree_table_cache: Optional[Dict[str, np.ndarray]] = None
        self._tree_halos_cache: Optional[Dict[str, np.ndarray]] = None

        # Load header info immediately (small)
        self._load_header()

    def _load_header(self) -> None:
        """Load header information from the tree file(s)."""
        # Always read header from first file
        with h5py.File(self._files[0], 'r') as f:
            # Header info
            header = f['Header']
            self.n_trees = int(header.attrs['Ntrees_Total'])
            self.n_halos = int(header.attrs['Nhalos_Total'])
            self.n_files = int(header.attrs['NumFiles'])
            self.last_snap = int(header.attrs['LastSnapShotNr'])

            # Time info
            self.snap_times = f['TreeTimes/Time'][:]
            self.snap_redshifts = f['TreeTimes/Redshift'][:]
            self.n_snapshots = len(self.snap_times)

            # Cosmological parameters
            params = f['Parameters']
            self.box_size = float(params.attrs['BoxSize'])
            self.hubble = float(params.attrs['HubbleParam'])
            self.omega_matter = float(params.attrs['Omega0'])
            self.omega_lambda = float(params.attrs['OmegaLambda'])

    @property
    def tree_table(self) -> Dict[str, np.ndarray]:
        """
        Lazy-load TreeTable datasets.

        Returns dict with keys: 'Length', 'StartOffset', 'TreeID'

        For distributed files, StartOffset values are adjusted to be
        global indices into the concatenated TreeHalos arrays.
        """
        if self._tree_table_cache is not None:
            return self._tree_table_cache

        if not self._distributed:
            # Single file - simple load
            self._tree_table_cache = {}
            with h5py.File(self._files[0], 'r') as f:
                for key in f['TreeTable'].keys():
                    self._tree_table_cache[key] = f['TreeTable'][key][:]
        else:
            # Distributed format: TreeTable entries are only in files where
            # Ntrees_ThisFile > 0. StartOffset values are already global indices
            # into the concatenated TreeHalos array.
            all_lengths = []
            all_start_offsets = []
            all_tree_ids = []

            for file_path in self._files:
                with h5py.File(file_path, 'r') as f:
                    n_trees_this_file = int(f['Header'].attrs['Ntrees_ThisFile'])
                    if n_trees_this_file == 0:
                        continue  # No TreeTable in this file

                    all_lengths.append(f['TreeTable']['Length'][:])
                    all_start_offsets.append(f['TreeTable']['StartOffset'][:])
                    all_tree_ids.append(f['TreeTable']['TreeID'][:])

            self._tree_table_cache = {
                'Length': np.concatenate(all_lengths),
                'StartOffset': np.concatenate(all_start_offsets),
                'TreeID': np.concatenate(all_tree_ids),
            }

        return self._tree_table_cache

    @property
    def tree_halos(self) -> Dict[str, np.ndarray]:
        """
        Lazy-load TreeHalos datasets.

        Returns dict with all halo properties and linking arrays.
        Note: Only loads essential fields to save memory.

        For distributed files, arrays are concatenated from all files.
        """
        if self._tree_halos_cache is not None:
            return self._tree_halos_cache

        # Essential fields to load
        essential_fields = [
            'SnapNum', 'SubhaloNr', 'GroupNr', 'TreeID', 'TreeIndex',
            'SubhaloPos', 'SubhaloVel', 'SubhaloMass',
            'Group_M_Crit200', 'Group_R_Crit200',
            'TreeMainProgenitor', 'TreeDescendant',
            'TreeFirstProgenitor', 'TreeNextProgenitor',
        ]

        if not self._distributed:
            # Single file - simple load
            self._tree_halos_cache = {}
            with h5py.File(self._files[0], 'r') as f:
                for key in essential_fields:
                    if key in f['TreeHalos']:
                        self._tree_halos_cache[key] = f['TreeHalos'][key][:]
        else:
            # Distributed - concatenate from all files
            # First pass: determine which fields exist
            with h5py.File(self._files[0], 'r') as f:
                available_fields = [k for k in essential_fields if k in f['TreeHalos']]

            # Collect data from all files
            file_data = {field: [] for field in available_fields}

            for file_path in self._files:
                with h5py.File(file_path, 'r') as f:
                    for field in available_fields:
                        file_data[field].append(f['TreeHalos'][field][:])

            # Concatenate
            self._tree_halos_cache = {
                field: np.concatenate(arrays)
                for field, arrays in file_data.items()
            }

        return self._tree_halos_cache

    def clear_cache(self) -> None:
        """Clear cached data to free memory."""
        self._tree_table_cache = None
        self._tree_halos_cache = None

    def get_tree_indices(self, tree_id: int) -> np.ndarray:
        """
        Get TreeHalos indices for a specific tree.

        Parameters
        ----------
        tree_id : int
            Tree identifier (0 to n_trees-1)

        Returns
        -------
        indices : np.ndarray
            Array of indices into TreeHalos arrays for this tree
        """
        if tree_id < 0 or tree_id >= self.n_trees:
            raise ValueError(f"tree_id must be 0 to {self.n_trees-1}, got {tree_id}")

        start = self.tree_table['StartOffset'][tree_id]
        length = self.tree_table['Length'][tree_id]
        return np.arange(start, start + length)

    def get_root_halo_index(self, tree_id: int) -> int:
        """
        Get TreeHalos index of the root (final snapshot) halo for a tree.

        The root halo is typically the first halo in each tree at the
        final snapshot.

        Parameters
        ----------
        tree_id : int
            Tree identifier (0 to n_trees-1)

        Returns
        -------
        index : int
            Index into TreeHalos arrays
        """
        return int(self.tree_table['StartOffset'][tree_id])

    def find_halo_in_tree(
        self,
        snap_num: int,
        subhalo_nr: int,
        tree_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Find TreeHalos index for a given snapshot/subhalo combination.

        Parameters
        ----------
        snap_num : int
            Snapshot number
        subhalo_nr : int
            Subhalo index at that snapshot
        tree_id : int, optional
            If provided, only search within this tree

        Returns
        -------
        index : int or None
            TreeHalos index, or None if not found
        """
        snap_nums = self.tree_halos['SnapNum']
        subhalo_nrs = self.tree_halos['SubhaloNr']

        if tree_id is not None:
            # Search only within specified tree
            indices = self.get_tree_indices(tree_id)
            mask = (snap_nums[indices] == snap_num) & (subhalo_nrs[indices] == subhalo_nr)
            matches = indices[mask]
        else:
            # Search all halos
            mask = (snap_nums == snap_num) & (subhalo_nrs == subhalo_nr)
            matches = np.where(mask)[0]

        return None if len(matches) == 0 else int(matches[0])

    def get_scale_factor(self, snap_num: int) -> float:
        """Get scale factor for a snapshot number."""
        return float(self.snap_times[snap_num])

    def get_redshift(self, snap_num: int) -> float:
        """Get redshift for a snapshot number."""
        return float(self.snap_redshifts[snap_num])

    def snap_num_to_index(self, snap_num: int) -> int:
        """Convert snapshot number to index in snap_times array."""
        # snap_times is indexed by snapshot number
        return snap_num

    def find_nearest_snapshot(self, scale_factor: float) -> int:
        """
        Find snapshot number nearest to given scale factor.

        Parameters
        ----------
        scale_factor : float
            Target scale factor

        Returns
        -------
        snap_num : int
            Nearest snapshot number
        """
        idx = np.argmin(np.abs(self.snap_times - scale_factor))
        return int(idx)

    def __repr__(self) -> str:
        if self._distributed:
            return (
                f"MergerTree('{self.tree_file.parent.name}/', "
                f"n_trees={self.n_trees}, n_halos={self.n_halos}, "
                f"n_files={self.n_files}, "
                f"a=[{self.snap_times.min():.3f}, {self.snap_times.max():.3f}])"
            )
        return (
            f"MergerTree('{self.tree_file.name}', "
            f"n_trees={self.n_trees}, n_halos={self.n_halos}, "
            f"a=[{self.snap_times.min():.3f}, {self.snap_times.max():.3f}])"
        )
