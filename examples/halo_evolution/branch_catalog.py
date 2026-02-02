"""
Branch Catalog Reader

Fast random access to pre-extracted main progenitor branches.
Enables efficient SLURM array jobs without loading full merger trees.

Usage:
    catalog = BranchCatalog('branch_catalog.h5')

    # Get a specific branch
    branch = catalog.get_branch(tree_idx=0)  # Most massive halo

    # Iterate over mass range
    for branch in catalog.iter_branches(mass_min=100.0, mass_max=1000.0):
        print(f"Tree {branch.tree_id}: M200c = {branch.root_m200c:.2e}")

    # Get branches for array job
    task_branches = catalog.get_branches_for_task(task_id=5, n_tasks=100)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union
import numpy as np
import h5py


@dataclass
class Branch:
    """
    Main progenitor branch for a single tree.

    Attributes
    ----------
    tree_id : int
        Original tree identifier in merger tree file
    tree_idx : int
        Index in this catalog (0 = most massive)
    root_m200c : float
        Root halo M200c [10^10 Msun/h]
    root_r200c : float
        Root halo R200c [Mpc/h]
    root_snap : int
        Root halo snapshot number
    root_subhalo : int
        Root halo subhalo index
    a_min : float
        Earliest scale factor in branch
    a_max : float
        Latest scale factor in branch
    snap_num : np.ndarray
        Snapshot numbers (shape: n_halos)
    subhalo_nr : np.ndarray
        Subhalo indices (shape: n_halos)
    scale_factor : np.ndarray
        Scale factors (shape: n_halos)
    position : np.ndarray
        Comoving positions [Mpc/h] (shape: n_halos, 3)
    velocity : np.ndarray
        Velocities [km/s] (shape: n_halos, 3)
    m200c : np.ndarray
        M200 critical [10^10 Msun/h] (shape: n_halos)
    r200c : np.ndarray
        R200 critical [Mpc/h] (shape: n_halos)
    mass : np.ndarray
        Subhalo mass [10^10 Msun/h] (shape: n_halos)
    """
    tree_id: int
    tree_idx: int
    root_m200c: float
    root_r200c: float
    root_snap: int
    root_subhalo: int
    a_min: float
    a_max: float
    snap_num: np.ndarray
    subhalo_nr: np.ndarray
    scale_factor: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    m200c: np.ndarray
    r200c: np.ndarray
    mass: np.ndarray

    def __len__(self) -> int:
        """Number of halos in branch."""
        return len(self.snap_num)

    def __repr__(self) -> str:
        return (
            f"Branch(tree_id={self.tree_id}, n_halos={len(self)}, "
            f"M200c={self.root_m200c:.2e}, a=[{self.a_min:.3f}, {self.a_max:.3f}])"
        )


class BranchCatalog:
    """
    Reader for pre-extracted branch catalog files.

    Provides fast random access to main progenitor branches without
    loading the full merger tree.

    Parameters
    ----------
    catalog_file : str or Path
        Path to branch_catalog.h5 file

    Attributes
    ----------
    n_trees : int
        Number of trees in catalog
    n_total_halos : int
        Total number of halos across all branches
    box_size : float
        Simulation box size [Mpc/h]
    hubble : float
        Hubble parameter h
    snap_times : np.ndarray
        Scale factors for each snapshot
    snap_redshifts : np.ndarray
        Redshifts for each snapshot

    Examples
    --------
    >>> catalog = BranchCatalog('branch_catalog.h5')
    >>> print(f"Catalog has {catalog.n_trees} trees")
    >>>
    >>> # Get most massive tree's branch
    >>> branch = catalog.get_branch(0)
    >>> print(f"Branch has {len(branch)} halos")
    >>>
    >>> # Iterate over mass range
    >>> for branch in catalog.iter_branches(mass_min=100.0):
    ...     print(f"M200c = {branch.root_m200c:.2e}")
    """

    def __init__(self, catalog_file: Union[str, Path]):
        self.catalog_file = Path(catalog_file)
        if not self.catalog_file.exists():
            raise FileNotFoundError(f"Catalog file not found: {self.catalog_file}")

        # Load header and index (always in memory)
        self._load_header_and_index()

    def _load_header_and_index(self) -> None:
        """Load header and index arrays into memory."""
        with h5py.File(self.catalog_file, 'r') as f:
            # Header
            header = f['Header']
            self.n_trees = int(header.attrs['n_trees'])
            self.n_total_halos = int(header.attrs['n_total_halos'])
            self.source_file = str(header.attrs['source_file'])
            self.box_size = float(header.attrs['box_size'])
            self.hubble = float(header.attrs['hubble'])
            self.omega_matter = float(header.attrs['omega_matter'])
            self.omega_lambda = float(header.attrs['omega_lambda'])
            self.n_snapshots = int(header.attrs['n_snapshots'])
            self.last_snap = int(header.attrs['last_snap'])

            # Optional filter info
            self.mass_min_filter = header.attrs.get('mass_min_filter', None)
            self.a_min_filter = header.attrs.get('a_min_filter', None)

            # Snapshot times
            self.snap_times = f['TreeTimes/Time'][:]
            self.snap_redshifts = f['TreeTimes/Redshift'][:]

            # Branch index (always loaded)
            idx = f['BranchIndex']
            self._tree_id = idx['tree_id'][:]
            self._start_offset = idx['start_offset'][:]
            self._length = idx['length'][:]
            self._root_snap = idx['root_snap'][:]
            self._root_subhalo = idx['root_subhalo'][:]
            self._root_m200c = idx['root_m200c'][:]
            self._root_r200c = idx['root_r200c'][:]
            self._a_min = idx['a_min'][:]
            self._a_max = idx['a_max'][:]

    def get_branch(self, tree_idx: int) -> Branch:
        """
        Get a branch by catalog index.

        Parameters
        ----------
        tree_idx : int
            Index in catalog (0 = most massive, sorted by root M200c)

        Returns
        -------
        branch : Branch
            Branch data for this tree
        """
        if tree_idx < 0 or tree_idx >= self.n_trees:
            raise IndexError(f"tree_idx must be 0 to {self.n_trees - 1}, got {tree_idx}")

        start = self._start_offset[tree_idx]
        length = self._length[tree_idx]
        end = start + length

        if length == 0:
            # Empty branch
            return Branch(
                tree_id=int(self._tree_id[tree_idx]),
                tree_idx=tree_idx,
                root_m200c=float(self._root_m200c[tree_idx]),
                root_r200c=float(self._root_r200c[tree_idx]),
                root_snap=int(self._root_snap[tree_idx]),
                root_subhalo=int(self._root_subhalo[tree_idx]),
                a_min=0.0,
                a_max=0.0,
                snap_num=np.array([], dtype=np.int32),
                subhalo_nr=np.array([], dtype=np.int32),
                scale_factor=np.array([], dtype=np.float32),
                position=np.array([], dtype=np.float32).reshape(0, 3),
                velocity=np.array([], dtype=np.float32).reshape(0, 3),
                m200c=np.array([], dtype=np.float32),
                r200c=np.array([], dtype=np.float32),
                mass=np.array([], dtype=np.float32),
            )

        # Load data from file
        with h5py.File(self.catalog_file, 'r') as f:
            data = f['BranchData']
            return Branch(
                tree_id=int(self._tree_id[tree_idx]),
                tree_idx=tree_idx,
                root_m200c=float(self._root_m200c[tree_idx]),
                root_r200c=float(self._root_r200c[tree_idx]),
                root_snap=int(self._root_snap[tree_idx]),
                root_subhalo=int(self._root_subhalo[tree_idx]),
                a_min=float(self._a_min[tree_idx]),
                a_max=float(self._a_max[tree_idx]),
                snap_num=data['snap_num'][start:end],
                subhalo_nr=data['subhalo_nr'][start:end],
                scale_factor=data['scale_factor'][start:end],
                position=data['position'][start:end],
                velocity=data['velocity'][start:end],
                m200c=data['m200c'][start:end],
                r200c=data['r200c'][start:end],
                mass=data['mass'][start:end],
            )

    def get_branch_by_tree_id(self, tree_id: int) -> Branch:
        """
        Get a branch by original merger tree ID.

        Parameters
        ----------
        tree_id : int
            Original tree identifier from merger tree file

        Returns
        -------
        branch : Branch
            Branch data for this tree

        Raises
        ------
        ValueError
            If tree_id not found in catalog
        """
        matches = np.where(self._tree_id == tree_id)[0]
        if len(matches) == 0:
            raise ValueError(f"Tree ID {tree_id} not found in catalog")
        return self.get_branch(int(matches[0]))

    def get_indices_in_mass_range(
        self,
        mass_min: Optional[float] = None,
        mass_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get catalog indices for trees in a mass range.

        Parameters
        ----------
        mass_min : float, optional
            Minimum root M200c [10^10 Msun/h]
        mass_max : float, optional
            Maximum root M200c [10^10 Msun/h]

        Returns
        -------
        indices : np.ndarray
            Array of catalog indices matching the criteria
        """
        mask = np.ones(self.n_trees, dtype=bool)

        if mass_min is not None:
            mask &= self._root_m200c >= mass_min
        if mass_max is not None:
            mask &= self._root_m200c <= mass_max

        return np.where(mask)[0]

    def iter_branches(
        self,
        mass_min: Optional[float] = None,
        mass_max: Optional[float] = None,
        indices: Optional[np.ndarray] = None,
    ) -> Iterator[Branch]:
        """
        Iterate over branches matching criteria.

        Parameters
        ----------
        mass_min : float, optional
            Minimum root M200c [10^10 Msun/h]
        mass_max : float, optional
            Maximum root M200c [10^10 Msun/h]
        indices : np.ndarray, optional
            Specific indices to iterate over (overrides mass filters)

        Yields
        ------
        branch : Branch
            Branch data for each matching tree
        """
        if indices is None:
            indices = self.get_indices_in_mass_range(mass_min, mass_max)

        for idx in indices:
            yield self.get_branch(idx)

    def get_branches_for_task(
        self,
        task_id: int,
        n_tasks: int,
        mass_min: Optional[float] = None,
        mass_max: Optional[float] = None,
    ) -> List[int]:
        """
        Get branch indices for a SLURM array task.

        Divides the catalog evenly among tasks, with earlier tasks
        getting slightly more if not evenly divisible.

        Parameters
        ----------
        task_id : int
            SLURM_ARRAY_TASK_ID (0-indexed)
        n_tasks : int
            Total number of array tasks
        mass_min : float, optional
            Minimum root M200c [10^10 Msun/h]
        mass_max : float, optional
            Maximum root M200c [10^10 Msun/h]

        Returns
        -------
        indices : list of int
            Catalog indices for this task to process
        """
        all_indices = self.get_indices_in_mass_range(mass_min, mass_max)
        n_total = len(all_indices)

        if n_total == 0:
            return []

        # Divide evenly
        base_count = n_total // n_tasks
        remainder = n_total % n_tasks

        # Earlier tasks get one extra if there's a remainder
        if task_id < remainder:
            start = task_id * (base_count + 1)
            count = base_count + 1
        else:
            start = remainder * (base_count + 1) + (task_id - remainder) * base_count
            count = base_count

        return list(all_indices[start:start + count])

    def get_scale_factor(self, snap_num: int) -> float:
        """Get scale factor for a snapshot number."""
        return float(self.snap_times[snap_num])

    def get_redshift(self, snap_num: int) -> float:
        """Get redshift for a snapshot number."""
        return float(self.snap_redshifts[snap_num])

    def summary(self) -> str:
        """Return a summary string of catalog contents."""
        lines = [
            f"BranchCatalog: {self.catalog_file.name}",
            f"  Trees: {self.n_trees:,}",
            f"  Total halos: {self.n_total_halos:,}",
            f"  Box size: {self.box_size:.1f} Mpc/h",
            f"  M200c range: [{self._root_m200c.min():.2e}, {self._root_m200c.max():.2e}] 10^10 Msun/h",
            f"  Scale factor range: [{self._a_min[self._a_min > 0].min():.4f}, {self._a_max.max():.4f}]",
        ]
        if self.mass_min_filter is not None:
            lines.append(f"  Mass filter applied: M200c >= {self.mass_min_filter:.2e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BranchCatalog('{self.catalog_file.name}', "
            f"n_trees={self.n_trees:,}, n_halos={self.n_total_halos:,})"
        )

    def __len__(self) -> int:
        return self.n_trees

    def __getitem__(self, idx: int) -> Branch:
        return self.get_branch(idx)


def unwrap_branch_coordinates(branch: Branch, box_size: float) -> Branch:
    """
    Unwrap periodic coordinates for smooth tracking.

    Modifies positions in-place and returns the branch.

    Parameters
    ----------
    branch : Branch
        Branch to unwrap (modified in-place)
    box_size : float
        Periodic box size [Mpc/h]

    Returns
    -------
    branch : Branch
        Same branch with unwrapped positions
    """
    if len(branch) < 2:
        return branch

    half_box = box_size / 2.0

    for i in range(1, len(branch)):
        diff = branch.position[i] - branch.position[i - 1]

        for dim in range(3):
            if diff[dim] > half_box:
                branch.position[i:, dim] -= box_size
            elif diff[dim] < -half_box:
                branch.position[i:, dim] += box_size

    return branch
