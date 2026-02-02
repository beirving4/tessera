"""
Halo Branch Tracing

Trace progenitor and descendant branches through merger trees.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Handle imports for both package and standalone usage
try:
    from .merger_tree import MergerTree
except ImportError:
    from merger_tree import MergerTree


@dataclass
class HaloInfo:
    """
    Information about a halo at one snapshot.

    Attributes
    ----------
    tree_index : int
        Index in TreeHalos arrays
    snap_num : int
        Snapshot number
    subhalo_nr : int
        Subhalo index in catalog
    group_nr : int
        FOF group index
    scale_factor : float
        Scale factor a = 1/(1+z)
    redshift : float
        Redshift z
    position : np.ndarray
        (3,) comoving position [Mpc/h]
    velocity : np.ndarray
        (3,) velocity [km/s]
    mass : float
        Subhalo mass [Msun/h] (in code units, typically 10^10 Msun/h)
    m200c : float
        M200c [Msun/h] (in code units)
    r200c : float
        R200c [Mpc/h]
    """
    tree_index: int
    snap_num: int
    subhalo_nr: int
    group_nr: int
    scale_factor: float
    redshift: float
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    m200c: float
    r200c: float


class HaloTracker:
    """
    Trace halo evolution through merger tree.

    Parameters
    ----------
    tree : MergerTree
        Merger tree reader instance

    Examples
    --------
    >>> tree = MergerTree('trees.hdf5')
    >>> tracker = HaloTracker(tree)
    >>>
    >>> # Trace from the most massive z=0 halo (tree 0, subhalo 0)
    >>> branch = tracker.trace_main_branch(
    ...     snap_num=74,      # Final snapshot
    ...     subhalo_nr=0,     # Most massive subhalo
    ...     backward=True,    # Trace progenitors
    ...     forward=False     # Don't trace descendants (already at final)
    ... )
    >>>
    >>> print(f"Branch spans {len(branch)} snapshots")
    >>> for halo in branch:
    ...     print(f"  a={halo.scale_factor:.3f}: M200={halo.m200c:.2e}")
    """

    def __init__(self, tree: MergerTree):
        self.tree = tree
        # Cache for tree start offsets (for converting relative to absolute indices)
        self._start_offsets = self.tree.tree_table['StartOffset']

    def _get_tree_start(self, absolute_idx: int) -> int:
        """Get the start offset for the tree containing this halo."""
        tree_id = int(self.tree.tree_halos['TreeID'][absolute_idx])
        return int(self._start_offsets[tree_id])

    def _relative_to_absolute(self, relative_idx: int, tree_start: int) -> int:
        """Convert relative tree index to absolute TreeHalos index."""
        return -1 if relative_idx == -1 else tree_start + relative_idx

    def _make_halo_info(self, tree_index: int) -> HaloInfo:
        """Create HaloInfo from TreeHalos index."""
        halos = self.tree.tree_halos

        snap_num = int(halos['SnapNum'][tree_index])

        return HaloInfo(
            tree_index=tree_index,
            snap_num=snap_num,
            subhalo_nr=int(halos['SubhaloNr'][tree_index]),
            group_nr=int(halos['GroupNr'][tree_index]),
            scale_factor=self.tree.get_scale_factor(snap_num),
            redshift=self.tree.get_redshift(snap_num),
            position=halos['SubhaloPos'][tree_index].copy(),
            velocity=halos['SubhaloVel'][tree_index].copy(),
            mass=float(halos['SubhaloMass'][tree_index]),
            m200c=float(halos['Group_M_Crit200'][tree_index]),
            r200c=float(halos['Group_R_Crit200'][tree_index]),
        )

    def trace_main_branch(
        self,
        snap_num: int,
        subhalo_nr: int,
        backward: bool = True,
        forward: bool = True,
        a_min: Optional[float] = None,
        a_max: Optional[float] = None,
    ) -> List[HaloInfo]:
        """
        Trace the main branch through a halo.

        Parameters
        ----------
        snap_num : int
            Reference snapshot number
        subhalo_nr : int
            Subhalo index at reference snapshot
        backward : bool
            Trace progenitors (earlier times)
        forward : bool
            Trace descendants (later times)
        a_min, a_max : float, optional
            Limit tracing to this scale factor range

        Returns
        -------
        branch : List[HaloInfo]
            Halo info at each snapshot, sorted by scale factor (early to late)
        """
        # Find the starting halo
        start_idx = self.tree.find_halo_in_tree(snap_num, subhalo_nr)
        if start_idx is None:
            return []

        halos = self.tree.tree_halos
        # Get tree start offset for converting relative indices
        tree_start = self._get_tree_start(start_idx)

        branch_indices = [start_idx]

        # Trace backward (progenitors)
        if backward:
            current = start_idx
            main_prog = halos['TreeMainProgenitor']
            while True:
                # TreeMainProgenitor stores relative indices within the tree
                prog_rel = main_prog[current]
                prog_idx = self._relative_to_absolute(prog_rel, tree_start)
                if prog_idx == -1:
                    break

                # Check scale factor limit
                if a_min is not None:
                    prog_snap = halos['SnapNum'][prog_idx]
                    prog_a = self.tree.get_scale_factor(prog_snap)
                    if prog_a < a_min:
                        break

                branch_indices.append(prog_idx)
                current = prog_idx

        # Trace forward (descendants)
        if forward:
            current = start_idx
            descendant = halos['TreeDescendant']

            while True:
                # TreeDescendant stores relative indices within the tree
                desc_rel = descendant[current]
                desc_idx = self._relative_to_absolute(desc_rel, tree_start)
                if desc_idx == -1:
                    break

                # Check scale factor limit
                if a_max is not None:
                    desc_snap = halos['SnapNum'][desc_idx]
                    desc_a = self.tree.get_scale_factor(desc_snap)
                    if desc_a > a_max:
                        break

                branch_indices.append(desc_idx)
                current = desc_idx

        # Remove duplicates and sort by scale factor
        branch_indices = list(set(branch_indices))

        # Create HaloInfo objects
        branch = [self._make_halo_info(idx) for idx in branch_indices]

        # Sort by scale factor (early to late)
        branch.sort(key=lambda h: h.scale_factor)

        return branch

    def trace_from_tree_id(
        self,
        tree_id: int,
        backward: bool = True,
        forward: bool = True,
        a_min: Optional[float] = None,
        a_max: Optional[float] = None,
    ) -> List[HaloInfo]:
        """
        Trace main branch starting from a tree's root halo.

        Parameters
        ----------
        tree_id : int
            Tree identifier (0 = most massive z=0 halo)
        backward, forward : bool
            Direction to trace
        a_min, a_max : float, optional
            Scale factor limits

        Returns
        -------
        branch : List[HaloInfo]
            Sorted by scale factor (early to late)
        """
        root_idx = self.tree.get_root_halo_index(tree_id)
        snap_num = int(self.tree.tree_halos['SnapNum'][root_idx])
        subhalo_nr = int(self.tree.tree_halos['SubhaloNr'][root_idx])

        return self.trace_main_branch(
            snap_num=snap_num,
            subhalo_nr=subhalo_nr,
            backward=backward,
            forward=forward,
            a_min=a_min,
            a_max=a_max,
        )

    def trace_all_progenitors(
        self,
        snap_num: int,
        subhalo_nr: int,
        a_min: Optional[float] = None,
    ) -> Dict[int, List[HaloInfo]]:
        """
        Trace ALL progenitor branches (not just main).

        Returns dict mapping snap_num to list of progenitors at that snapshot.
        Useful for visualizing mergers.

        Parameters
        ----------
        snap_num : int
            Reference snapshot
        subhalo_nr : int
            Subhalo index
        a_min : float, optional
            Minimum scale factor

        Returns
        -------
        progenitors : dict
            Mapping snap_num -> List[HaloInfo]
        """
        start_idx = self.tree.find_halo_in_tree(snap_num, subhalo_nr)
        if start_idx is None:
            return {}

        halos = self.tree.tree_halos
        first_prog = halos['TreeFirstProgenitor']
        next_prog = halos['TreeNextProgenitor']

        # Get tree start offset for converting relative indices
        tree_start = self._get_tree_start(start_idx)

        progenitors: Dict[int, List[HaloInfo]] = {}

        # BFS through all progenitors
        to_visit = [start_idx]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited or current == -1:
                continue
            visited.add(current)

            # Check scale factor limit
            current_snap = halos['SnapNum'][current]
            current_a = self.tree.get_scale_factor(current_snap)
            if a_min is not None and current_a < a_min:
                continue

            # Add to result
            halo_info = self._make_halo_info(current)
            if current_snap not in progenitors:
                progenitors[current_snap] = []
            progenitors[current_snap].append(halo_info)

            # Add all progenitors to visit list (convert relative to absolute)
            prog_rel = first_prog[current]
            prog = self._relative_to_absolute(prog_rel, tree_start)
            while prog != -1:
                to_visit.append(prog)
                prog_rel = next_prog[prog]
                prog = self._relative_to_absolute(prog_rel, tree_start)

        return progenitors

    def get_halo_at_snapshot(
        self,
        branch: List[HaloInfo],
        snap_num: int,
    ) -> Optional[HaloInfo]:
        """Get halo info from branch at specific snapshot."""
        return next((halo for halo in branch if halo.snap_num == snap_num), None)

    def get_halo_at_scale_factor(
        self,
        branch: List[HaloInfo],
        scale_factor: float,
        tolerance: float = 0.01,
    ) -> Optional[HaloInfo]:
        """Get halo info from branch nearest to given scale factor."""
        if not branch:
            return None

        # Find nearest
        best = min(branch, key=lambda h: abs(h.scale_factor - scale_factor))

        # Check tolerance
        if abs(best.scale_factor - scale_factor) / scale_factor > tolerance:
            return None

        return best

    def unwrap_coordinates(
        self,
        branch: List[HaloInfo],
        box_size: Optional[float] = None,
    ) -> List[HaloInfo]:
        """
        Unwrap periodic coordinates for smooth tracking.

        Ensures position changes are continuous (no jumps across box boundary).
        Modifies positions in-place and returns the branch.

        Parameters
        ----------
        branch : List[HaloInfo]
            Branch to unwrap (modified in-place)
        box_size : float, optional
            Box size (default: from merger tree)

        Returns
        -------
        branch : List[HaloInfo]
            Same branch with unwrapped positions
        """
        if len(branch) < 2:
            return branch

        if box_size is None:
            box_size = self.tree.box_size

        half_box = box_size / 2.0

        for i in range(1, len(branch)):
            prev_pos = branch[i - 1].position
            curr_pos = branch[i].position

            for dim in range(3):
                diff = curr_pos[dim] - prev_pos[dim]
                if diff > half_box:
                    # Jumped from high to low, add box_size
                    branch[i].position[dim] -= box_size
                elif diff < -half_box:
                    # Jumped from low to high, subtract box_size
                    branch[i].position[dim] += box_size

        return branch

    @staticmethod
    def unwrap_coordinates_static(
        positions: np.ndarray,
        box_size: float,
    ) -> np.ndarray:
        """
        Unwrap periodic coordinates (static method for testing).

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 3) array of positions at successive times
        box_size : float
            Periodic box size

        Returns
        -------
        unwrapped : np.ndarray
            Positions with jumps across boundaries removed
        """
        unwrapped = positions.copy()
        half_box = box_size / 2.0

        for i in range(1, len(positions)):
            diff = unwrapped[i] - unwrapped[i - 1]

            for dim in range(3):
                if diff[dim] > half_box:
                    unwrapped[i:, dim] -= box_size
                elif diff[dim] < -half_box:
                    unwrapped[i:, dim] += box_size

        return unwrapped
