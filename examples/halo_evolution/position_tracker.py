"""
Position-Based Halo Tracking

Alternative to merger tree linking that tracks halos by spatial position and mass.
Useful for:
- Validating merger tree results
- Handling tracking errors in merger trees
- Cases where merger trees are unavailable

The position-matching algorithm uses a combined score of:
- Spatial distance (with periodic boundary handling)
- Mass similarity (log-ratio)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from pathlib import Path
import h5py


@dataclass
class TrackedHalo:
    """
    Information about a halo at one snapshot.

    Attributes
    ----------
    snap_num : int
        Snapshot number
    group_nr : int
        FOF group index
    subhalo_nr : int
        Subhalo index in catalog (if available)
    scale_factor : float
        Scale factor a = 1/(1+z)
    position : np.ndarray
        (3,) comoving position [Mpc/h or cMpc/h]
    m200c : float
        M200c [Msun/h] (in code units, typically 10^10 Msun/h)
    r200c : float
        R200c [Mpc/h or cMpc/h]
    match_score : float
        Quality score of the match (lower is better)
    match_distance : float
        Spatial distance from previous halo [Mpc/h or cMpc/h]
    """
    snap_num: int
    group_nr: int
    subhalo_nr: Optional[int]
    scale_factor: float
    position: np.ndarray
    m200c: float
    r200c: float
    match_score: float = 0.0
    match_distance: float = 0.0


def periodic_distance(pos1: np.ndarray, pos2: np.ndarray, box_size: float) -> float:
    """
    Compute distance between two positions with periodic boundaries.

    Parameters
    ----------
    pos1, pos2 : np.ndarray
        (3,) position arrays
    box_size : float
        Periodic box size

    Returns
    -------
    distance : float
        Minimum distance accounting for periodic wrapping
    """
    delta = np.abs(pos1 - pos2)
    delta = np.minimum(delta, box_size - delta)
    return np.sqrt(np.sum(delta**2))


def periodic_distances_vectorized(
    target_pos: np.ndarray,
    positions: np.ndarray,
    box_size: float
) -> np.ndarray:
    """
    Compute distances from target to all positions (vectorized).

    Parameters
    ----------
    target_pos : np.ndarray
        (3,) target position
    positions : np.ndarray
        (N, 3) array of candidate positions
    box_size : float
        Periodic box size

    Returns
    -------
    distances : np.ndarray
        (N,) array of distances
    """
    delta = np.abs(positions - target_pos)
    delta = np.minimum(delta, box_size - delta)
    return np.sqrt(np.sum(delta**2, axis=1))


class HaloCatalogLoader:
    """
    Load halo data from FOF/SUBFIND catalogs.

    Handles both single-file and sharded catalogs.
    """

    def __init__(self, catalog_dir: Path):
        """
        Initialize catalog loader.

        Parameters
        ----------
        catalog_dir : Path
            Directory containing halo catalogs
        """
        self.catalog_dir = Path(catalog_dir)

    def _find_catalog_files(self, snap_num: int) -> List[Path]:
        """Find all catalog files for a snapshot (handles sharding)."""
        # Try different naming patterns
        patterns = [
            f"fof_subhalo_tab_{snap_num:03d}.*.hdf5",  # Sharded
            f"fof_subhalo_tab_{snap_num:03d}.hdf5",    # Single file
            f"groups_{snap_num:03d}.*.hdf5",           # Alternative naming
            f"groups_{snap_num:03d}.hdf5",
        ]

        for pattern in patterns:
            files = sorted(self.catalog_dir.glob(pattern))
            if files:
                return files

        return []

    def get_available_snapshots(self) -> List[int]:
        """Get list of available snapshot numbers."""
        snapshots = set()
        for f in self.catalog_dir.glob("fof_subhalo_tab_*.hdf5"):
            # Extract snapshot number from filename
            parts = f.stem.split("_")
            snap_str = parts[-1].split(".")[0]  # Handle sharded files
            try:
                snapshots.add(int(snap_str))
            except ValueError:
                continue

        # Try alternative naming
        for f in self.catalog_dir.glob("groups_*.hdf5"):
            parts = f.stem.split("_")
            snap_str = parts[-1].split(".")[0]
            try:
                snapshots.add(int(snap_str))
            except ValueError:
                continue

        return sorted(snapshots)

    def load_groups(
        self,
        snap_num: int,
        fields: List[str] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load FOF group data from catalog.

        Parameters
        ----------
        snap_num : int
            Snapshot number
        fields : List[str], optional
            Fields to load. Default: position, M200c, R200c

        Returns
        -------
        data : dict or None
            Dictionary of arrays, or None if catalog not found
        """
        if fields is None:
            fields = ['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200']

        files = self._find_catalog_files(snap_num)
        if not files:
            return None

        # Collect data from all shards
        all_data = {f: [] for f in fields}
        scale_factor = None

        for fpath in files:
            try:
                with h5py.File(fpath, 'r') as hf:
                    if 'Group' not in hf:
                        continue

                    for field in fields:
                        if field in hf['Group']:
                            all_data[field].append(hf['Group'][field][:])

                    # Get scale factor from header
                    if scale_factor is None and 'Header' in hf:
                        scale_factor = hf['Header'].attrs.get('Time', None)

            except Exception as e:
                print(f"Warning: Error reading {fpath}: {e}")
                continue

        # Concatenate shards
        result = {}
        for field in fields:
            if all_data[field]:
                result[field] = np.concatenate(all_data[field], axis=0)

        if not result:
            return None

        result['scale_factor'] = scale_factor
        return result


class PositionMatchingTracker:
    """
    Track halos across snapshots using position and mass matching.

    This provides an alternative to merger tree-based tracking that can be
    used to validate tree results or when trees are unavailable.

    Parameters
    ----------
    catalog_loader : HaloCatalogLoader
        Loader for halo catalogs
    box_size : float
        Simulation box size [Mpc/h or cMpc/h]
    search_radius : float, optional
        Maximum search radius for matches [same units as box_size]
    mass_weight : float, optional
        Weight for mass similarity in scoring (default: 0.3)
    min_mass_ratio : float, optional
        Minimum mass ratio to consider a match (default: 0.05)

    Examples
    --------
    >>> loader = HaloCatalogLoader('/path/to/catalogs')
    >>> tracker = PositionMatchingTracker(loader, box_size=32.0)
    >>>
    >>> # Track backward from snap 74
    >>> branch = tracker.track_branch(
    ...     start_snap=74,
    ...     start_group_nr=0,
    ...     backward=True,
    ...     forward=False
    ... )
    >>>
    >>> for halo in branch:
    ...     print(f"Snap {halo.snap_num}: M200c={halo.m200c:.2e}, dist={halo.match_distance:.2f}")
    """

    def __init__(
        self,
        catalog_loader: HaloCatalogLoader,
        box_size: float,
        search_radius: float = 5.0,
        mass_weight: float = 0.3,
        min_mass_ratio: float = 0.05,
    ):
        self.loader = catalog_loader
        self.box_size = box_size
        self.search_radius = search_radius
        self.mass_weight = mass_weight
        self.min_mass_ratio = min_mass_ratio

        # Cache for loaded catalogs
        self._catalog_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _get_catalog(self, snap_num: int) -> Optional[Dict[str, np.ndarray]]:
        """Get catalog data with caching."""
        if snap_num not in self._catalog_cache:
            data = self.loader.load_groups(snap_num)
            if data is not None:
                self._catalog_cache[snap_num] = data
        return self._catalog_cache.get(snap_num)

    def clear_cache(self):
        """Clear the catalog cache."""
        self._catalog_cache.clear()

    def find_best_match(
        self,
        target_pos: np.ndarray,
        target_mass: float,
        positions: np.ndarray,
        masses: np.ndarray,
        search_radius: Optional[float] = None,
    ) -> Tuple[Optional[int], float, float]:
        """
        Find the best matching halo at a snapshot.

        Parameters
        ----------
        target_pos : np.ndarray
            (3,) position of target halo
        target_mass : float
            Mass of target halo
        positions : np.ndarray
            (N, 3) positions of candidate halos
        masses : np.ndarray
            (N,) masses of candidate halos
        search_radius : float, optional
            Override default search radius

        Returns
        -------
        best_idx : int or None
            Index of best match, or None if no match found
        best_score : float
            Quality score (lower is better)
        best_distance : float
            Spatial distance to match
        """
        if search_radius is None:
            search_radius = self.search_radius

        if len(positions) == 0:
            return None, float('inf'), float('inf')

        # Compute distances to all candidates
        distances = periodic_distances_vectorized(target_pos, positions, self.box_size)

        # Filter by search radius and minimum mass
        min_mass = target_mass * self.min_mass_ratio
        valid_mask = (distances < search_radius) & (masses > min_mass)

        if not np.any(valid_mask):
            return None, float('inf'), float('inf')

        # Compute scores for valid candidates
        valid_indices = np.where(valid_mask)[0]
        best_idx = None
        best_score = float('inf')
        best_distance = float('inf')

        for idx in valid_indices:
            dist = distances[idx]
            mass = masses[idx]

            # Combined score: distance + weighted log mass ratio
            if target_mass > 0 and mass > 0:
                mass_ratio = abs(np.log10(mass / target_mass))
            else:
                mass_ratio = 0

            score = dist + self.mass_weight * mass_ratio

            if score < best_score:
                best_score = score
                best_idx = idx
                best_distance = dist

        return best_idx, best_score, best_distance

    def track_branch(
        self,
        start_snap: int,
        start_group_nr: int,
        backward: bool = True,
        forward: bool = False,
        snap_min: Optional[int] = None,
        snap_max: Optional[int] = None,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TrackedHalo]:
        """
        Track a halo branch across snapshots.

        Parameters
        ----------
        start_snap : int
            Starting snapshot number
        start_group_nr : int
            FOF group number at starting snapshot
        backward : bool
            Track backward in time (progenitors)
        forward : bool
            Track forward in time (descendants)
        snap_min, snap_max : int, optional
            Limit tracking to this snapshot range
        verbose : bool
            Print progress information
        progress_callback : callable, optional
            Called with (current_snap, total_snaps) for progress updates

        Returns
        -------
        branch : List[TrackedHalo]
            Tracked halos sorted by scale factor (early to late)
        """
        # Get starting halo
        start_data = self._get_catalog(start_snap)
        if start_data is None:
            raise ValueError(f"Could not load catalog for snapshot {start_snap}")

        if start_group_nr >= len(start_data['GroupPos']):
            raise ValueError(f"Group {start_group_nr} not found in snapshot {start_snap}")

        # Create starting halo info
        start_halo = TrackedHalo(
            snap_num=start_snap,
            group_nr=start_group_nr,
            subhalo_nr=None,
            scale_factor=start_data.get('scale_factor', 0.0),
            position=start_data['GroupPos'][start_group_nr].copy(),
            m200c=start_data['Group_M_Crit200'][start_group_nr],
            r200c=start_data.get('Group_R_Crit200', np.zeros(1))[start_group_nr]
                  if 'Group_R_Crit200' in start_data else 0.0,
            match_score=0.0,
            match_distance=0.0,
        )

        branch = [start_halo]

        # Get available snapshots
        available_snaps = self.loader.get_available_snapshots()
        if not available_snaps:
            return branch

        # Apply limits
        if snap_min is not None:
            available_snaps = [s for s in available_snaps if s >= snap_min]
        if snap_max is not None:
            available_snaps = [s for s in available_snaps if s <= snap_max]

        # Track backward
        if backward:
            backward_snaps = sorted([s for s in available_snaps if s < start_snap], reverse=True)
            current_pos = start_halo.position.copy()
            current_mass = start_halo.m200c

            for i, snap in enumerate(backward_snaps):
                if progress_callback:
                    progress_callback(i + 1, len(backward_snaps))

                data = self._get_catalog(snap)
                if data is None:
                    if verbose:
                        print(f"  Snap {snap}: No catalog data")
                    continue

                positions = data['GroupPos']
                masses = data['Group_M_Crit200']

                best_idx, score, distance = self.find_best_match(
                    current_pos, current_mass, positions, masses
                )

                if best_idx is None:
                    if verbose:
                        print(f"  Snap {snap}: No match found")
                    continue

                # Create halo info
                halo = TrackedHalo(
                    snap_num=snap,
                    group_nr=best_idx,
                    subhalo_nr=None,
                    scale_factor=data.get('scale_factor', 0.0),
                    position=positions[best_idx].copy(),
                    m200c=masses[best_idx],
                    r200c=data.get('Group_R_Crit200', np.zeros(1))[best_idx]
                          if 'Group_R_Crit200' in data and best_idx < len(data.get('Group_R_Crit200', [])) else 0.0,
                    match_score=score,
                    match_distance=distance,
                )

                if verbose:
                    print(f"  Snap {snap}: Group {best_idx}, M200c={masses[best_idx]:.2e}, "
                          f"dist={distance:.3f}")

                branch.append(halo)

                # Update tracking position/mass
                current_pos = halo.position.copy()
                current_mass = halo.m200c

        # Track forward
        if forward:
            forward_snaps = sorted([s for s in available_snaps if s > start_snap])
            current_pos = start_halo.position.copy()
            current_mass = start_halo.m200c

            for i, snap in enumerate(forward_snaps):
                if progress_callback:
                    progress_callback(i + 1, len(forward_snaps))

                data = self._get_catalog(snap)
                if data is None:
                    if verbose:
                        print(f"  Snap {snap}: No catalog data")
                    continue

                positions = data['GroupPos']
                masses = data['Group_M_Crit200']

                best_idx, score, distance = self.find_best_match(
                    current_pos, current_mass, positions, masses
                )

                if best_idx is None:
                    if verbose:
                        print(f"  Snap {snap}: No match found")
                    continue

                halo = TrackedHalo(
                    snap_num=snap,
                    group_nr=best_idx,
                    subhalo_nr=None,
                    scale_factor=data.get('scale_factor', 0.0),
                    position=positions[best_idx].copy(),
                    m200c=masses[best_idx],
                    r200c=data.get('Group_R_Crit200', np.zeros(1))[best_idx]
                          if 'Group_R_Crit200' in data and best_idx < len(data.get('Group_R_Crit200', [])) else 0.0,
                    match_score=score,
                    match_distance=distance,
                )

                if verbose:
                    print(f"  Snap {snap}: Group {best_idx}, M200c={masses[best_idx]:.2e}, "
                          f"dist={distance:.3f}")

                branch.append(halo)
                current_pos = halo.position.copy()
                current_mass = halo.m200c

        # Sort by scale factor (early to late)
        branch.sort(key=lambda h: h.scale_factor)

        return branch

    def compare_with_merger_tree(
        self,
        tree_branch: List,  # List[HaloInfo] from HaloTracker
        verbose: bool = True,
    ) -> Dict[int, Dict]:
        """
        Compare position-matched tracking with merger tree results.

        Parameters
        ----------
        tree_branch : List[HaloInfo]
            Branch from HaloTracker.trace_main_branch()
        verbose : bool
            Print comparison details

        Returns
        -------
        comparison : dict
            Per-snapshot comparison with:
            - 'tree_group': Group from merger tree
            - 'pos_group': Group from position matching
            - 'match': Whether they agree
            - 'mass_ratio': Ratio of M200c values
        """
        if not tree_branch:
            return {}

        # Get reference snapshot/halo from tree
        # Use the latest snapshot as reference
        ref_halo = max(tree_branch, key=lambda h: h.scale_factor)

        # Do position-matched tracking
        pos_branch = self.track_branch(
            start_snap=ref_halo.snap_num,
            start_group_nr=ref_halo.group_nr,
            backward=True,
            forward=True,
            verbose=False,
        )

        # Build lookup by snapshot
        tree_by_snap = {h.snap_num: h for h in tree_branch}
        pos_by_snap = {h.snap_num: h for h in pos_branch}

        comparison = {}
        all_snaps = sorted(set(tree_by_snap.keys()) | set(pos_by_snap.keys()))

        for snap in all_snaps:
            tree_h = tree_by_snap.get(snap)
            pos_h = pos_by_snap.get(snap)

            if tree_h and pos_h:
                match = (tree_h.group_nr == pos_h.group_nr)
                if tree_h.m200c > 0 and pos_h.m200c > 0:
                    mass_ratio = pos_h.m200c / tree_h.m200c
                else:
                    mass_ratio = None

                comparison[snap] = {
                    'tree_group': tree_h.group_nr,
                    'pos_group': pos_h.group_nr,
                    'match': match,
                    'mass_ratio': mass_ratio,
                    'tree_m200c': tree_h.m200c,
                    'pos_m200c': pos_h.m200c,
                }

                if verbose and not match:
                    print(f"Snap {snap}: MISMATCH - Tree group {tree_h.group_nr} vs "
                          f"Position group {pos_h.group_nr}")
                    if mass_ratio:
                        print(f"  Mass ratio (pos/tree): {mass_ratio:.2f}")

            elif tree_h:
                comparison[snap] = {
                    'tree_group': tree_h.group_nr,
                    'pos_group': None,
                    'match': False,
                    'mass_ratio': None,
                }
            elif pos_h:
                comparison[snap] = {
                    'tree_group': None,
                    'pos_group': pos_h.group_nr,
                    'match': False,
                    'mass_ratio': None,
                }

        return comparison

    def unwrap_coordinates(
        self,
        branch: List[TrackedHalo],
    ) -> List[TrackedHalo]:
        """
        Unwrap periodic coordinates for smooth tracking visualization.

        Modifies positions in-place to remove jumps across box boundaries.

        Parameters
        ----------
        branch : List[TrackedHalo]
            Branch to unwrap (modified in-place)

        Returns
        -------
        branch : List[TrackedHalo]
            Same branch with unwrapped positions
        """
        if len(branch) < 2:
            return branch

        half_box = self.box_size / 2.0

        for i in range(1, len(branch)):
            prev_pos = branch[i - 1].position
            curr_pos = branch[i].position

            for dim in range(3):
                diff = curr_pos[dim] - prev_pos[dim]
                if diff > half_box:
                    branch[i].position[dim] -= self.box_size
                elif diff < -half_box:
                    branch[i].position[dim] += self.box_size

        return branch


def validate_branch_catalog(
    catalog_path: Path,
    halo_catalog_dir: Path,
    box_size: float,
    branch_id: int = 0,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Validate a branch catalog entry using position matching.

    Parameters
    ----------
    catalog_path : Path
        Path to branch catalog HDF5 file
    halo_catalog_dir : Path
        Directory containing FOF/SUBFIND catalogs
    box_size : float
        Simulation box size
    branch_id : int
        Branch to validate
    verbose : bool
        Print progress

    Returns
    -------
    result : dict
        Validation results including:
        - 'valid': Overall validity
        - 'n_mismatches': Number of snapshot mismatches
        - 'mismatch_snaps': List of mismatched snapshots
        - 'comparison': Per-snapshot comparison data
    """
    # Load branch catalog
    with h5py.File(catalog_path, 'r') as hf:
        # Get branch info
        start_offset = hf['BranchIndex/start_offset'][branch_id]
        length = hf['BranchIndex/length'][branch_id]

        # Get halo data for this branch
        snap_nums = hf['HaloData/snap_num'][start_offset:start_offset+length]
        group_nrs = hf['HaloData/group_nr'][start_offset:start_offset+length] \
                    if 'HaloData/group_nr' in hf else None
        m200c = hf['HaloData/m200c'][start_offset:start_offset+length]
        scale_factors = hf['HaloData/scale_factor'][start_offset:start_offset+length]

    # Get final snapshot info for starting position match
    final_snap = snap_nums[-1]
    final_group = group_nrs[-1] if group_nrs is not None else 0

    if verbose:
        print(f"Validating branch {branch_id}: {length} snapshots")
        print(f"Final snapshot: {final_snap}, group: {final_group}")

    # Create tracker and do position matching
    loader = HaloCatalogLoader(halo_catalog_dir)
    tracker = PositionMatchingTracker(loader, box_size)

    pos_branch = tracker.track_branch(
        start_snap=final_snap,
        start_group_nr=final_group,
        backward=True,
        forward=False,
        verbose=verbose,
    )

    # Compare
    pos_by_snap = {h.snap_num: h for h in pos_branch}

    comparison = {}
    mismatches = []

    for i, snap in enumerate(snap_nums):
        catalog_m200c = m200c[i]
        pos_h = pos_by_snap.get(snap)

        if pos_h:
            if catalog_m200c > 0 and pos_h.m200c > 0:
                mass_ratio = pos_h.m200c / catalog_m200c
            else:
                mass_ratio = None

            # Consider it a match if mass ratio is close to 1
            is_match = mass_ratio is not None and 0.8 < mass_ratio < 1.2

            comparison[snap] = {
                'catalog_m200c': catalog_m200c,
                'pos_m200c': pos_h.m200c,
                'mass_ratio': mass_ratio,
                'match': is_match,
            }

            if not is_match:
                mismatches.append(snap)
        else:
            comparison[snap] = {
                'catalog_m200c': catalog_m200c,
                'pos_m200c': None,
                'mass_ratio': None,
                'match': False,
            }
            mismatches.append(snap)

    result = {
        'valid': len(mismatches) == 0,
        'n_mismatches': len(mismatches),
        'mismatch_snaps': mismatches,
        'comparison': comparison,
        'total_snaps': len(snap_nums),
    }

    if verbose:
        print(f"\nValidation result: {len(mismatches)}/{len(snap_nums)} mismatches")
        if mismatches:
            print(f"Mismatch snapshots: {mismatches}")

    return result
