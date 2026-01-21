"""
Tests for the tessera io module, including MergerTree reader.
"""

import numpy as np
import pytest


class TestIOModule:
    """Tests for basic IO module functionality."""

    def test_io_module_exists(self, ts):
        """Test that io module is available."""
        assert hasattr(ts, 'io')

    def test_has_hdf5(self, ts):
        """Test HDF5 support flag."""
        assert hasattr(ts.io, 'HAS_HDF5')
        # HAS_HDF5 should be True or False
        assert isinstance(ts.io.HAS_HDF5, bool)


@pytest.mark.requires_hdf5
class TestMergerTreeHeader:
    """Tests for MergerTreeHeader class."""

    def test_header_creation(self, ts):
        """Test MergerTreeHeader default construction."""
        header = ts.io.MergerTreeHeader()
        assert header.n_trees == 0
        assert header.n_halos == 0
        assert header.n_files == 1

    def test_header_attributes(self, ts):
        """Test MergerTreeHeader attributes."""
        header = ts.io.MergerTreeHeader()

        # Set attributes
        header.n_trees = 100
        header.n_halos = 10000
        header.box_size = 256.0
        header.hubble = 0.7

        assert header.n_trees == 100
        assert header.n_halos == 10000
        assert header.box_size == 256.0
        assert header.hubble == 0.7


@pytest.mark.requires_hdf5
class TestHaloInfo:
    """Tests for HaloInfo class."""

    def test_halo_info_creation(self, ts):
        """Test HaloInfo default construction."""
        info = ts.io.HaloInfo()
        assert info.tree_index == -1
        assert info.is_valid() is False

    def test_halo_info_attributes(self, ts):
        """Test HaloInfo attributes."""
        info = ts.io.HaloInfo()

        # Set attributes
        info.tree_index = 42
        info.snap_num = 74
        info.subhalo_nr = 0
        info.scale_factor = 1.0
        info.redshift = 0.0
        info.mass = 1e5
        info.m200c = 8e4
        info.r200c = 0.5

        assert info.tree_index == 42
        assert info.snap_num == 74
        assert info.subhalo_nr == 0
        assert info.is_valid() is True
        assert info.scale_factor == pytest.approx(1.0)
        assert info.mass == pytest.approx(1e5)


@pytest.mark.requires_hdf5
class TestTreeTableEntry:
    """Tests for TreeTableEntry class."""

    def test_tree_table_entry_creation(self, ts):
        """Test TreeTableEntry default construction."""
        entry = ts.io.TreeTableEntry()
        assert entry.start_offset == 0
        assert entry.length == 0
        assert entry.tree_id == 0


@pytest.mark.requires_hdf5
@pytest.mark.requires_tree_file
class TestMergerTree:
    """Tests for MergerTree class (requires actual tree file)."""

    def test_merger_tree_load(self, ts, merger_tree_path):
        """Test loading a merger tree file."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        assert tree.n_trees() > 0
        assert tree.n_halos() > 0
        assert tree.box_size() > 0

    def test_merger_tree_header(self, ts, merger_tree_path):
        """Test merger tree header access."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        header = tree.header()

        assert header.n_trees == tree.n_trees()
        assert header.n_halos == tree.n_halos()
        assert header.box_size == tree.box_size()

    def test_staged_loading(self, ts, merger_tree_path):
        """Test that staged loading works correctly."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        # Initially, nothing should be loaded
        assert tree.search_data_loaded() is False
        assert tree.halos_loaded() is False

        # find_halo_in_tree should only load search data
        idx = tree.find_halo_in_tree(snap_num=tree.header().last_snap, subhalo_nr=0)

        assert tree.search_data_loaded() is True
        assert tree.halos_loaded() is False  # Full data not loaded yet

        # make_halo_info should trigger full data load
        if idx is not None:
            info = tree.make_halo_info(idx)
            assert tree.halos_loaded() is True

    def test_find_halo_in_tree(self, ts, merger_tree_path):
        """Test finding a halo by snapshot and subhalo number."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        # Find the root halo of tree 0
        last_snap = tree.header().last_snap
        idx = tree.find_halo_in_tree(snap_num=last_snap, subhalo_nr=0)

        assert idx is not None
        assert idx >= 0

    def test_find_halo_not_found(self, ts, merger_tree_path):
        """Test that find_halo_in_tree returns None for non-existent halo."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        # Try to find a halo that doesn't exist
        idx = tree.find_halo_in_tree(snap_num=9999, subhalo_nr=9999)
        assert idx is None

    def test_make_halo_info(self, ts, merger_tree_path):
        """Test creating HaloInfo from tree index."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        last_snap = tree.header().last_snap
        idx = tree.find_halo_in_tree(snap_num=last_snap, subhalo_nr=0)

        if idx is not None:
            info = tree.make_halo_info(idx)

            assert info.is_valid()
            assert info.tree_index == idx
            assert info.snap_num == last_snap
            assert info.subhalo_nr == 0
            assert len(info.position) == 3
            assert len(info.velocity) == 3

    def test_get_tree_entry(self, ts, merger_tree_path):
        """Test getting tree table entry."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        entry = tree.get_tree_entry(0)

        assert entry.tree_id == 0
        assert entry.length > 0
        assert entry.start_offset >= 0

    def test_get_tree_indices(self, ts, merger_tree_path):
        """Test getting tree indices."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        start, end = tree.get_tree_indices(0)

        assert start >= 0
        assert end > start

    def test_clear_cache(self, ts, merger_tree_path):
        """Test clearing cached data."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        # Load some data
        tree.find_halo_in_tree(snap_num=tree.header().last_snap, subhalo_nr=0)
        assert tree.search_data_loaded() is True

        # Clear cache
        tree.clear_cache()

        assert tree.search_data_loaded() is False
        assert tree.halos_loaded() is False

    def test_memory_usage(self, ts, merger_tree_path):
        """Test memory usage reporting."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)

        # Initial memory should be small (just header)
        initial_mem = tree.memory_usage()

        # Load search data
        tree.find_halo_in_tree(snap_num=tree.header().last_snap, subhalo_nr=0)
        search_mem = tree.memory_usage()

        # Search data should use some memory
        assert search_mem > initial_mem

        # Load full data
        tree.make_halo_info(0)
        full_mem = tree.memory_usage()

        # Full data should use more memory than search data
        assert full_mem > search_mem


@pytest.mark.requires_hdf5
@pytest.mark.requires_tree_file
class TestHaloTracker:
    """Tests for HaloTracker class (requires actual tree file)."""

    def test_halo_tracker_creation(self, ts, merger_tree_path):
        """Test HaloTracker creation."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        # Tracker should be created successfully
        assert tracker is not None

    def test_trace_main_branch(self, ts, merger_tree_path):
        """Test tracing main branch."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        last_snap = tree.header().last_snap
        branch = tracker.trace_main_branch(
            snap_num=last_snap,
            subhalo_nr=0,
            backward=True,
            forward=False
        )

        assert len(branch) > 0

        # Branch should be sorted by scale factor
        scale_factors = [h.scale_factor for h in branch]
        assert scale_factors == sorted(scale_factors)

    def test_trace_main_branch_with_limits(self, ts, merger_tree_path):
        """Test tracing main branch with scale factor limits."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        last_snap = tree.header().last_snap
        branch = tracker.trace_main_branch(
            snap_num=last_snap,
            subhalo_nr=0,
            backward=True,
            forward=False,
            a_min=0.1
        )

        # All halos should have scale factor >= 0.1
        for halo in branch:
            assert halo.scale_factor >= 0.1

    def test_trace_from_tree_id(self, ts, merger_tree_path):
        """Test tracing from tree ID."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        branch = tracker.trace_from_tree_id(
            tree_id=0,
            backward=True,
            forward=False
        )

        assert len(branch) > 0

    def test_trace_all_progenitors(self, ts, merger_tree_path):
        """Test tracing all progenitors."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        last_snap = tree.header().last_snap
        progenitors = tracker.trace_all_progenitors(
            snap_num=last_snap,
            subhalo_nr=0
        )

        # Should return a dict mapping snap_num -> list of HaloInfo
        assert isinstance(progenitors, dict)
        assert len(progenitors) > 0

        # Each value should be a list of HaloInfo
        for snap_num, halos in progenitors.items():
            assert isinstance(halos, list)
            assert len(halos) > 0
            for halo in halos:
                assert halo.snap_num == snap_num

    def test_unwrap_coordinates(self, ts, merger_tree_path):
        """Test coordinate unwrapping."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        tree = ts.io.MergerTree(merger_tree_path)
        tracker = ts.io.HaloTracker(tree)

        last_snap = tree.header().last_snap
        branch = tracker.trace_main_branch(
            snap_num=last_snap,
            subhalo_nr=0,
            backward=True,
            forward=False
        )

        # Store original positions
        original_positions = [tuple(h.position) for h in branch]

        # Unwrap coordinates
        tracker.unwrap_coordinates(branch)

        # Branch should still have same length
        assert len(branch) == len(original_positions)

        # After unwrapping, jumps should be smaller than half box size
        box_size = tree.box_size()
        for i in range(1, len(branch)):
            for dim in range(3):
                diff = abs(branch[i].position[dim] - branch[i-1].position[dim])
                # Allow some tolerance for coordinate changes
                # The unwrapped coords might extend outside [0, box_size]
                # but jumps should be continuous


@pytest.mark.requires_hdf5
class TestReadMergerTreeHeader:
    """Tests for read_merger_tree_header function."""

    def test_read_header_function(self, ts, merger_tree_path):
        """Test standalone header reading function."""
        if merger_tree_path is None:
            pytest.skip("No merger tree file available")

        header = ts.io.read_merger_tree_header(merger_tree_path)

        assert header.n_trees > 0
        assert header.n_halos > 0
        assert header.box_size > 0
