# Chunked Extraction Mode - Known Issues

## Overview

The chunked/low-memory extraction mode (`chunk_size > 0`) for `extract_all_branches_parallel` is currently broken. Normal mode (`chunk_size=0`) works correctly.

**Goal**: Process large merger tree files (60-100 GB) on memory-constrained systems by loading halo data in chunks rather than all at once.

## Symptoms

1. Chunked mode hangs after processing the first chunk
2. Debug output shows wrong number of halos loaded:
   ```
   [DEBUG]   Loading range [0, 9819)
   [DEBUG]   Loaded 63890 halos (expected 9819)
   ```
3. The function `load_tree_halos_range()` appears to be called but loads ALL halos instead of the requested range

## Root Cause Analysis

### Primary Issue: `tree_halos()` vs `tree_halos_loaded()`

The `MergerTree` class has two methods for accessing halo data:

```cpp
// Triggers lazy load of ALL halo data
const TreeHalosSOA& tree_halos();

// Returns current data without triggering load
const TreeHalosSOA& tree_halos_loaded() const;
```

**Problem**: Multiple code paths were calling `tree_halos()` after `load_tree_halos_range()`, which overwrites the range-loaded data with full data.

### Fixes Applied (but still not working)

1. **Line ~1434** (root halo loading in `extract_all_branches_parallel`):
   ```cpp
   // BEFORE (bug)
   tree.load_tree_halos_range(min_idx, max_idx + 1);
   const auto& halos = tree.tree_halos();  // Triggers full load!

   // AFTER (fix)
   tree.load_tree_halos_range(min_idx, max_idx + 1);
   const auto& halos = tree.tree_halos_loaded();
   ```

2. **Line ~1168** (in `extract_branches_chunked`):
   ```cpp
   // BEFORE (bug)
   tree.load_tree_halos_range(chunk_start, chunk_end);
   const auto& halos = tree.tree_halos();  // Triggers full load!

   // AFTER (fix)
   tree.load_tree_halos_range(chunk_start, chunk_end);
   const auto& halos = tree.tree_halos_loaded();
   ```

### Remaining Mystery

Even after applying these fixes, the debug output still showed 63890 halos being loaded instead of the expected range. Debug statements added *inside* `load_tree_halos_range()` did not appear in output:

```cpp
void MergerTree::load_tree_halos_range(int64_t start_idx, int64_t end_idx) {
    std::cerr << "[DEBUG load_range] Called..." << std::endl;  // NEVER PRINTED
    // ...
}
```

This suggests one of:
1. **Module caching issue**: Python loading old `.so` from different location
2. **Build issue**: Object file not being rebuilt despite source changes
3. **Hidden call path**: Some other code path calling `tree_halos()` or loading data

## Files Involved

- `src/io/merger_tree.cpp`:
  - `load_tree_halos_range()` - Range loading implementation (~line 597)
  - `extract_branches_chunked()` - Chunked extraction logic (~line 1097)
  - `extract_all_branches_parallel()` - Main entry point (~line 1355)

- `include/io/merger_tree.h`:
  - `tree_halos_loaded()` method - Added to access data without lazy load

## Debugging Steps Tried

1. Added debug output throughout code
2. Verified source changes with grep
3. Forced recompilation with `touch` and `make`
4. Removed object files and rebuilt
5. Copied `.so` to multiple locations
6. Checked module load path with `tessera.__file__`

## Investigation TODOs

- [ ] Verify which `.so` Python actually loads at runtime
- [ ] Add debug output to `load_full_halo_data()` to trace unexpected calls
- [ ] Check if `HaloTracker` or other classes call `tree_halos()` unexpectedly
- [ ] Test with minimal C++ unit test (bypass Python bindings)
- [ ] Check if HighFive hyperslab selection is actually working correctly
- [ ] Verify `clear_halos_cache()` actually clears the vectors

## Test Commands

```bash
# Normal mode (works)
KMP_DUPLICATE_LIB_OK=TRUE python -c "
import tessera as ts
tree = ts.io.MergerTree('path/to/trees.hdf5')
result = ts.io.extract_all_branches_parallel(tree, chunk_size=0)
print(f'Extracted {len(result.tree_id)} branches')
"

# Chunked mode (broken)
KMP_DUPLICATE_LIB_OK=TRUE python -c "
import tessera as ts
tree = ts.io.MergerTree('path/to/trees.hdf5')
result = ts.io.extract_all_branches_parallel(tree, chunk_size=10000)
print(f'Extracted {len(result.tree_id)} branches')
"
```

## Related Code

Key methods to understand:

```cpp
// Clears all halo data
void MergerTree::clear_halos_cache() {
    tree_halos_.clear();
    search_loaded_ = false;
    halos_loaded_ = false;
    range_loaded_ = false;
    loaded_range_start_ = 0;
    loaded_range_end_ = 0;
}

// Loads a range of halos using HDF5 hyperslab
void MergerTree::load_tree_halos_range(int64_t start_idx, int64_t end_idx) {
    clear_halos_cache();
    // ... reads slice with HighFive select()
    range_loaded_ = true;
    loaded_range_start_ = start_idx;
    loaded_range_end_ = end_idx;
    halos_loaded_ = false;  // Not fully loaded
}

// Triggers full data load - THE PROBLEM
const TreeHalosSOA& MergerTree::tree_halos() {
    load_full_halo_data();  // Overwrites range data!
    return tree_halos_;
}
```

## Priority

Low - Normal mode works and handles typical use cases. Chunked mode is only needed for very large simulations (60-100 GB tree files) on memory-constrained systems.

---
*Last updated: 2026-02-02*
