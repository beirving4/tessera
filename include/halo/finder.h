#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include "halo/grid.h"

namespace asymptotic_tetra {
namespace halo {

/**
 * Cell-aligned bounding box for spatial queries.
 */
struct Bounds {
    std::array<int, 3> origin = {0, 0, 0};
    std::array<int, 3> span = {0, 0, 0};
    
    /**
     * Create a cell-aligned bounding box around a sphere.
     * 
     * @param pos Sphere center position
     * @param r Sphere radius
     * @param cw Cell width
     * @param width Box width (for periodic wrapping)
     */
    void sphere_bounds(const std::array<double, 3>& pos, double r, 
                       double cw, double width) {
        for (int i = 0; i < 3; ++i) {
            double min_val = pos[i] - r;
            double max_val = pos[i] + r;
            
            if (min_val < 0) min_val += width;
            if (min_val > max_val) max_val += width;
            
            int min_cell = static_cast<int>(min_val / cw);
            int max_cell = static_cast<int>(max_val / cw);
            
            origin[i] = min_cell;
            span[i] = max_cell - min_cell + 1;
        }
    }
    
    /**
     * Convert non-periodic indices to periodic indices relative to bounds.
     */
    void convert_indices(int x, int y, int z, int width,
                        int& bx, int& by, int& bz) const {
        bx = x - origin[0];
        if (bx < 0) bx += width;
        by = y - origin[1];
        if (by < 0) by += width;
        bz = z - origin[2];
        if (bz < 0) bz += width;
    }
    
    /**
     * Check if a value is inside the bounds along a given dimension.
     */
    bool inside(int val, int width, int dim) const {
        int lo = origin[dim];
        int hi = origin[dim] + span[dim];
        
        if (val >= hi) {
            val -= width;
        } else if (val < lo) {
            val += width;
        }
        return val < hi && val >= lo;
    }
};

/**
 * SubhaloFinder computes which halos in a collection are subhalos of other
 * halos based purely on position and radius information.
 * 
 * Two halos are considered overlapping if their spheres (defined by position
 * and radius) intersect. The smaller halo is considered the subhalo.
 */
class SubhaloFinder {
public:
    /**
     * Create a new SubhaloFinder for the given spatial grid.
     * 
     * @param grid Spatial hash grid containing halo positions
     */
    explicit SubhaloFinder(HaloGrid* grid);
    
    /**
     * Find all subhalo relationships.
     * 
     * After calling this, use Hosts(), Subhalos(), etc. to query relationships.
     * 
     * @param xs X coordinates of halo centers
     * @param ys Y coordinates of halo centers
     * @param zs Z coordinates of halo centers
     * @param rs Radii of halos (will be multiplied by mult)
     * @param mult Multiplier for radii (typically 1.0)
     */
    void find_subhalos(std::vector<double>& xs, std::vector<double>& ys,
                       std::vector<double>& zs, std::vector<double>& rs,
                       double mult);
    
    /**
     * Get start and end indices in the subhalos array for a halo.
     */
    std::pair<int, int> start_end(int ih) const;
    
    /**
     * Get the number of halos that intersect with halo ih.
     */
    int intersect_count(int ih) const;
    
    /**
     * Get the number of halos that host halo ih (i.e., ih is inside them).
     */
    int host_count(int ih) const;
    
    /**
     * Get the number of subhalos of halo ih.
     */
    int subhalo_count(int ih) const;
    
    /**
     * Get all halos that intersect with halo ih (both hosts and subhalos).
     */
    std::vector<int> intersects(int ih) const;
    
    /**
     * Get all halos that host halo ih (ih is a subhalo of these).
     */
    std::vector<int> hosts(int ih) const;
    
    /**
     * Get all subhalos of halo ih.
     */
    std::vector<int> subhalos(int ih) const;

private:
    HaloGrid* grid_;
    std::vector<int> subhalo_starts_;
    std::vector<int> subhalos_;
    
    void start_subhalos(int i);
    void mark_subhalos(int ih, const std::vector<int>& idxs,
                       const std::vector<double>& xs, const std::vector<double>& ys,
                       const std::vector<double>& zs, const std::vector<double>& rs);
    void cross_match();
};

/**
 * Compute cumulative maximum from end to start.
 * 
 * For each position i, returns the maximum value in xs[i:].
 */
std::vector<double> cumulative_max(const std::vector<double>& xs);

} // namespace halo
} // namespace asymptotic_tetra
