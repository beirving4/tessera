#pragma once

#include <vector>
#include <cstdint>

namespace tessera {
namespace halo {

constexpr int GRID_TAIL = -1;

/**
 * Spatial hash grid for efficient neighbor lookups.
 * 
 * This is a linked-list based spatial hash where each cell contains
 * a head pointer, and each data element points to the next element
 * in the same cell.
 */
class HaloGrid {
public:
    int cells = 0;           // Number of cells per dimension
    double cell_width = 0;   // Physical width of each cell
    double box_width = 0;    // Total box width
    
    std::vector<int> heads;  // Grid-sized: head of linked list for each cell
    std::vector<int> next;   // Data-sized: next pointer for each element
    
    HaloGrid() = default;
    
    /**
     * Create a new spatial grid.
     * 
     * @param num_cells Number of cells per dimension
     * @param width Total box width
     * @param data_len Number of data elements to store
     */
    HaloGrid(int num_cells, double width, int data_len)
        : cells(num_cells),
          cell_width(width / static_cast<double>(num_cells)),
          box_width(width),
          heads(num_cells * num_cells * num_cells, GRID_TAIL),
          next(data_len) {}
    
    /**
     * Get the length (number of elements) in a cell.
     */
    int length(int idx) const {
        int n = 0;
        int ptr = heads[idx];
        while (ptr != GRID_TAIL) {
            n++;
            ptr = next[ptr];
        }
        return n;
    }
    
    /**
     * Insert elements into the grid based on their positions.
     * 
     * @param xs X coordinates
     * @param ys Y coordinates
     * @param zs Z coordinates
     */
    void insert(const std::vector<double>& xs, 
                const std::vector<double>& ys, 
                const std::vector<double>& zs) {
        for (size_t i = 0; i < xs.size(); ++i) {
            double x = xs[i];
            double y = ys[i];
            double z = zs[i];
            
            // Apply periodic boundary
            if (x >= box_width) x -= box_width;
            else if (x < 0) x += box_width;
            if (y >= box_width) y -= box_width;
            else if (y < 0) y += box_width;
            if (z >= box_width) z -= box_width;
            else if (z < 0) z += box_width;
            
            int ix = static_cast<int>(x / cell_width);
            int iy = static_cast<int>(y / cell_width);
            int iz = static_cast<int>(z / cell_width);
            
            // Clamp to valid range
            if (ix >= cells) ix = cells - 1;
            if (iy >= cells) iy = cells - 1;
            if (iz >= cells) iz = cells - 1;
            
            int idx = ix + iy * cells + iz * cells * cells;
            
            next[i] = heads[idx];
            heads[idx] = static_cast<int>(i);
        }
    }
    
    /**
     * Get total number of cells.
     */
    int total_cells() const {
        return static_cast<int>(heads.size());
    }
    
    /**
     * Get the maximum number of elements in any cell.
     */
    int max_length() const {
        int max = 0;
        for (int i = 0; i < total_cells(); ++i) {
            int len = length(i);
            if (len > max) max = len;
        }
        return max;
    }
    
    /**
     * Get the average number of elements per cell.
     */
    int average_length() const {
        return static_cast<int>(next.size()) / total_cells();
    }
    
    /**
     * Read all element indices in a cell into a buffer.
     * 
     * @param idx Cell index
     * @param buf Buffer to fill (will be resized as needed)
     * @return Reference to the filled buffer
     */
    std::vector<int>& read_indices(int idx, std::vector<int>& buf) const {
        buf.clear();
        int ptr = heads[idx];
        while (ptr != GRID_TAIL) {
            buf.push_back(ptr);
            ptr = next[ptr];
        }
        return buf;
    }
};

} // namespace halo
} // namespace tessera
