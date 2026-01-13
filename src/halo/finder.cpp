#include "halo/finder.h"
#include <algorithm>
#include <numeric>

namespace tessera {
namespace halo {

std::vector<double> cumulative_max(const std::vector<double>& xs) {
    std::vector<double> ms(xs.size());
    if (xs.empty()) return ms;
    
    double max_val = xs.back();
    for (int i = static_cast<int>(xs.size()) - 1; i >= 0; --i) {
        if (xs[i] > max_val) max_val = xs[i];
        ms[i] = max_val;
    }
    return ms;
}

SubhaloFinder::SubhaloFinder(HaloGrid* grid)
    : grid_(grid),
      subhalo_starts_(grid->next.size()),
      subhalos_() 
{
    subhalos_.reserve(static_cast<size_t>(2.5 * grid->next.size()));
}

void SubhaloFinder::find_subhalos(
    std::vector<double>& xs, std::vector<double>& ys,
    std::vector<double>& zs, std::vector<double>& rs,
    double mult
) {
    Bounds b;
    std::array<double, 3> pos;
    int c = grid_->cells;
    
    std::vector<int> buf;
    buf.reserve(grid_->max_length());
    
    // Scale radii
    for (auto& r : rs) r *= mult;
    
    std::vector<double> c_max = cumulative_max(rs);
    
    for (size_t ih = 0; ih < rs.size(); ++ih) {
        start_subhalos(static_cast<int>(ih));
        double max_sr = c_max[ih];
        double rh = rs[ih];
        
        pos[0] = xs[ih];
        pos[1] = ys[ih];
        pos[2] = zs[ih];
        
        b.sphere_bounds(pos, rh + max_sr, grid_->cell_width, grid_->box_width);
        
        for (int dz = 0; dz < b.span[2]; ++dz) {
            int z = b.origin[2] + dz;
            if (z >= c) z -= c;
            int z_off = z * c * c;
            
            for (int dy = 0; dy < b.span[1]; ++dy) {
                int y = b.origin[1] + dy;
                if (y >= c) y -= c;
                int y_off = y * c;
                
                for (int dx = 0; dx < b.span[0]; ++dx) {
                    int x = b.origin[0] + dx;
                    if (x >= c) x -= c;
                    
                    int idx = z_off + y_off + x;
                    
                    grid_->read_indices(idx, buf);
                    mark_subhalos(static_cast<int>(ih), buf, xs, ys, zs, rs);
                }
            }
        }
    }
    
    cross_match();
    
    // Restore radii
    for (auto& r : rs) r /= mult;
}

std::pair<int, int> SubhaloFinder::start_end(int ih) const {
    if (ih == static_cast<int>(subhalo_starts_.size()) - 1) {
        return {subhalo_starts_[ih], static_cast<int>(subhalos_.size())};
    } else {
        return {subhalo_starts_[ih], subhalo_starts_[ih + 1]};
    }
}

int SubhaloFinder::intersect_count(int ih) const {
    auto [start, end] = start_end(ih);
    return end - start;
}

int SubhaloFinder::host_count(int ih) const {
    auto is = intersects(ih);
    for (size_t i = 0; i < is.size(); ++i) {
        if (is[i] > ih) return static_cast<int>(i);
    }
    return static_cast<int>(is.size());
}

int SubhaloFinder::subhalo_count(int ih) const {
    return intersect_count(ih) - host_count(ih);
}

std::vector<int> SubhaloFinder::intersects(int ih) const {
    auto [start, end] = start_end(ih);
    return std::vector<int>(subhalos_.begin() + start, subhalos_.begin() + end);
}

std::vector<int> SubhaloFinder::hosts(int ih) const {
    auto is = intersects(ih);
    int hc = host_count(ih);
    is.resize(hc);
    return is;
}

std::vector<int> SubhaloFinder::subhalos(int ih) const {
    auto is = intersects(ih);
    int hc = host_count(ih);
    return std::vector<int>(is.begin() + hc, is.end());
}

void SubhaloFinder::start_subhalos(int i) {
    subhalo_starts_[i] = static_cast<int>(subhalos_.size());
}

void SubhaloFinder::mark_subhalos(
    int ih, const std::vector<int>& idxs,
    const std::vector<double>& xs, const std::vector<double>& ys,
    const std::vector<double>& zs, const std::vector<double>& rs
) {
    double hx = xs[ih];
    double hy = ys[ih];
    double hz = zs[ih];
    double hr = rs[ih];
    
    for (int j : idxs) {
        if (j <= ih) continue;
        
        double sx = xs[j];
        double sy = ys[j];
        double sz = zs[j];
        double sr = rs[j];
        
        double dx = hx - sx;
        double dy = hy - sy;
        double dz = hz - sz;
        double dr = hr + sr;
        
        if (dr * dr >= dx * dx + dy * dy + dz * dz) {
            subhalos_.push_back(j);
        }
    }
}

void SubhaloFinder::cross_match() {
    // Count how many times each halo appears as a subhalo
    std::vector<int> counts(subhalo_starts_.size(), 0);
    for (size_t ih = 0; ih < subhalo_starts_.size(); ++ih) {
        auto [start, end] = start_end(static_cast<int>(ih));
        for (int ish = start; ish < end; ++ish) {
            int sh = subhalos_[ish];
            counts[sh]++;
        }
    }
    
    // Compute host start positions
    std::vector<int> host_starts(counts.size());
    for (size_t i = 1; i < counts.size(); ++i) {
        host_starts[i] = host_starts[i - 1] + counts[i - 1];
    }
    
    // Build hosts array
    std::vector<int> hosts_arr(subhalos_.size());
    for (size_t ih = 0; ih < subhalo_starts_.size(); ++ih) {
        auto [start, end] = start_end(static_cast<int>(ih));
        for (int ish = start; ish < end; ++ish) {
            int sh = subhalos_[ish];
            hosts_arr[host_starts[sh]] = static_cast<int>(ih);
            host_starts[sh]++;
        }
    }
    
    // Reset host_starts
    for (size_t i = 0; i < counts.size(); ++i) {
        host_starts[i] -= counts[i];
    }
    
    // Merge hosts and subhalos
    std::vector<int> sub_starts = subhalo_starts_;
    std::vector<int> subs = subhalos_;
    std::vector<int> new_subhalos(hosts_arr.size() + subhalos_.size());
    std::vector<int> new_starts(sub_starts.size());
    
    int j = 0;
    for (size_t ih = 0; ih < counts.size(); ++ih) {
        new_starts[ih] = j;
        
        // Add hosts first
        subhalo_starts_ = host_starts;
        auto [h_start, h_end] = start_end(static_cast<int>(ih));
        for (int ish = h_start; ish < h_end; ++ish) {
            new_subhalos[j++] = hosts_arr[ish];
        }
        
        // Then add subhalos
        subhalo_starts_ = sub_starts;
        auto [s_start, s_end] = start_end(static_cast<int>(ih));
        for (int ish = s_start; ish < s_end; ++ish) {
            new_subhalos[j++] = subs[ish];
        }
    }
    
    subhalos_ = std::move(new_subhalos);
    subhalo_starts_ = std::move(new_starts);
    
    // Sort each halo's intersections
    for (size_t i = 0; i < subhalo_starts_.size(); ++i) {
        auto [start, end] = start_end(static_cast<int>(i));
        std::sort(subhalos_.begin() + start, subhalos_.begin() + end);
    }
}

} // namespace halo
} // namespace tessera
