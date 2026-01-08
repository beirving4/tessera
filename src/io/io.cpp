#include "io/headers.h"
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace asymptotic_tetra {
namespace io {

bool is_system_little_endian() {
    int32_t test = 1;
    return *reinterpret_cast<int8_t*>(&test) == 1;
}

ByteOrder system_byte_order() {
    return is_system_little_endian() ? ByteOrder::Little : ByteOrder::Big;
}

geom::CellBounds SheetHeader::cell_bounds(int cells) const {
    geom::CellBounds cb;
    double cell_width = total_width / static_cast<double>(cells);

    for (int j = 0; j < 3; ++j) {
        cb.origin[j] = static_cast<int>(std::floor(origin[j] / cell_width));
        cb.width[j] = 1 + static_cast<int>(
            std::floor((origin[j] + width[j]) / cell_width)
        );
        cb.width[j] -= cb.origin[j];
    }

    return cb;
}

CatalogHeader GadgetHeader::standardize() const {
    CatalogHeader h;

    h.count = static_cast<int64_t>(n_part[1]) + 
              (static_cast<int64_t>(n_part[0]) << 32);
    h.total_count = static_cast<int64_t>(n_part_total[1]) + 
                    (static_cast<int64_t>(n_part_total[0]) << 32);
    h.mass = mass[1];
    h.total_width = box_size;
    h.width = -1.0;

    h.cosmo.z = redshift;
    h.cosmo.omega_m = omega0;
    h.cosmo.omega_l = omega_lambda;
    h.cosmo.h100 = hubble_param;

    return h;
}

double GadgetHeader::wrap_distance(double x) const {
    if (x < 0) {
        return x + box_size;
    } else if (x >= box_size) {
        return x - box_size;
    }
    return x;
}

// Helper function for byte swapping
template<typename T>
void byte_swap(T& value) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
    for (size_t i = 0; i < sizeof(T) / 2; ++i) {
        std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
    }
}

// Read int32 with endianness handling
int32_t read_int32(std::ifstream& f, ByteOrder order) {
    int32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if ((order == ByteOrder::Little) != is_system_little_endian()) {
        byte_swap(n);
    }
    return n;
}

// Determine endianness from flag
ByteOrder endianness_from_flag(int32_t flag) {
    if (flag == ENDIANNESS_LITTLE) {
        return ByteOrder::Little;
    } else if (flag == ENDIANNESS_BIG) {
        return ByteOrder::Big;
    }
    throw std::runtime_error("Unrecognized endianness flag");
}

} // namespace io
} // namespace asymptotic_tetra
