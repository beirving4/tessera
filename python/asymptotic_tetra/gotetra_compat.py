"""
Backward compatibility module for the original gotetra Python interface.

This module provides the same API as the original gotetra.py file,
allowing existing code to work with the new C++ backend.

Usage:
    from asymptotic_tetra.gotetra_compat import read_header, read_grid
    
    # Or for full compatibility:
    import asymptotic_tetra.gotetra_compat as gotetra
    header = gotetra.read_header("file.gtet")
    grid = gotetra.read_grid("file.gtet")
"""

import array
import struct
import sys
import numpy as np

# Quantity constants matching original
DENSITY = 0
DENSITY_GRADIENT = 1
VELOCITY = 2
VELOCITY_DIVERGENCE = 3
VELOCITY_CURL = 4


class Sizes:
    """Header size information for different format versions."""
    
    def __init__(self, ver):
        if ver == 1:
            self.header = 232
            self.type_info = 24
            self.cosmo_info = 64
            self.render_info = 40
            self.location_info = 104
        elif ver == 2:
            self.header = 336
            self.type_info = 24
            self.cosmo_info = 64
            self.render_info = 40
            self.location_info = 104
            self.velocity_info = 104
        else:
            raise ValueError(f"Unrecognized gotetra output version: {ver}")


def _read_endianness_version(s):
    flag = struct.unpack("q", s)
    ver = flag[0] & 0xffffffff
    end = -1 if ver >> 31 != 0 else 0
    ver = 1 + (~ver & 0x00000000ffffffff) if end else ver
    return end, ver


def _read_endianness_flag(s):
    return _read_endianness_version(s)[0]


def _read_version(s):
    return _read_endianness_version(s)[1]


def little_endian(end):
    return end == -1


def endian_unpack(fmt, s, end):
    fmt = "<" + fmt if little_endian(end) else ">" + fmt
    return struct.unpack(fmt, s)


class TypeInfo:
    """System information about the file format."""
    
    def __init__(self, s, end):
        self.endianness_flag = end
        fmt = "qqq"
        data = endian_unpack(fmt, s, self.endianness_flag)
        self.header_size = data[0]
        self.grid_type = data[1]
        self.is_vector_grid = not (self.grid_type == DENSITY or 
                                   self.grid_type == VELOCITY_DIVERGENCE)

    def endianness_str(self):
        return "Little Endian" if little_endian(self.endianness_flag) else "Big Endian"

    def grid_type_str(self):
        names = {
            DENSITY: "Density",
            DENSITY_GRADIENT: "Density Gradient",
            VELOCITY: "Velocity",
            VELOCITY_DIVERGENCE: "Velocity Divergence",
            VELOCITY_CURL: "Velocity Curl"
        }
        return names.get(self.grid_type, "Unknown")


class CosmoInfo:
    """Cosmological parameters from the simulation."""
    
    def __init__(self, s, end):
        fmt = "d" * 8
        data = endian_unpack(fmt, s, end)
        self.redshift = data[0]
        self.scale_factor = data[1]
        self.omega_m = data[2]
        self.omega_l = data[3]
        self.h0 = data[4]
        self.rho_mean = data[5]
        self.rho_critical = data[6]
        self.box_width = data[7]


class RenderInfo:
    """Rendering parameters used to generate the file."""
    
    def __init__(self, s, end):
        fmt = "qqqqq"
        data = endian_unpack(fmt, s, end)
        self.particles = data[0]
        self.total_pixels = data[1]
        self.subsample_length = data[2]
        self.min_projection_depth = data[3]
        self.projection_axis = data[4]


class LocationInfo:
    """Physical location and dimensions of the grid."""
    
    def __init__(self, s, end):
        fmt = ("d" * 6) + ("q" * 6) + "d"
        data = endian_unpack(fmt, s, end)
        self.origin = np.array([data[0], data[1], data[2]])
        self.span = np.array([data[3], data[4], data[5]])
        self.pixel_origin = np.array([data[6], data[7], data[8]])
        self.pixel_span = np.array([data[9], data[10], data[11]])
        self.pixel_width = data[12]


class Header:
    """Complete header information from a gotetra output file."""
    
    def __init__(self, s, sizes):
        end = _read_endianness_flag(s[0:8])
        self.version = _read_version(s[0:8])
        self.sizes = sizes

        type_start = 8
        type_end = sizes.type_info + type_start
        cosmo_start = type_end
        cosmo_end = cosmo_start + sizes.cosmo_info
        render_start = cosmo_end
        render_end = render_start + sizes.render_info
        loc_start = render_end
        loc_end = render_end + sizes.location_info

        self.type = TypeInfo(s[type_start:type_end], end)
        self.cosmo = CosmoInfo(s[cosmo_start:cosmo_end], end)
        self.render = RenderInfo(s[render_start:render_end], end)
        self.loc = LocationInfo(s[loc_start:loc_end], end)

        self.dim = self.loc.pixel_span
        self.pw = self.loc.pixel_width
        self.axis = self.render.projection_axis


def read_header(filename):
    """
    Read the header information from a gotetra output file.
    
    Args:
        filename: Path to the .gtet file
        
    Returns:
        Header object with file metadata
    """
    with open(filename, "rb") as fp:
        flag_s = fp.read(8)
        ver = _read_version(flag_s)
        sizes = Sizes(ver)
        s = fp.read(sizes.header)
    return Header(flag_s + s, sizes)


def read_grid(filename):
    """
    Read the grid data from a gotetra output file.
    
    Args:
        filename: Path to the .gtet file
        
    Returns:
        numpy.ndarray: 3D array for scalar fields, or tuple of 3 arrays for vectors
    """
    hd = read_header(filename)

    def maybe_swap(xs):
        endianness = sys.byteorder
        if endianness == "little" and hd.type.endianness_flag == -1:
            return
        elif endianness == "big" and hd.type.endianness_flag == 0:
            return
        xs.byteswap()

    n = 1
    for i in range(3):
        if i != hd.axis:
            n *= hd.dim[i]

    if hd.axis == 0:
        j, k = 1, 2
    elif hd.axis == 1:
        j, k = 0, 2
    else:
        j, k = 0, 1

    if hd.type.is_vector_grid:
        xs, ys, zs = array.array("f"), array.array("f"), array.array("f")
        with open(filename, "rb") as fp:
            fp.read(hd.sizes.header + 8)
            xs.fromfile(fp, n)
            ys.fromfile(fp, n)
            zs.fromfile(fp, n)

        maybe_swap(xs)
        maybe_swap(ys)
        maybe_swap(zs)

        if hd.axis == -1:
            xs = np.reshape(xs, (hd.dim[2], hd.dim[1], hd.dim[0]))
            ys = np.reshape(ys, (hd.dim[2], hd.dim[1], hd.dim[0]))
            zs = np.reshape(zs, (hd.dim[2], hd.dim[1], hd.dim[0]))
        else:
            xs = np.reshape(xs, (hd.dim[k], hd.dim[j]))
            ys = np.reshape(ys, (hd.dim[k], hd.dim[j]))
            zs = np.reshape(zs, (hd.dim[k], hd.dim[j]))

        return np.array([xs, ys, zs])
    else:
        xs = array.array("f")
        with open(filename, "rb") as fp:
            fp.read(hd.sizes.header + 8)
            xs.fromfile(fp, n)

        maybe_swap(xs)
        if hd.axis == -1:
            xs = np.reshape(xs, (hd.dim[2], hd.dim[1], hd.dim[0]))
        else:
            xs = np.reshape(xs, (hd.dim[k], hd.dim[j]))
        return xs


# For command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]} requires a target file")
        sys.exit(1)
    
    print(read_header(sys.argv[1]))
