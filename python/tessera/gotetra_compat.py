"""
Backward compatibility module for the original gotetra Python interface.

This module provides the same API as the original gotetra.py file,
allowing existing code to work with the new C++ backend.

Usage:
    from tessera.gotetra_compat import read_header, read_grid

    # Or for full compatibility:
    import tessera.gotetra_compat as gotetra
    header = gotetra.read_header("file.gtet")
    grid = gotetra.read_grid("file.gtet")
"""

from __future__ import annotations

import array
import struct
import sys
from typing import Tuple, Union

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

# Quantity constants matching original
DENSITY = 0
DENSITY_GRADIENT = 1
VELOCITY = 2
VELOCITY_DIVERGENCE = 3
VELOCITY_CURL = 4


@define(init=False)
class Sizes:
    """Header size information for different format versions."""
    
    header: int = field()
    type_info: int = field()
    cosmo_info: int = field()
    render_info: int = field()
    location_info: int = field()
    velocity_info: int = field(default=0)
    
    def __init__(self, ver: int) -> None:
        if ver == 1:
            self.__attrs_init__(
                header=232,
                type_info=24,
                cosmo_info=64,
                render_info=40,
                location_info=104,
            )
        elif ver == 2:
            self.__attrs_init__(
                header=336,
                type_info=24,
                cosmo_info=64,
                render_info=40,
                location_info=104,
                velocity_info=104,
            )
        else:
            raise ValueError(f"Unrecognized gotetra output version: {ver}")


def _read_endianness_version(s: bytes) -> Tuple[int, int]:
    flag = struct.unpack("q", s)
    ver = flag[0] & 0xffffffff
    end = -1 if ver >> 31 != 0 else 0
    ver = 1 + (~ver & 0x00000000ffffffff) if end else ver
    return end, ver


def _read_endianness_flag(s: bytes) -> int:
    return _read_endianness_version(s)[0]


def _read_version(s: bytes) -> int:
    return _read_endianness_version(s)[1]


def little_endian(end: int) -> bool:
    return end == -1


def endian_unpack(fmt: str, s: bytes, end: int) -> Tuple[int | float, ...]:
    fmt = f"<{fmt}" if little_endian(end) else f">{fmt}"
    return struct.unpack(fmt, s)


@define(init=False)
class TypeInfo:
    """System information about the file format."""
    
    endianness_flag: int = field()
    header_size: int | float = field()
    grid_type: int | float = field()
    is_vector_grid: bool = field()
    
    def __init__(self, s: bytes, end: int) -> None:
        fmt = "qqq"
        data = endian_unpack(fmt, s, end)
        self.__attrs_init__(
            endianness_flag=end,
            header_size=data[0],
            grid_type=data[1],
            is_vector_grid=data[1] not in [DENSITY, VELOCITY_DIVERGENCE],
        )

    def endianness_str(self) -> str:
        return "Little Endian" if little_endian(self.endianness_flag) else "Big Endian"

    def grid_type_str(self) -> str:
        names = {
            DENSITY: "Density",
            DENSITY_GRADIENT: "Density Gradient",
            VELOCITY: "Velocity",
            VELOCITY_DIVERGENCE: "Velocity Divergence",
            VELOCITY_CURL: "Velocity Curl"
        }
        return names.get(self.grid_type, "Unknown")


@define(init=False)
class CosmoInfo:
    """Cosmological parameters from the simulation."""
    
    redshift: float = field()
    scale_factor: float = field()
    omega_m: float = field()
    omega_l: float = field()
    h0: float = field()
    rho_mean: float = field()
    rho_critical: float = field()
    box_width: float = field()
    
    def __init__(self, s: bytes, end: int) -> None:
        fmt = "d" * 8
        data = endian_unpack(fmt, s, end)
        self.__attrs_init__(
            redshift=data[0],
            scale_factor=data[1],
            omega_m=data[2],
            omega_l=data[3],
            h0=data[4],
            rho_mean=data[5],
            rho_critical=data[6],
            box_width=data[7],
        )


@define(init=False)
class RenderInfo:
    """Rendering parameters used to generate the file."""
    
    particles: int | float = field()
    total_pixels: int | float = field()
    subsample_length: int | float = field()
    min_projection_depth: int | float = field()
    projection_axis: int | float = field()
    
    def __init__(self, s: bytes, end: int) -> None:
        fmt = "qqqqq"
        data = endian_unpack(fmt, s, end)
        self.__attrs_init__(
            particles=data[0],
            total_pixels=data[1],
            subsample_length=data[2],
            min_projection_depth=data[3],
            projection_axis=data[4],
        )


@define(init=False)
class LocationInfo:
    """Physical location and dimensions of the grid."""
    
    origin: NDArray[np.floating] = field(eq=False)
    span: NDArray[np.floating] = field(eq=False)
    pixel_origin: NDArray[np.integer] = field(eq=False)
    pixel_span: NDArray[np.integer] = field(eq=False)
    pixel_width: float = field()
    
    def __init__(self, s: bytes, end: int) -> None:
        fmt = ("d" * 6) + ("q" * 6) + "d"
        data = endian_unpack(fmt, s, end)
        self.__attrs_init__(
            origin=np.array([data[0], data[1], data[2]]),
            span=np.array([data[3], data[4], data[5]]),
            pixel_origin=np.array([data[6], data[7], data[8]]),
            pixel_span=np.array([data[9], data[10], data[11]]),
            pixel_width=data[12],
        )


@define(init=False)
class Header:
    """Complete header information from a gotetra output file."""
    
    version: int = field()
    sizes: Sizes = field()
    type: TypeInfo = field()
    cosmo: CosmoInfo = field()
    render: RenderInfo = field()
    loc: LocationInfo = field()
    dim: NDArray[np.integer] = field(eq=False)
    pw: float = field()
    axis: int | float = field()
    
    def __init__(self, s: bytes, sizes: Sizes) -> None:
        end = _read_endianness_flag(s[:8])
        version = _read_version(s[:8])

        type_start = 8
        type_end = sizes.type_info + type_start
        cosmo_start = type_end
        cosmo_end = cosmo_start + sizes.cosmo_info
        render_start = cosmo_end
        render_end = render_start + sizes.render_info
        loc_start = render_end
        loc_end = render_end + sizes.location_info

        type_info = TypeInfo(s[type_start:type_end], end)
        cosmo_info = CosmoInfo(s[cosmo_start:cosmo_end], end)
        render_info = RenderInfo(s[render_start:render_end], end)
        loc_info = LocationInfo(s[loc_start:loc_end], end)

        self.__attrs_init__(
            version=version,
            sizes=sizes,
            type=type_info,
            cosmo=cosmo_info,
            render=render_info,
            loc=loc_info,
            dim=loc_info.pixel_span,
            pw=loc_info.pixel_width,
            axis=render_info.projection_axis,
        )


def read_header(filename: str) -> Header:
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


def read_grid(filename: str) -> NDArray[np.floating]:
    """
    Read the grid data from a gotetra output file.
    
    Args:
        filename: Path to the .gtet file
        
    Returns:
        numpy.ndarray: 3D array for scalar fields, or tuple of 3 arrays for vectors
    """
    hd = read_header(filename)

    def maybe_swap(xs: array.array[float]) -> None:
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
