#!/usr/bin/env python3
"""
3D Visualization for Halo Evolution

Generate volume renderings and isosurface visualizations of dark matter halos.
Requires PyVista: pip install pyvista
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


@dataclass
class Render3DConfig:
    """Configuration for 3D rendering."""

    # Rendering mode
    mode: str = 'volume'
    """Rendering mode: 'volume', 'isosurface', or 'both'."""

    # Volume rendering settings
    opacity: str = 'sigmoid_5'
    """Opacity transfer function: 'linear', 'sigmoid', 'sigmoid_5', 'sigmoid_10', 'geom'."""

    cmap: str = 'viridis'
    """Colormap for volume rendering."""

    # Isosurface settings
    isosurface_levels: List[float] = field(default_factory=lambda: [10, 100, 1000])
    """Overdensity levels (1+delta) for isosurfaces."""

    isosurface_opacity: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])
    """Opacity for each isosurface level."""

    isosurface_colors: List[str] = field(default_factory=lambda: ['blue', 'green', 'yellow'])
    """Colors for each isosurface level."""

    # Camera settings
    camera_position: str = 'iso'
    """Camera position: 'iso', 'xy', 'xz', 'yz', or custom (x, y, z) tuple."""

    zoom: float = 1.2
    """Camera zoom factor."""

    # Output settings
    window_size: Tuple[int, int] = (1024, 1024)
    """Output image size in pixels."""

    background: str = 'white'
    """Background color."""

    show_axes: bool = True
    """Show coordinate axes."""

    show_bounds: bool = False
    """Show bounding box."""

    # Animation settings
    n_rotation_frames: int = 36
    """Number of frames for rotation animation."""

    rotation_axis: str = 'z'
    """Axis to rotate around for animation."""


def _check_pyvista():
    """Check if PyVista is available."""
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista is required for 3D visualization. "
            "Install with: pip install pyvista"
        )


def density_to_pyvista(
    density: np.ndarray,
    box_size: float,
    center_on_origin: bool = True,
    as_point_data: bool = False,
) -> "pv.ImageData":
    """
    Convert a 3D density array to a PyVista ImageData object.

    Parameters
    ----------
    density : np.ndarray
        3D density array with shape (nx, ny, nz)
    box_size : float
        Physical size of the box
    center_on_origin : bool
        If True, center the grid on the origin
    as_point_data : bool
        If True, store as point data (required for contour/isosurface).
        If False, store as cell data (better for volume rendering).

    Returns
    -------
    grid : pv.ImageData
        PyVista grid with density data
    """
    _check_pyvista()

    nx, ny, nz = density.shape
    spacing = box_size / nx

    if as_point_data:
        # For point data, dimensions match the data shape
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(spacing, spacing, spacing),
        )
        if center_on_origin:
            grid.origin = (-box_size / 2, -box_size / 2, -box_size / 2)
        grid.point_data['density'] = density.flatten(order='F')
    else:
        # For cell data, dimensions are +1 in each direction
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(spacing, spacing, spacing),
        )
        if center_on_origin:
            grid.origin = (-box_size / 2, -box_size / 2, -box_size / 2)
        grid.cell_data['density'] = density.flatten(order='F')

    return grid


def render_volume(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    config: Optional[Render3DConfig] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create a volume rendering of a 3D density field.

    Parameters
    ----------
    density : np.ndarray
        3D density array
    output_path : Path
        Output image path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    config : Render3DConfig, optional
        Rendering configuration
    title : str, optional
        Title for the plot
    """
    _check_pyvista()

    cfg = config or Render3DConfig()

    # Convert to overdensity
    overdensity = density / mean_density

    # Create PyVista grid
    grid = density_to_pyvista(overdensity, box_size)

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=cfg.window_size)
    plotter.set_background(cfg.background)

    # Add volume
    plotter.add_volume(
        grid,
        scalars='density',
        cmap=cfg.cmap,
        opacity=cfg.opacity,
        shade=True,
    )

    # Configure camera
    if cfg.camera_position == 'iso':
        plotter.camera_position = 'iso'
    elif cfg.camera_position in ('xy', 'xz', 'yz'):
        plotter.camera_position = cfg.camera_position
    else:
        plotter.camera_position = cfg.camera_position

    plotter.camera.zoom(cfg.zoom)

    # Add decorations
    if cfg.show_axes:
        plotter.add_axes()

    if cfg.show_bounds:
        plotter.add_bounding_box()

    if title:
        plotter.add_title(title, font_size=14)

    # Save
    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved volume rendering to {output_path}")


def render_isosurface(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    config: Optional[Render3DConfig] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create an isosurface rendering of a 3D density field.

    Parameters
    ----------
    density : np.ndarray
        3D density array
    output_path : Path
        Output image path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    config : Render3DConfig, optional
        Rendering configuration
    title : str, optional
        Title for the plot
    """
    _check_pyvista()

    cfg = config or Render3DConfig()

    # Convert to overdensity
    overdensity = density / mean_density

    # Create PyVista grid with point data (required for contour)
    grid = density_to_pyvista(overdensity, box_size, as_point_data=True)

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=cfg.window_size)
    plotter.set_background(cfg.background)

    # Add isosurfaces
    for level, opacity, color in zip(
        cfg.isosurface_levels,
        cfg.isosurface_opacity,
        cfg.isosurface_colors,
    ):
        try:
            contour = grid.contour([level], scalars='density')
            if contour.n_points > 0:
                plotter.add_mesh(
                    contour,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                )
        except Exception as e:
            # Skip if isosurface level doesn't exist in data
            pass

    # Configure camera
    if cfg.camera_position == 'iso':
        plotter.camera_position = 'iso'
    elif cfg.camera_position in ('xy', 'xz', 'yz'):
        plotter.camera_position = cfg.camera_position
    else:
        plotter.camera_position = cfg.camera_position

    plotter.camera.zoom(cfg.zoom)

    # Add decorations
    if cfg.show_axes:
        plotter.add_axes()

    if cfg.show_bounds:
        plotter.add_bounding_box()

    if title:
        plotter.add_title(title, font_size=14)

    # Save
    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved isosurface rendering to {output_path}")


def render_3d(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    config: Optional[Render3DConfig] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create a 3D rendering (volume, isosurface, or both).

    Parameters
    ----------
    density : np.ndarray
        3D density array
    output_path : Path
        Output image path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    config : Render3DConfig, optional
        Rendering configuration
    title : str, optional
        Title for the plot
    """
    cfg = config or Render3DConfig()

    if cfg.mode == 'volume':
        render_volume(density, output_path, box_size, mean_density, cfg, title)
    elif cfg.mode == 'isosurface':
        render_isosurface(density, output_path, box_size, mean_density, cfg, title)
    elif cfg.mode == 'both':
        # Combined rendering
        _render_combined(density, output_path, box_size, mean_density, cfg, title)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def _render_combined(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float,
    config: Render3DConfig,
    title: Optional[str],
) -> None:
    """Render both volume and isosurfaces together."""
    _check_pyvista()

    # Convert to overdensity
    overdensity = density / mean_density

    # Create PyVista grid with point data (required for contour)
    grid = density_to_pyvista(overdensity, box_size, as_point_data=True)

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=config.window_size)
    plotter.set_background(config.background)

    # Add volume (with reduced opacity)
    plotter.add_volume(
        grid,
        scalars='density',
        cmap=config.cmap,
        opacity='linear',
        shade=True,
    )

    # Add isosurfaces
    for level, opacity, color in zip(
        config.isosurface_levels,
        config.isosurface_opacity,
        config.isosurface_colors,
    ):
        try:
            contour = grid.contour([level], scalars='density')
            if contour.n_points > 0:
                plotter.add_mesh(
                    contour,
                    color=color,
                    opacity=opacity * 0.5,  # Reduce for combined view
                    smooth_shading=True,
                )
        except Exception:
            pass

    # Configure camera
    plotter.camera_position = 'iso'
    plotter.camera.zoom(config.zoom)

    if config.show_axes:
        plotter.add_axes()

    if title:
        plotter.add_title(title, font_size=14)

    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved combined 3D rendering to {output_path}")


def create_rotation_animation(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    config: Optional[Render3DConfig] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create a rotating animation of the 3D density field.

    Parameters
    ----------
    density : np.ndarray
        3D density array
    output_path : Path
        Output GIF or MP4 path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    config : Render3DConfig, optional
        Rendering configuration
    title : str, optional
        Title for the plot
    """
    _check_pyvista()

    cfg = config or Render3DConfig()

    # Convert to overdensity
    overdensity = density / mean_density

    # Create PyVista grid (use point data if mode includes isosurface)
    use_point_data = cfg.mode in ('isosurface', 'both')
    grid = density_to_pyvista(overdensity, box_size, as_point_data=use_point_data)

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=cfg.window_size)
    plotter.set_background(cfg.background)

    # Add visualization based on mode
    if cfg.mode in ('volume', 'both'):
        plotter.add_volume(
            grid,
            scalars='density',
            cmap=cfg.cmap,
            opacity=cfg.opacity,
            shade=True,
        )

    if cfg.mode in ('isosurface', 'both'):
        for level, opacity, color in zip(
            cfg.isosurface_levels,
            cfg.isosurface_opacity,
            cfg.isosurface_colors,
        ):
            try:
                contour = grid.contour([level], scalars='density')
                if contour.n_points > 0:
                    plotter.add_mesh(
                        contour,
                        color=color,
                        opacity=opacity,
                        smooth_shading=True,
                    )
            except Exception:
                pass

    if cfg.show_axes:
        plotter.add_axes()

    if title:
        plotter.add_title(title, font_size=14)

    # Set initial camera position
    plotter.camera_position = 'iso'
    plotter.camera.zoom(cfg.zoom)

    # Create animation
    output_path = Path(output_path)
    print(f"Creating rotation animation ({cfg.n_rotation_frames} frames)...")

    plotter.open_gif(str(output_path)) if output_path.suffix == '.gif' else None

    # Rotate and capture frames
    for i in range(cfg.n_rotation_frames):
        angle = 360 * i / cfg.n_rotation_frames
        plotter.camera.azimuth = angle
        plotter.write_frame() if output_path.suffix == '.gif' else None

    plotter.close()

    print(f"Saved rotation animation to {output_path}")


def create_multiview_figure(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    config: Optional[Render3DConfig] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create a multi-view figure showing the halo from different angles.

    Parameters
    ----------
    density : np.ndarray
        3D density array
    output_path : Path
        Output image path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    config : Render3DConfig, optional
        Rendering configuration
    title : str, optional
        Title for the plot
    """
    _check_pyvista()

    cfg = config or Render3DConfig()

    # Convert to overdensity
    overdensity = density / mean_density

    # Create PyVista grid (use point data if mode includes isosurface)
    use_point_data = cfg.mode in ('isosurface', 'both')
    grid = density_to_pyvista(overdensity, box_size, as_point_data=use_point_data)

    # Create 2x2 subplot
    plotter = pv.Plotter(
        off_screen=True,
        shape=(2, 2),
        window_size=(cfg.window_size[0] * 2, cfg.window_size[1] * 2),
    )

    views = [
        ('iso', 'Isometric'),
        ('xy', 'Top (XY)'),
        ('xz', 'Front (XZ)'),
        ('yz', 'Side (YZ)'),
    ]

    for idx, (view, view_title) in enumerate(views):
        row, col = divmod(idx, 2)
        plotter.subplot(row, col)
        plotter.set_background(cfg.background)

        # Add volume or isosurfaces
        if cfg.mode in ('volume', 'both'):
            plotter.add_volume(
                grid,
                scalars='density',
                cmap=cfg.cmap,
                opacity=cfg.opacity,
                shade=True,
            )

        if cfg.mode in ('isosurface', 'both'):
            for level, opacity, color in zip(
                cfg.isosurface_levels,
                cfg.isosurface_opacity,
                cfg.isosurface_colors,
            ):
                try:
                    contour = grid.contour([level], scalars='density')
                    if contour.n_points > 0:
                        plotter.add_mesh(
                            contour,
                            color=color,
                            opacity=opacity,
                            smooth_shading=True,
                        )
                except Exception:
                    pass

        plotter.camera_position = view
        plotter.camera.zoom(cfg.zoom)
        plotter.add_title(view_title, font_size=12)

        if cfg.show_axes:
            plotter.add_axes()

    if title:
        plotter.add_title(title, font_size=16)

    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved multi-view figure to {output_path}")


def create_evolution_3d_figure(
    densities: Dict[float, np.ndarray],
    output_path: Path,
    box_size: float,
    mean_density: float = 1.0,
    epochs: Optional[List[float]] = None,
    config: Optional[Render3DConfig] = None,
) -> None:
    """
    Create a multi-panel figure showing 3D halo evolution across epochs.

    Parameters
    ----------
    densities : dict
        Mapping of scale_factor -> 3D density array
    output_path : Path
        Output image path
    box_size : float
        Physical size of the box
    mean_density : float
        Mean density for overdensity calculation
    epochs : list, optional
        Scale factors to show (default: all)
    config : Render3DConfig, optional
        Rendering configuration
    """
    _check_pyvista()

    cfg = config or Render3DConfig()

    # Select epochs
    available = sorted(densities.keys())
    if epochs is None:
        epochs_to_plot = available
    else:
        epochs_to_plot = []
        for target in epochs:
            nearest = min(available, key=lambda x: abs(x - target))
            if nearest not in epochs_to_plot:
                epochs_to_plot.append(nearest)

    n_panels = len(epochs_to_plot)
    if n_panels == 0:
        print("No epochs to plot")
        return

    # Create subplot grid
    n_cols = min(n_panels, 4)
    n_rows = (n_panels + n_cols - 1) // n_cols

    plotter = pv.Plotter(
        off_screen=True,
        shape=(n_rows, n_cols),
        window_size=(cfg.window_size[0] * n_cols, cfg.window_size[1] * n_rows),
    )

    # Determine if we need point data (for isosurface mode)
    use_point_data = cfg.mode in ('isosurface', 'both')

    for idx, epoch in enumerate(epochs_to_plot):
        row, col = divmod(idx, n_cols)
        plotter.subplot(row, col)
        plotter.set_background(cfg.background)

        density = densities[epoch]
        overdensity = density / mean_density
        grid = density_to_pyvista(overdensity, box_size, as_point_data=use_point_data)

        # Add visualization
        if cfg.mode in ('volume', 'both'):
            plotter.add_volume(
                grid,
                scalars='density',
                cmap=cfg.cmap,
                opacity=cfg.opacity,
                shade=True,
            )

        if cfg.mode in ('isosurface', 'both'):
            for level, opacity, color in zip(
                cfg.isosurface_levels,
                cfg.isosurface_opacity,
                cfg.isosurface_colors,
            ):
                try:
                    contour = grid.contour([level], scalars='density')
                    if contour.n_points > 0:
                        plotter.add_mesh(
                            contour,
                            color=color,
                            opacity=opacity,
                            smooth_shading=True,
                        )
                except Exception:
                    pass

        plotter.camera_position = 'iso'
        plotter.camera.zoom(cfg.zoom)
        plotter.add_title(f'$a = {epoch:.2f}$', font_size=12)

        if cfg.show_axes:
            plotter.add_axes()

    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved 3D evolution figure to {output_path}")
