"""
Fourier-space operations for periodic density fields.

Provides FFT-based smoothing of 3D grids at arbitrary scales, spherically-averaged
power spectrum P(k) computation, and variance σ_R² from both real-space fields and
P(k) integration. Built on numpy.fft — no FFTW dependency required.

Designed to work with CIC density grids from ``ts.density.compute_cic_density()``.
The CIC grid is returned as a flat array; reshape to (N, N, N) before passing here.

Example usage:
    >>> import tessera as ts
    >>> import numpy as np
    >>> from tessera.utils.fourier import smooth_field_fourier, compute_power_spectrum
    >>>
    >>> # Compute CIC density, convert to overdensity
    >>> result = ts.density.compute_cic_density(positions, box_size=256.0, output_cells=256)
    >>> grid = np.array(result.grid_density).reshape(256, 256, 256)
    >>> delta = grid / grid.mean() - 1.0
    >>>
    >>> # Smooth at scale R = 4 Mpc/h
    >>> delta_R = smooth_field_fourier(delta, box_size=256.0, R=4.0)
    >>>
    >>> # Compute power spectrum
    >>> pk = compute_power_spectrum(delta, box_size=256.0, deconvolve_cic=True)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def gaussian_window(k: NDArray | float, R: float) -> NDArray | float:
    """Gaussian smoothing window in Fourier space.

    W_G(kR) = exp(-k²R²/2)

    Parameters
    ----------
    k : array_like
        Wavenumber magnitude(s).
    R : float
        Smoothing scale (same units as 1/k).
    """
    return np.exp(-0.5 * (k * R) ** 2)


def tophat_window(k: NDArray | float, R: float) -> NDArray | float:
    """Real-space top-hat smoothing window in Fourier space.

    W_TH(kR) = 3 [sin(kR) - kR cos(kR)] / (kR)³

    Handles kR → 0 limit (W → 1) without division by zero.

    Parameters
    ----------
    k : array_like
        Wavenumber magnitude(s).
    R : float
        Smoothing scale (same units as 1/k).
    """
    kR = np.asarray(k * R, dtype=np.float64)
    result = np.ones_like(kR)
    nonzero = kR > 1e-10
    x = kR[nonzero]
    result[nonzero] = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return result


def cic_window_3d(
    kx: NDArray, ky: NDArray, kz: NDArray, cell_width: float
) -> NDArray:
    """CIC (Cloud-in-Cell) assignment window in 3D Fourier space.

    W_CIC(k) = Π_i sinc(k_i Δ/2)²

    where sinc(x) = sin(x)/x (unnormalized) and Δ = cell_width.

    Parameters
    ----------
    kx, ky, kz : ndarray
        Wavenumber components (broadcastable).
    cell_width : float
        Grid cell width Δ = L / N_grid.
    """
    # np.sinc uses the normalized convention: sinc(x) = sin(πx)/(πx)
    # We need unnormalized sinc: sin(x)/x, so pass x/π to np.sinc
    half_delta = cell_width / 2.0
    wx = np.sinc(kx * half_delta / np.pi) ** 2
    wy = np.sinc(ky * half_delta / np.pi) ** 2
    wz = np.sinc(kz * half_delta / np.pi) ** 2
    return wx * wy * wz


# ---------------------------------------------------------------------------
# k-grid construction (internal)
# ---------------------------------------------------------------------------

def _build_k_grid_rfft(N: int, box_size: float):
    """Build 3D wavenumber grid for rfftn output.

    Returns kx, ky, kz arrays (broadcastable via meshgrid indexing='ij')
    and the scalar k magnitude grid.

    The last axis uses rfftfreq (length N//2+1), the first two use fftfreq.
    """
    dk = 2.0 * np.pi / box_size  # fundamental mode spacing

    freq_full = np.fft.fftfreq(N, d=1.0 / N)   # integer frequencies 0..N/2, -N/2+1..-1
    freq_half = np.fft.rfftfreq(N, d=1.0 / N)   # integer frequencies 0..N/2

    kx_1d = freq_full * dk
    ky_1d = freq_full * dk
    kz_1d = freq_half * dk

    kx, ky, kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    return kx, ky, kz, k_mag


# ---------------------------------------------------------------------------
# FFT smoothing
# ---------------------------------------------------------------------------

def smooth_field_fourier(
    grid: NDArray[np.floating],
    box_size: float,
    R: float,
    window: str = 'gaussian',
) -> NDArray[np.floating]:
    """Apply Fourier-space smoothing to a 3D periodic density/overdensity grid.

    1. FFT the input grid (real-to-complex via rfftn)
    2. Build 3D k-grid from box_size and grid shape
    3. Multiply by W(kR) in Fourier space
    4. Inverse FFT back to real space

    Parameters
    ----------
    grid : ndarray, shape (N, N, N)
        3D density or overdensity field.
    box_size : float
        Physical box size (determines k spacing).
    R : float
        Smoothing scale in same units as box_size.
    window : str
        Window function: ``'gaussian'`` or ``'tophat'``.

    Returns
    -------
    smoothed : ndarray, shape (N, N, N)
        Smoothed field (real-valued).

    Notes
    -----
    Smoothing is a linear operation, so it works on either ρ(x) or δ(x).
    Uses ``np.fft.rfftn``/``irfftn`` for the real-to-complex optimization.
    """
    grid = np.asarray(grid, dtype=np.float64)
    N = grid.shape[0]
    assert grid.shape == (N, N, N), f"Expected cubic grid, got shape {grid.shape}"

    # Forward FFT
    grid_k = np.fft.rfftn(grid)

    # Build k-grid
    _, _, _, k_mag = _build_k_grid_rfft(N, box_size)

    # Apply window
    if window == 'gaussian':
        W = gaussian_window(k_mag, R)
    elif window == 'tophat':
        W = tophat_window(k_mag, R)
    else:
        raise ValueError(f"Unknown window type: {window!r}. Use 'gaussian' or 'tophat'.")

    grid_k *= W

    # Inverse FFT
    smoothed = np.fft.irfftn(grid_k, s=(N, N, N))

    return smoothed


# ---------------------------------------------------------------------------
# Power spectrum
# ---------------------------------------------------------------------------

def compute_power_spectrum(
    grid: NDArray[np.floating],
    box_size: float,
    n_bins: int | None = None,
    deconvolve_cic: bool = False,
    subtract_shot_noise: bool = False,
    n_particles: int | None = None,
) -> dict:
    """Compute spherically-averaged P(k) from a 3D periodic overdensity field.

    Parameters
    ----------
    grid : ndarray, shape (N, N, N)
        3D overdensity field δ(x) = ρ/ρ̄ − 1.
    box_size : float
        Physical box side length L (determines k spacing).
    n_bins : int, optional
        Number of radial k-bins. Default: N // 2.
    deconvolve_cic : bool
        If True, deconvolve the CIC window function |W_CIC(k)|².
    subtract_shot_noise : bool
        If True, subtract shot noise P_shot = L³/N_particles.
        Requires ``n_particles``.
    n_particles : int, optional
        Number of particles (for shot noise subtraction).

    Returns
    -------
    result : dict
        ``'k'``
            Bin centers in wavenumber units (h/Mpc if box_size in Mpc/h).
        ``'Pk'``
            P(k) in volume units ((Mpc/h)³ if box_size in Mpc/h).
        ``'delta_squared'``
            Dimensionless power Δ²(k) = k³ P(k) / (2π²).
        ``'n_modes'``
            Number of Fourier modes per bin.
        ``'k_fundamental'``
            Fundamental mode 2π/L.
        ``'k_nyquist'``
            Grid Nyquist frequency π N / L.

    Notes
    -----
    Normalization: ``P(k) = V / N_grid² × ⟨|FFT[δ]|²⟩_shell``, where V = L³
    and N_grid = N³ is the total number of grid cells.
    """
    grid = np.asarray(grid, dtype=np.float64)
    N = grid.shape[0]
    assert grid.shape == (N, N, N), f"Expected cubic grid, got shape {grid.shape}"

    if n_bins is None:
        n_bins = N // 2

    V = box_size ** 3
    N_grid = N ** 3
    k_fundamental = 2.0 * np.pi / box_size
    k_nyquist = np.pi * N / box_size
    cell_width = box_size / N

    # Forward FFT (real-to-complex)
    delta_k = np.fft.rfftn(grid)

    # Build k-grid
    kx, ky, kz, k_mag = _build_k_grid_rfft(N, box_size)

    # Power: |δ(k)|² with volume normalization
    # P(k) = V / N_grid² * |FFT[δ]|²
    power = (V / N_grid ** 2) * np.abs(delta_k) ** 2

    # CIC deconvolution: divide by |W_CIC(k)|²
    if deconvolve_cic:
        w_cic = cic_window_3d(kx, ky, kz, cell_width)
        # Avoid amplifying noise near Nyquist — only deconvolve where window > threshold
        safe = w_cic > 0.01
        power[safe] /= w_cic[safe] ** 2
        # Zero out modes beyond Nyquist where CIC correction is unreliable
        power[~safe] = 0.0

    # Accounting for rfft symmetry: modes with kz > 0 and kz < k_Nyq are
    # represented once in the rfft but appear twice in the full FFT.
    # We need to weight them by 2 for correct shell averaging.
    rfft_weight = np.full_like(k_mag, 2.0)
    rfft_weight[:, :, 0] = 1.0           # kz = 0 plane
    if N % 2 == 0:
        rfft_weight[:, :, N // 2] = 1.0  # kz = Nyquist plane

    # Shell-average in linear k-bins
    bin_edges = np.linspace(0, k_nyquist, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Digitize k values into bins (0 = below first edge, n_bins = above last)
    k_flat = k_mag.ravel()
    power_flat = power.ravel()
    weight_flat = rfft_weight.ravel()

    bin_idx = np.digitize(k_flat, bin_edges) - 1  # 0-indexed bins

    Pk = np.zeros(n_bins)
    n_modes = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        in_bin = bin_idx == i
        if np.any(in_bin):
            weights_in_bin = weight_flat[in_bin]
            Pk[i] = np.sum(power_flat[in_bin] * weights_in_bin) / np.sum(weights_in_bin)
            n_modes[i] = int(np.sum(weights_in_bin))

    # Shot noise subtraction
    if subtract_shot_noise:
        if n_particles is None:
            raise ValueError("n_particles required for shot noise subtraction")
        P_shot = V / n_particles
        Pk -= P_shot

    # Exclude empty bins and DC mode
    valid = (n_modes > 0) & (bin_centers > 0)

    # Dimensionless power
    delta_squared = np.zeros(n_bins)
    delta_squared[valid] = bin_centers[valid] ** 3 * Pk[valid] / (2.0 * np.pi ** 2)

    return {
        'k': bin_centers[valid],
        'Pk': Pk[valid],
        'delta_squared': delta_squared[valid],
        'n_modes': n_modes[valid],
        'k_fundamental': k_fundamental,
        'k_nyquist': k_nyquist,
    }


# ---------------------------------------------------------------------------
# Variance σ_R²
# ---------------------------------------------------------------------------

def sigma_R_squared_from_pk(
    k: NDArray[np.floating],
    Pk: NDArray[np.floating],
    R: float,
    window: str = 'gaussian',
    cic_cell_width: float | None = None,
) -> float:
    """Compute σ_R² by integrating P(k) with an effective window function.

    σ_R² = (1/2π²) ∫ dk k² P(k) |W_eff(k)|²

    where W_eff = W_smooth(kR) × W_CIC(k) if ``cic_cell_width`` is given,
    otherwise W_eff = W_smooth(kR).

    Uses the trapezoidal rule for numerical integration.

    Parameters
    ----------
    k : ndarray
        Wavenumber array.
    Pk : ndarray
        Power spectrum P(k) at those wavenumbers.
    R : float
        Smoothing scale.
    window : str
        ``'gaussian'`` or ``'tophat'``.
    cic_cell_width : float, optional
        If given, include the 1D CIC window sinc²(k Δ/2) for each axis
        in the effective window. This accounts for the CIC grid assignment
        when comparing to a smoothed CIC field.
    """
    k = np.asarray(k, dtype=np.float64)
    Pk = np.asarray(Pk, dtype=np.float64)

    if window == 'gaussian':
        W = gaussian_window(k, R)
    elif window == 'tophat':
        W = tophat_window(k, R)
    else:
        raise ValueError(f"Unknown window: {window!r}")

    W_eff_sq = W ** 2

    # Include CIC window (isotropic approximation: |W_CIC|^2 ≈ sinc(k Δ/2)^6
    # since the full 3D CIC is product of 3 axes, and for spherical averaging
    # we approximate each axis contribution as sinc(k Δ/(2√3))^2 ...
    # Actually, the correct isotropic approximation for the angle-averaged
    # CIC window squared is: <|W_CIC|^2> = <Π sinc^4(k_i Δ/2)>
    # For spherical averaging, a standard approximation is sinc^2(kΔ/2)^3
    # but more precisely we use the 1D CIC per axis averaged.
    # For consistency with the binned P(k) deconvolution, we use the
    # isotropic approximation: sinc(kΔ/2)^6
    if cic_cell_width is not None:
        half_delta = cic_cell_width / 2.0
        w_cic_1d = np.sinc(k * half_delta / np.pi) ** 2  # one axis CIC
        W_eff_sq *= w_cic_1d ** 3  # three axes

    integrand = k ** 2 * Pk * W_eff_sq / (2.0 * np.pi ** 2)
    return float(np.trapz(integrand, k))


def sigma_R_squared_from_field(grid: NDArray[np.floating]) -> float:
    """Compute variance σ² = ⟨δ²⟩ − ⟨δ⟩² directly from a grid.

    Parameters
    ----------
    grid : ndarray
        Overdensity field δ(x) (any shape).

    Returns
    -------
    variance : float
        Field variance.
    """
    grid = np.asarray(grid, dtype=np.float64)
    return float(np.var(grid))


# ---------------------------------------------------------------------------
# CIC grid configuration
# ---------------------------------------------------------------------------

def recommend_cic_config(
    n_particles: int,
    box_size: float,
    softening: float | None = None,
    min_particles_per_cell: int = 20,
    prefer_power_of_two: bool = True,
) -> dict:
    """Recommend CIC grid resolution for defensible density PDFs.

    Applies the numerical constraints from Klypin et al. (2018, MNRAS 481, 4588)
    to select the largest grid that avoids discreteness artifacts in the
    volume-weighted density PDF:

    1. **Force resolution**: Δx > 8ε (cell must span ≥ 8 softening lengths)
    2. **Particle sampling**: N_particles / N_grid³ ≥ ``min_particles_per_cell``

    The binding constraint determines the maximum N_grid. If
    ``prefer_power_of_two`` is True (default), the largest power-of-2
    satisfying both constraints is selected (optimal for FFT).

    Parameters
    ----------
    n_particles : int
        Total number of particles in the simulation.
    box_size : float
        Simulation box side length L (comoving, same units as softening).
    softening : float, optional
        Gravitational softening length ε. If None, only the particle
        sampling constraint is applied.
    min_particles_per_cell : int
        Minimum mean number of particles per grid cell. Klypin et al.
        recommend 10–20; default is 20 (conservative).
    prefer_power_of_two : bool
        If True, round down to the nearest power of 2 for FFT efficiency.

    Returns
    -------
    config : dict
        ``'n_grid'``
            Recommended grid cells per dimension.
        ``'cell_width'``
            Cell size Δx = L / N_grid.
        ``'particles_per_cell'``
            Mean particles per cell = N_particles / N_grid³.
        ``'rho_one'``
            Single-particle density ρ_one = (N_grid / N_p^(1/3))³.
            The PDF is unreliable below ~10–20 × ρ_one.
        ``'rho_reliable_min'``
            Minimum ρ/ρ̄ where the PDF is reliable (= min_particles_per_cell × ρ_one).
        ``'n_grid_max'``
            Maximum N_grid before constraints are violated.
        ``'binding_constraint'``
            Which constraint limits N_grid: ``'particles'`` or ``'softening'``.
        ``'warnings'``
            List of warning strings (empty if configuration looks good).

    References
    ----------
    Klypin A., Prada F., Betancort-Rijo J., Albareti F.D., 2018,
    MNRAS, 481, 4588. Section 2, Appendix A, Eq. (A1).

    Examples
    --------
    >>> config = recommend_cic_config(n_particles=1024**3, box_size=512.0,
    ...                               softening=0.005)
    >>> config['n_grid']
    256
    >>> config['particles_per_cell']
    64.0
    """
    n_p_cbrt = round(n_particles ** (1.0 / 3.0))
    warnings = []

    # Constraint 1: particle sampling
    # N_grid³ ≤ N_particles / min_particles_per_cell
    n_grid_max_particles = int(np.floor(
        (n_particles / min_particles_per_cell) ** (1.0 / 3.0)
    ))

    # Constraint 2: force resolution (if softening given)
    if softening is not None and softening > 0:
        n_grid_max_softening = int(np.floor(box_size / (8.0 * softening)))
    else:
        n_grid_max_softening = n_p_cbrt  # no constraint

    # N_grid should not exceed particle grid size
    n_grid_max = min(n_grid_max_particles, n_grid_max_softening, n_p_cbrt)

    if n_grid_max_particles <= n_grid_max_softening:
        binding = 'particles'
    else:
        binding = 'softening'

    if n_grid_max < 16:
        warnings.append(
            f"Maximum grid size is very small ({n_grid_max}). "
            "Consider using a higher-resolution simulation."
        )

    # Select N_grid
    if prefer_power_of_two:
        # Largest power of 2 ≤ n_grid_max
        if n_grid_max >= 16:
            n_grid = 1 << int(np.floor(np.log2(n_grid_max)))
        else:
            n_grid = max(n_grid_max, 1)
    else:
        n_grid = n_grid_max

    # Derived quantities
    cell_width = box_size / n_grid
    particles_per_cell = n_particles / n_grid ** 3
    rho_one = (n_grid / n_p_cbrt) ** 3
    rho_reliable_min = min_particles_per_cell * rho_one

    # Sanity warnings
    if particles_per_cell < 10:
        warnings.append(
            f"Only {particles_per_cell:.1f} particles per cell — "
            "PDF will have discreteness artifacts at low densities."
        )

    if softening is not None and cell_width < 8 * softening:
        warnings.append(
            f"Cell width ({cell_width:.4f}) < 8ε ({8*softening:.4f}) — "
            "density field is under-resolved relative to force resolution."
        )

    return {
        'n_grid': n_grid,
        'cell_width': cell_width,
        'particles_per_cell': particles_per_cell,
        'rho_one': rho_one,
        'rho_reliable_min': rho_reliable_min,
        'n_grid_max': n_grid_max,
        'binding_constraint': binding,
        'warnings': warnings,
    }
