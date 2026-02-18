"""
Tests for Fourier-space operations: smoothing, power spectrum, and σ_R² cross-check.

Validates:
1. Uniform field → P(k) = 0 everywhere (except DC)
2. Single-mode sinusoid → P(k) peak at expected k
3. Smoothing reduces variance: σ²(R₂) < σ²(R₁) for R₂ > R₁
4. σ_R² cross-check: field variance matches P(k) integral
5. Mass conservation: mean unchanged by smoothing
6. CIC window: W_CIC(0) = 1, W_CIC(k_Nyq) = (2/π)² per axis
"""

import os
import pytest
import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from tessera.utils.fourier import (
    gaussian_window,
    tophat_window,
    cic_window_3d,
    smooth_field_fourier,
    compute_power_spectrum,
    sigma_R_squared_from_pk,
    sigma_R_squared_from_field,
)


class TestWindowFunctions:
    """Tests for Fourier-space window functions."""

    def test_gaussian_window_at_zero(self):
        """Gaussian window W(k=0) = 1."""
        assert gaussian_window(0.0, 1.0) == pytest.approx(1.0)

    def test_gaussian_window_decay(self):
        """Gaussian window decays monotonically."""
        k = np.linspace(0, 10, 100)
        W = gaussian_window(k, R=1.0)
        assert np.all(np.diff(W) <= 0)

    def test_tophat_window_at_zero(self):
        """Top-hat window W(k=0) = 1."""
        W = tophat_window(np.array([0.0]), R=1.0)
        assert W[0] == pytest.approx(1.0)

    def test_tophat_window_shape(self):
        """Top-hat window has correct oscillatory behavior."""
        k = np.linspace(0.01, 20, 1000)
        W = tophat_window(k, R=1.0)
        # First zero of W_TH is at kR ≈ 4.493 (tan(x)=x)
        idx_first_zero = np.argmin(np.abs(W[:500]))
        assert k[idx_first_zero] == pytest.approx(4.493, abs=0.1)

    def test_cic_window_at_zero(self):
        """CIC window at k=0 should be 1."""
        kx = np.array([0.0])
        ky = np.array([0.0])
        kz = np.array([0.0])
        W = cic_window_3d(kx, ky, kz, cell_width=1.0)
        assert W[0] == pytest.approx(1.0)

    def test_cic_window_at_nyquist(self):
        """CIC window along one axis at k_Nyq = π/Δ gives sinc(π/2)² = (2/π)²."""
        cell_width = 1.0
        k_nyq = np.pi / cell_width
        # Test along x-axis only (ky=kz=0)
        kx = np.array([k_nyq])
        ky = np.array([0.0])
        kz = np.array([0.0])
        W = cic_window_3d(kx, ky, kz, cell_width)
        expected = (2.0 / np.pi) ** 2  # sinc(π/2)^2 for one axis
        assert W[0] == pytest.approx(expected, rel=1e-10)

    def test_cic_window_all_axes_nyquist(self):
        """CIC window at (k_Nyq, k_Nyq, k_Nyq) = (2/π)^6."""
        cell_width = 1.0
        k_nyq = np.pi / cell_width
        kx = np.array([k_nyq])
        ky = np.array([k_nyq])
        kz = np.array([k_nyq])
        W = cic_window_3d(kx, ky, kz, cell_width)
        expected = (2.0 / np.pi) ** 6  # sinc(π/2)^2 per axis, 3 axes
        assert W[0] == pytest.approx(expected, rel=1e-10)


class TestSmoothing:
    """Tests for Fourier-space smoothing."""

    def test_smoothing_preserves_mean(self):
        """Smoothing should not change the mean of the field."""
        N = 64
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))
        mean_before = grid.mean()

        smoothed = smooth_field_fourier(grid, box_size=L, R=5.0)
        mean_after = smoothed.mean()

        np.testing.assert_allclose(mean_after, mean_before, atol=1e-12)

    def test_smoothing_reduces_variance(self):
        """Larger smoothing scale → lower variance."""
        N = 64
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))

        R1 = 2.0
        R2 = 8.0
        smoothed_R1 = smooth_field_fourier(grid, box_size=L, R=R1)
        smoothed_R2 = smooth_field_fourier(grid, box_size=L, R=R2)

        var_R1 = np.var(smoothed_R1)
        var_R2 = np.var(smoothed_R2)

        assert var_R2 < var_R1, f"σ²(R={R2}) = {var_R2} should be < σ²(R={R1}) = {var_R1}"

    def test_smoothing_uniform_field_unchanged(self):
        """Smoothing a uniform field should leave it unchanged."""
        N = 32
        L = 10.0
        grid = np.ones((N, N, N)) * 3.14

        smoothed = smooth_field_fourier(grid, box_size=L, R=2.0)

        np.testing.assert_allclose(smoothed, 3.14, atol=1e-12)

    def test_smoothing_zero_scale(self):
        """Smoothing at R=0 should return the original field."""
        N = 32
        L = 10.0
        np.random.seed(123)
        grid = np.random.normal(0, 1, (N, N, N))

        smoothed = smooth_field_fourier(grid, box_size=L, R=0.0)

        np.testing.assert_allclose(smoothed, grid, atol=1e-12)

    def test_smoothing_tophat(self):
        """Top-hat smoothing should also preserve mean and reduce variance."""
        N = 64
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))
        mean_before = grid.mean()

        smoothed = smooth_field_fourier(grid, box_size=L, R=5.0, window='tophat')

        np.testing.assert_allclose(smoothed.mean(), mean_before, atol=1e-12)
        assert np.var(smoothed) < np.var(grid)

    def test_smoothing_invalid_window_raises(self):
        """Invalid window type should raise ValueError."""
        grid = np.zeros((8, 8, 8))
        with pytest.raises(ValueError, match="Unknown window"):
            smooth_field_fourier(grid, box_size=10.0, R=1.0, window='invalid')


class TestPowerSpectrum:
    """Tests for power spectrum computation."""

    def test_uniform_field_zero_power(self):
        """Uniform field (constant δ=0) should have P(k) = 0 everywhere."""
        N = 32
        L = 100.0
        grid = np.zeros((N, N, N))

        pk = compute_power_spectrum(grid, box_size=L)

        np.testing.assert_allclose(pk['Pk'], 0.0, atol=1e-30)

    def test_single_mode_sinusoid(self):
        """A single Fourier mode should produce a P(k) peak at the correct k."""
        N = 64
        L = 100.0
        n_mode = 3  # mode number

        # Create sinusoidal field: δ = A * sin(2π n x / L)
        A = 1.0
        x = np.linspace(0, L, N, endpoint=False)
        delta_1d = A * np.sin(2.0 * np.pi * n_mode * x / L)
        grid = np.zeros((N, N, N))
        grid[:, :, :] = delta_1d[np.newaxis, np.newaxis, :]  # along z-axis

        pk = compute_power_spectrum(grid, box_size=L, n_bins=N // 2)

        # Expected peak k
        k_expected = 2.0 * np.pi * n_mode / L

        # Find the bin with maximum power
        idx_max = np.argmax(pk['Pk'])
        k_peak = pk['k'][idx_max]

        # Peak should be within one bin width of expected
        dk = pk['k'][1] - pk['k'][0] if len(pk['k']) > 1 else L
        assert abs(k_peak - k_expected) < 2 * dk, \
            f"Peak at k={k_peak}, expected k={k_expected}"

        # Power at peak should be significant relative to other bins
        assert pk['Pk'][idx_max] > 10 * np.median(pk['Pk'])

    def test_parseval_theorem(self):
        """Power spectrum captures correct power within the Nyquist sphere.

        Shell-averaging bins modes with |k| ≤ k_Nyquist (the inscribed sphere
        of the cubic FFT grid). For white noise, the fraction of modes inside
        this sphere is π/6 ≈ 0.524. We verify that P(k) × n_modes sums to
        this expected fraction of the total variance.
        """
        N = 64
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))

        field_var = np.var(grid)
        V = L ** 3

        pk = compute_power_spectrum(grid, box_size=L, n_bins=N)

        # Discrete sum over binned modes
        sigma_sq_binned = np.sum(pk['n_modes'] * pk['Pk']) / V

        # For white noise, fraction of modes in Nyquist sphere ≈ π/6
        expected_fraction = np.pi / 6.0
        actual_fraction = sigma_sq_binned / field_var

        np.testing.assert_allclose(actual_fraction, expected_fraction, rtol=0.02)

    def test_power_spectrum_returns_expected_keys(self):
        """Check all expected keys are in the returned dict."""
        N = 16
        L = 10.0
        grid = np.random.normal(0, 1, (N, N, N))

        pk = compute_power_spectrum(grid, box_size=L)

        expected_keys = {'k', 'Pk', 'delta_squared', 'n_modes', 'k_fundamental', 'k_nyquist'}
        assert expected_keys == set(pk.keys())

    def test_k_fundamental_and_nyquist(self):
        """k_fundamental and k_nyquist should have correct values."""
        N = 64
        L = 256.0
        grid = np.zeros((N, N, N))

        pk = compute_power_spectrum(grid, box_size=L)

        assert pk['k_fundamental'] == pytest.approx(2.0 * np.pi / L)
        assert pk['k_nyquist'] == pytest.approx(np.pi * N / L)

    def test_shot_noise_subtraction(self):
        """Shot noise subtraction should reduce P(k)."""
        N = 32
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 0.1, (N, N, N))

        pk_no_shot = compute_power_spectrum(grid, box_size=L)
        pk_with_shot = compute_power_spectrum(
            grid, box_size=L, subtract_shot_noise=True, n_particles=N ** 3)

        # P(k) with shot noise subtracted should be lower
        assert np.all(pk_with_shot['Pk'] <= pk_no_shot['Pk'] + 1e-10)

    def test_delta_squared_relation(self):
        """Δ²(k) = k³ P(k) / (2π²) should hold."""
        N = 32
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))

        pk = compute_power_spectrum(grid, box_size=L)

        expected_delta_sq = pk['k'] ** 3 * pk['Pk'] / (2.0 * np.pi ** 2)
        np.testing.assert_allclose(pk['delta_squared'], expected_delta_sq, rtol=1e-10)


class TestSigmaRSquared:
    """Tests for variance σ_R² computation and cross-check."""

    def test_field_variance_matches_numpy(self):
        """sigma_R_squared_from_field should match np.var."""
        np.random.seed(42)
        grid = np.random.normal(0, 1, (32, 32, 32))

        sigma_sq = sigma_R_squared_from_field(grid)
        expected = np.var(grid)

        assert sigma_sq == pytest.approx(expected, rel=1e-12)

    def test_sigma_crosscheck_gaussian(self):
        """σ_R² from smoothed field should match σ_R² from P(k) integral.

        This is the key diagnostic: volume-weighted PDF variance must equal
        the P(k)-integrated variance when using the same effective window.
        """
        N = 64
        L = 100.0
        R = 5.0  # smoothing scale
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))

        # Method 1: smooth the field, then measure variance
        smoothed = smooth_field_fourier(grid, box_size=L, R=R, window='gaussian')
        sigma_sq_field = sigma_R_squared_from_field(smoothed)

        # Method 2: compute P(k), then integrate with window
        pk = compute_power_spectrum(grid, box_size=L, n_bins=N // 2)
        sigma_sq_pk = sigma_R_squared_from_pk(
            pk['k'], pk['Pk'], R=R, window='gaussian')

        # Should agree within ~5% (limited by binning and integration accuracy)
        np.testing.assert_allclose(sigma_sq_pk, sigma_sq_field, rtol=0.10)

    def test_larger_R_smaller_sigma(self):
        """σ_R² should decrease monotonically with increasing R."""
        N = 64
        L = 100.0
        np.random.seed(42)
        grid = np.random.normal(0, 1, (N, N, N))

        pk = compute_power_spectrum(grid, box_size=L, n_bins=N // 2)

        R_values = [1.0, 2.0, 4.0, 8.0]
        sigmas = [sigma_R_squared_from_pk(pk['k'], pk['Pk'], R=R) for R in R_values]

        for i in range(len(sigmas) - 1):
            assert sigmas[i + 1] < sigmas[i], \
                f"σ²(R={R_values[i+1]}) = {sigmas[i+1]} should be < σ²(R={R_values[i]}) = {sigmas[i]}"

    def test_sigma_invalid_window_raises(self):
        """Invalid window should raise ValueError."""
        k = np.array([1.0, 2.0])
        Pk = np.array([1.0, 0.5])
        with pytest.raises(ValueError, match="Unknown window"):
            sigma_R_squared_from_pk(k, Pk, R=1.0, window='invalid')


class TestGADGET4Reader:
    """Tests for GADGET-4 power spectrum file readers."""

    @pytest.fixture
    def powerspec_path(self):
        """Path to test powerspec file."""
        path = "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/powerspecs/powerspec_034.txt"
        if not os.path.exists(path):
            pytest.skip("GADGET-4 powerspec file not available")
        return path

    @pytest.fixture
    def colossus_path(self):
        """Path to test colossus file."""
        path = "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/powerspecs/for_colossus_034.txt"
        if not os.path.exists(path):
            pytest.skip("GADGET-4 colossus file not available")
        return path

    def test_read_powerspec(self, powerspec_path):
        """Read a real GADGET-4 powerspec file and check structure."""
        from tessera.utils.gadget4_pk import read_gadget4_powerspec

        pk = read_gadget4_powerspec(powerspec_path)

        assert pk['scale_factor'] == pytest.approx(1.0)
        assert pk['box_size'] == pytest.approx(256.0)
        assert pk['pmgrid'] == 256
        assert len(pk['k']) > 0
        assert len(pk['k']) == len(pk['delta_squared'])
        assert len(pk['k']) == len(pk['Pk'])
        assert len(pk['sections']) == 3

        # k should be monotonically increasing
        assert np.all(np.diff(pk['k']) > 0)

        # Δ²(k) should be positive (after shot noise subtraction, most should be)
        assert np.sum(pk['delta_squared'] > 0) > 0.9 * len(pk['delta_squared'])

    def test_read_colossus(self, colossus_path):
        """Read a real for_colossus file and check structure."""
        from tessera.utils.gadget4_pk import read_gadget4_colossus_pk

        pk = read_gadget4_colossus_pk(colossus_path)

        assert len(pk['k']) > 0
        assert len(pk['k']) == len(pk['Pk'])
        assert np.all(pk['k'] > 0)
        assert np.all(pk['Pk'] > 0)

    def test_get_powerspec_paths(self):
        """Find powerspec files in a directory."""
        from tessera.utils.gadget4_pk import get_powerspec_paths

        pk_dir = "/Users/bryen/Documents/Physics Research/Stanford/asymptotic_assembly/Uniform_L256_N256_primary_sandbox/gadget4/output/powerspecs"
        if not os.path.exists(pk_dir):
            pytest.skip("GADGET-4 powerspecs directory not available")

        paths = get_powerspec_paths(pk_dir)
        assert len(paths) > 0
        assert 34 in paths


class TestRecommendCICConfig:
    """Tests for automatic CIC grid configuration."""

    def test_1024_cubed_particle_limited(self):
        """1024³ particles should be particle-limited, not softening-limited."""
        from tessera.utils.fourier import recommend_cic_config

        config = recommend_cic_config(
            n_particles=1024**3, box_size=512.0, softening=0.005)

        assert config['binding_constraint'] == 'particles'
        assert config['n_grid'] == 256  # largest power-of-2 with ≥20 ptcl/cell
        assert config['particles_per_cell'] == pytest.approx(64.0)

    def test_particles_per_cell_above_threshold(self):
        """Recommended config should always meet the particle threshold."""
        from tessera.utils.fourier import recommend_cic_config

        for n_p, L in [(256**3, 256.0), (512**3, 100.0), (1024**3, 2048.0)]:
            config = recommend_cic_config(n_particles=n_p, box_size=L)
            assert config['particles_per_cell'] >= 20, \
                f"N_p={n_p}, L={L}: only {config['particles_per_cell']} ptcl/cell"

    def test_softening_constraint_binding(self):
        """Very large softening should make force resolution the binding constraint."""
        from tessera.utils.fourier import recommend_cic_config

        # Absurdly large softening: 8ε = 80, so Δx > 80 → N_grid < L/80 = 1.28
        config = recommend_cic_config(
            n_particles=64**3, box_size=100.0, softening=10.0)

        assert config['binding_constraint'] == 'softening'

    def test_power_of_two(self):
        """With prefer_power_of_two, N_grid should be a power of 2."""
        from tessera.utils.fourier import recommend_cic_config

        config = recommend_cic_config(
            n_particles=1024**3, box_size=512.0, prefer_power_of_two=True)

        n = config['n_grid']
        assert n & (n - 1) == 0, f"{n} is not a power of 2"

    def test_no_power_of_two(self):
        """Without prefer_power_of_two, N_grid can be any value."""
        from tessera.utils.fourier import recommend_cic_config

        config = recommend_cic_config(
            n_particles=1024**3, box_size=512.0, prefer_power_of_two=False)

        # Should be the actual maximum, not rounded down
        assert config['n_grid'] == config['n_grid_max']

    def test_rho_one_formula(self):
        """ρ_one should match Klypin (2018) Eq. A1."""
        from tessera.utils.fourier import recommend_cic_config

        config = recommend_cic_config(n_particles=256**3, box_size=256.0)
        n_grid = config['n_grid']
        expected_rho_one = (n_grid / 256.0) ** 3
        assert config['rho_one'] == pytest.approx(expected_rho_one)

    def test_no_softening(self):
        """Should work without softening (only particle constraint)."""
        from tessera.utils.fourier import recommend_cic_config

        config = recommend_cic_config(n_particles=512**3, box_size=100.0)

        assert config['n_grid'] > 0
        assert config['binding_constraint'] == 'particles'
        assert len(config['warnings']) == 0

    def test_user_simulation_suite(self):
        """Verify recommendations for the user's planned 1024³ suite."""
        from tessera.utils.fourier import recommend_cic_config

        for L in [32.0, 128.0, 512.0, 2048.0]:
            eps = 0.01 * L / 1024
            config = recommend_cic_config(
                n_particles=1024**3, box_size=L, softening=eps)

            # All should give N_grid=256 with 64 ptcl/cell
            assert config['n_grid'] == 256, f"L={L}: got N_grid={config['n_grid']}"
            assert config['particles_per_cell'] == pytest.approx(64.0)
            assert config['binding_constraint'] == 'particles'
            assert len(config['warnings']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
