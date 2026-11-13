import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgrad.attributes import _derivative, gradient_attributes
from seisgrad.forward import acoustic_impedance
from seisgrad.metrics import (
    add_colored_impulsive_noise,
    average_precision,
    best_f1,
    paired_bootstrap_interval,
    rms_normalized_error,
    shape_rmse,
)
from seisgrad.model import Grid3D, build_incised_channel_model


class GradientAttributeTests(unittest.TestCase):
    def test_constant_volume_has_zero_gradient_attributes(self):
        volume = np.ones((8, 9, 10), dtype=float)
        result = gradient_attributes(
            volume,
            spacing=(12.5, 18.75, 0.004),
            reference_velocity_m_s=3000.0,
            sampling_aware=True,
        )
        for values in result.values():
            self.assertTrue(np.allclose(values, 0.0))

    def test_linear_physical_gradient_is_recovered(self):
        dx, dy, dt = 10.0, 20.0, 0.002
        velocity = 2500.0
        x = np.arange(8)[:, None, None] * dx
        y = np.arange(9)[None, :, None] * dy
        t = np.arange(10)[None, None, :] * dt
        volume = 2.0 * x + 3.0 * y + 4.0 * (velocity / 2.0) * t
        result = gradient_attributes(
            volume,
            spacing=(dx, dy, dt),
            reference_velocity_m_s=velocity,
            sampling_aware=True,
        )
        self.assertTrue(np.allclose(result["thdr"], np.hypot(2.0, 3.0)))
        self.assertTrue(np.allclose(result["asag"], np.sqrt(2.0**2 + 3.0**2 + 4.0**2)))

    def test_fourth_order_derivative_is_exact_for_quartic_interior(self):
        coordinate = np.arange(-8.0, 9.0)
        values = coordinate**4
        derivative = _derivative(values, spacing=1.0, axis=0, order=4)
        np.testing.assert_allclose(derivative[2:-2], 4.0 * coordinate[2:-2] ** 3)

    def test_invalid_derivative_order_is_rejected(self):
        with self.assertRaises(ValueError):
            _derivative(np.arange(8.0), spacing=1.0, axis=0, order=6)

    def test_acoustic_impedance(self):
        vp = np.array([2000.0, 2500.0])
        density = np.array([2100.0, 2200.0])
        np.testing.assert_allclose(acoustic_impedance(vp, density), vp * density)

    def test_grid_from_extent(self):
        grid = Grid3D.from_extent(
            extent_x_m=100.0,
            extent_y_m=200.0,
            extent_z_m=50.0,
            dx=10.0,
            dy=20.0,
            dz=5.0,
        )
        self.assertEqual((grid.nx, grid.ny, grid.nz), (11, 11, 11))
        self.assertAlmostEqual(grid.x[-1], 100.0)

    def test_incised_channel_is_continuous_geometry_with_lower_impedance_fill(self):
        grid = Grid3D.from_extent(
            extent_x_m=800.0,
            extent_y_m=800.0,
            extent_z_m=800.0,
            dx=25.0,
            dy=25.0,
            dz=5.0,
        )
        model = build_incised_channel_model(
            grid,
            orientation_deg=30.0,
            impedance_contrast_fraction=0.08,
        )
        self.assertEqual(model.channel_mask.shape, (grid.nx, grid.ny, grid.nz))
        self.assertTrue(np.any(model.channel_mask))
        self.assertTrue(np.any(model.channel_footprint))
        self.assertTrue(np.any(model.channel_signed_margin_distance_m < 0.0))
        self.assertTrue(np.any(model.channel_signed_margin_distance_m > 0.0))

        # Every fill voxel must project inside the analytical footprint, and
        # the fill impedance must be lower than its unmodified host layer.
        self.assertTrue(np.all(model.channel_footprint[model.channel_mask.any(axis=-1)]))
        layer_vp = np.array([2050.0, 2250.0, 2500.0, 2850.0, 3150.0, 3450.0, 3750.0])
        layer_rho = np.array([2000.0, 2080.0, 2170.0, 2290.0, 2390.0, 2480.0, 2570.0])
        host_impedance = layer_vp[model.layer_index] * layer_rho[model.layer_index]
        fill_impedance = model.vp * model.density
        self.assertTrue(np.all(fill_impedance[model.channel_mask] < host_impedance[model.channel_mask]))

    def test_incised_channel_rejects_invalid_contrast(self):
        grid = Grid3D(nx=9, ny=9, nz=161, dx=100.0, dy=100.0, dz=5.0)
        with self.assertRaises(ValueError):
            build_incised_channel_model(grid, impedance_contrast_fraction=0.0)

    def test_classification_metrics(self):
        labels = np.array([1, 0, 1, 0], dtype=bool)
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)
        f1, threshold = best_f1(labels, scores)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(threshold, 0.8)

    def test_rms_normalized_error_ignores_positive_global_scale(self):
        values = np.arange(1.0, 10.0).reshape(3, 3)
        self.assertAlmostEqual(rms_normalized_error(values, 7.5 * values), 0.0)
        self.assertAlmostEqual(
            shape_rmse(values, 7.5 * values),
            rms_normalized_error(values, 7.5 * values),
        )

    def test_revision_noise_and_bootstrap_are_reproducible(self):
        values = np.arange(64.0).reshape(4, 4, 4)
        first = add_colored_impulsive_noise(
            values, 10.0, rng=np.random.default_rng(17)
        )
        second = add_colored_impulsive_noise(
            values, 10.0, rng=np.random.default_rng(17)
        )
        np.testing.assert_array_equal(first, second)
        low, high = paired_bootstrap_interval(
            np.array([0.1, 0.2, 0.3]),
            rng=np.random.default_rng(19),
            resamples=1000,
        )
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
