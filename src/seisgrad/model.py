"""Continuous geological model sampled on a regular depth grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid3D:
    """Regular Cartesian grid with coordinates in metres."""

    nx: int = 96
    ny: int = 96
    nz: int = 192
    dx: float = 12.5
    dy: float = 12.5
    dz: float = 5.0

    @classmethod
    def from_extent(
        cls,
        *,
        extent_x_m: float,
        extent_y_m: float,
        extent_z_m: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> "Grid3D":
        """Construct a grid whose end points coincide with fixed extents."""

        values = (extent_x_m, extent_y_m, extent_z_m, dx, dy, dz)
        if min(values) <= 0.0:
            raise ValueError("extents and cell sizes must be positive")
        nx = int(round(extent_x_m / dx)) + 1
        ny = int(round(extent_y_m / dy)) + 1
        nz = int(round(extent_z_m / dz)) + 1
        if not np.isclose((nx - 1) * dx, extent_x_m):
            raise ValueError("extent_x_m must be divisible by dx")
        if not np.isclose((ny - 1) * dy, extent_y_m):
            raise ValueError("extent_y_m must be divisible by dy")
        if not np.isclose((nz - 1) * dz, extent_z_m):
            raise ValueError("extent_z_m must be divisible by dz")
        return cls(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.nx, dtype=float) * self.dx

    @property
    def y(self) -> np.ndarray:
        return np.arange(self.ny, dtype=float) * self.dy

    @property
    def z(self) -> np.ndarray:
        return np.arange(self.nz, dtype=float) * self.dz


@dataclass(frozen=True)
class LayeredFaultModel:
    """Physical properties and exact geometry of the synthetic model."""

    grid: Grid3D
    vp: np.ndarray
    density: np.ndarray
    layer_index: np.ndarray
    fault_mask: np.ndarray
    fault_signed_distance: np.ndarray


@dataclass(frozen=True)
class IncisedChannelModel:
    """Layered background containing one continuously defined incised channel."""

    grid: Grid3D
    vp: np.ndarray
    density: np.ndarray
    layer_index: np.ndarray
    channel_mask: np.ndarray
    channel_footprint: np.ndarray
    channel_signed_margin_distance_m: np.ndarray
    channel_top_depth_m: float


def build_layered_fault_model(
    grid: Grid3D,
    *,
    strike_deg: float = 35.0,
    dip_deg: float = 60.0,
    throw_m: float = 35.0,
    fold_amplitude_m: float = 18.0,
) -> LayeredFaultModel:
    """Build gently folded layers displaced by one oblique normal fault.

    The model is defined by continuous equations before it is sampled. The
    hanging-wall displacement is vertical, which keeps the diagnostic model
    deliberately simple while allowing arbitrary strike and dip.
    """

    x, y, z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    x0 = 0.52 * grid.x[-1]
    y0 = 0.48 * grid.y[-1]
    z0 = 0.50 * grid.z[-1]

    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    dip_direction = strike + np.pi / 2.0

    # Unit normal to a plane dipping along ``dip_direction``.
    nx = -np.sin(dip) * np.cos(dip_direction)
    ny = -np.sin(dip) * np.sin(dip_direction)
    nz = np.cos(dip)
    signed_distance = nx * (x - x0) + ny * (y - y0) + nz * (z - z0)

    fold = fold_amplitude_m * (
        0.65 * np.sin(2.0 * np.pi * x / max(grid.x[-1], grid.dx))
        + 0.35 * np.sin(2.0 * np.pi * y / max(grid.y[-1], grid.dy) + 0.6)
    )
    regional_dip = 0.025 * (x - x0) - 0.015 * (y - y0)
    hanging_wall_shift = throw_m * (signed_distance >= 0.0)
    restored_depth = z - fold - regional_dip - hanging_wall_shift

    boundaries_m = grid.z[-1] * np.array([0.12, 0.25, 0.39, 0.54, 0.70, 0.86])
    layer_index = np.digitize(restored_depth, boundaries_m)

    vp_values = np.array([2050.0, 2350.0, 2650.0, 2950.0, 3250.0, 3550.0, 3900.0])
    rho_values = np.array([2020.0, 2110.0, 2200.0, 2290.0, 2380.0, 2460.0, 2550.0])
    vp = vp_values[layer_index]
    density = rho_values[layer_index]

    voxel_diagonal = np.sqrt(grid.dx**2 + grid.dy**2 + grid.dz**2)
    fault_mask = np.abs(signed_distance) <= 0.55 * voxel_diagonal

    return LayeredFaultModel(
        grid=grid,
        vp=vp,
        density=density,
        layer_index=layer_index,
        fault_mask=fault_mask,
        fault_signed_distance=signed_distance,
    )


def build_incised_channel_model(
    grid: Grid3D,
    *,
    orientation_deg: float = 30.0,
    impedance_contrast_fraction: float = 0.08,
    channel_top_depth_m: float = 350.0,
    maximum_incision_m: float = 42.0,
    mean_half_width_m: float = 55.0,
    meander_amplitude_m: float = 115.0,
    meander_wavelength_m: float = 520.0,
) -> IncisedChannelModel:
    """Build a band-limited-test model with a meandering incised channel.

    The channel is defined analytically before sampling.  Its cross-section is
    approximately semi-elliptical, so the fill pinches out at the margins
    rather than forming an unrealistic vertically extruded ribbon.  Fill
    impedance contrast varies smoothly along the channel to include both
    strong and weak seismic segments without introducing discontinuities in
    the input equations.
    """

    if not 0.0 < impedance_contrast_fraction < 0.5:
        raise ValueError("impedance_contrast_fraction must lie between 0 and 0.5")
    if not 0.0 < channel_top_depth_m < grid.z[-1]:
        raise ValueError("channel_top_depth_m must lie inside the model")
    if min(
        maximum_incision_m,
        mean_half_width_m,
        meander_wavelength_m,
    ) <= 0.0:
        raise ValueError("channel dimensions must be positive")

    x, y, z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    x2, y2 = np.meshgrid(grid.x, grid.y, indexing="ij")
    x_center = 0.5 * grid.x[-1]
    y_center = 0.5 * grid.y[-1]
    angle = np.deg2rad(orientation_deg)

    # Coordinates parallel and normal to the mean channel direction.
    along = np.cos(angle) * (x2 - x_center) + np.sin(angle) * (y2 - y_center)
    across = -np.sin(angle) * (x2 - x_center) + np.cos(angle) * (y2 - y_center)
    centerline = meander_amplitude_m * (
        np.sin(2.0 * np.pi * along / meander_wavelength_m)
        + 0.18 * np.sin(4.0 * np.pi * along / meander_wavelength_m + 0.65)
    )
    half_width = mean_half_width_m * (
        1.0 + 0.14 * np.sin(2.0 * np.pi * along / 690.0 - 0.35)
    )
    cross_distance = across - centerline
    normalized_distance = np.abs(cross_distance) / half_width
    footprint = normalized_distance <= 1.0
    signed_margin_distance = np.abs(cross_distance) - half_width

    # A semi-elliptical erosional base produces zero thickness at the margins.
    incision = np.zeros_like(normalized_distance)
    incision[footprint] = maximum_incision_m * (
        1.0 - normalized_distance[footprint] ** 2
    ) ** 0.60
    channel_mask = (
        footprint[..., None]
        & (z >= channel_top_depth_m)
        & (z <= channel_top_depth_m + incision[..., None])
    )

    boundaries_m = np.array(
        [90.0, 190.0, channel_top_depth_m, 480.0, 620.0, 740.0]
    )
    if boundaries_m[-1] >= grid.z[-1]:
        boundaries_m = boundaries_m[boundaries_m < grid.z[-1]]
    layer_index = np.digitize(z, boundaries_m)
    vp_values = np.array([2050.0, 2250.0, 2500.0, 2850.0, 3150.0, 3450.0, 3750.0])
    rho_values = np.array([2000.0, 2080.0, 2170.0, 2290.0, 2390.0, 2480.0, 2570.0])
    vp = vp_values[layer_index].astype(float)
    density = rho_values[layer_index].astype(float)

    # The requested contrast is the maximum local impedance reduction.  The
    # smooth modulation creates low-reflectivity channel segments for testing
    # whether normalized attributes genuinely improve margin continuity.
    contrast_modulation = 0.35 + 0.65 * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * along / 720.0 + 0.80)
    )
    local_contrast = impedance_contrast_fraction * contrast_modulation
    property_scale = np.sqrt(1.0 - local_contrast)[..., None]
    vp = np.where(channel_mask, vp * property_scale, vp)
    density = np.where(channel_mask, density * property_scale, density)

    return IncisedChannelModel(
        grid=grid,
        vp=vp,
        density=density,
        layer_index=layer_index,
        channel_mask=channel_mask,
        channel_footprint=footprint,
        channel_signed_margin_distance_m=signed_margin_distance,
        channel_top_depth_m=channel_top_depth_m,
    )
