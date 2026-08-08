"""Sampling-aware gradient attributes for synthetic 3D seismic volumes."""

from .attributes import gradient_attributes
from .forward import (
    acoustic_impedance,
    depth_mask_to_regular_time,
    depth_property_to_regular_time,
    poststack_synthetic,
)
from .model import Grid3D, LayeredFaultModel, build_layered_fault_model

__all__ = [
    "Grid3D",
    "LayeredFaultModel",
    "acoustic_impedance",
    "build_layered_fault_model",
    "depth_mask_to_regular_time",
    "depth_property_to_regular_time",
    "gradient_attributes",
    "poststack_synthetic",
]
