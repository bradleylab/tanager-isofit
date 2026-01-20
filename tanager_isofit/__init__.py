"""
Tanager ISOFIT Pipeline - Convert Planet Tanager HDF5 data to surface reflectance.

This package provides tools for:
- Converting Tanager HDF5 radiance data to ENVI format
- Running ISOFIT atmospheric correction
- Validating results against NASA EMIT data
"""

__version__ = "0.1.0"

from tanager_isofit.convert import convert_tanager_to_envi
from tanager_isofit.geometry import calculate_solar_geometry, calculate_sensor_geometry
from tanager_isofit.isofit_runner import run_isofit_pipeline

__all__ = [
    "convert_tanager_to_envi",
    "calculate_solar_geometry",
    "calculate_sensor_geometry",
    "run_isofit_pipeline",
]
