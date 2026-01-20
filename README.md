# Tanager ISOFIT Pipeline

Convert Planet Tanager hyperspectral HDF5 data (TOA radiance) to surface reflectance using ISOFIT atmospheric correction.

## Features

- **HDF5 to ENVI Conversion**: Converts Tanager basic radiance HDF5 files to ISOFIT-compatible ENVI format
- **Pre-computed Geometry**: Uses sun/sensor angles directly from HDF5 metadata when available
- **sRTMnet Integration**: Uses neural network emulator (no MODTRAN license required)
- **ISOFIT Integration**: Wrapper for running ISOFIT atmospheric correction with sensible defaults
- **EMIT Validation**: Tools for comparing results against NASA EMIT L2A surface reflectance
- **CLI Interface**: Full command-line interface for all operations

## Installation

### Basic Installation

```bash
# From PyPI (when available)
pip install tanager-isofit

# Or from source
git clone https://github.com/bradleylab/tanager-isofit.git
cd tanager-isofit
pip install -e .
```

### With ISOFIT (for atmospheric correction)

```bash
pip install -e ".[isofit]"
```

### Full Development Installation

```bash
pip install -e ".[all]"
```

## Prerequisites for Full Pipeline

To run the complete atmospheric correction pipeline (not just conversion), you need:

### 1. ISOFIT Core

```bash
pip install isofit>=3.4.0
```

### 2. 6S Radiative Transfer Code

The 6S code is required for sRTMnet to function. Requires a Fortran compiler.

```bash
# Install gfortran (if not already installed)
# Ubuntu/Debian:
sudo apt-get install gfortran

# macOS:
brew install gcc

# Then download 6S via ISOFIT
isofit download sixs
```

### 3. sRTMnet Neural Network Emulator

```bash
isofit download srtmnet
```

This downloads the emulator to `~/.isofit/srtmnet/sRTMnet_v120.h5`

### 4. Surface Model

ISOFIT requires a surface reflectance prior model. You can either:

**Option A: Use an existing model** (if wavelengths match):
```bash
# Download ISOFIT examples which include surface models
isofit download examples
# Models in: ~/.isofit/examples/*/surface/surface.mat
```

**Option B: Build a model for Tanager wavelengths** (recommended):
```bash
# First run conversion to get wavelength file
tanager-isofit convert input.h5 output/

# Build surface model
isofit surface_model surface_config.json \
  --wavelength_path output/wavelengths.txt \
  --output_path ~/.isofit/tanager_surface.mat
```

See the `tanager_surface_config.json` example in this repository.

### 5. ISOFIT Data Files

```bash
isofit download data
```

### Verify Installation

```bash
# Check all components
ls ~/.isofit/srtmnet/sRTMnet_v120.h5   # sRTMnet emulator
ls ~/.isofit/sixs/sixs                  # 6S executable
ls ~/.isofit/data/                      # Aerosol models, noise files
```

## Quick Start

### Convert Only (No ISOFIT)

Convert Tanager HDF5 to ENVI format without atmospheric correction:

```bash
tanager-isofit convert input.h5 output/
```

Output files:
- `output/radiance` - TOA radiance ENVI file
- `output/loc` - Location (lon, lat, elevation)
- `output/obs` - Observation geometry
- `output/wavelengths.txt` - Wavelength table

### Full Pipeline (Convert + ISOFIT)

Run complete atmospheric correction:

```bash
tanager-isofit process input.h5 output/ \
  --surface-path ~/.isofit/tanager_surface.mat \
  --n-cores 4
```

Output reflectance will be in `output/isofit_working/output/`

### Process a Subset

For testing or memory-constrained systems:

```bash
# Process 100x100 pixel subset
tanager-isofit process input.h5 output/ \
  --subset "0,100,0,100" \
  --surface-path ~/.isofit/tanager_surface.mat \
  --n-cores 2
```

### Skip Empirical Line

For faster processing (less accurate for inhomogeneous scenes):

```bash
tanager-isofit process input.h5 output/ \
  --no-empirical-line \
  --surface-path ~/.isofit/tanager_surface.mat
```

## CLI Reference

### `tanager-isofit inspect`

Examine HDF5 file structure and validate Tanager format.

```bash
tanager-isofit inspect input.h5
```

### `tanager-isofit convert`

Convert HDF5 to ENVI format without ISOFIT processing.

```bash
tanager-isofit convert INPUT_H5 OUTPUT_DIR [OPTIONS]

Options:
  -s, --subset TEXT       Subset as 'row_start,row_end,col_start,col_end'
  --fast-solar/--no-fast-solar  Use fast grid-interpolated solar geometry (default: on)
```

### `tanager-isofit process`

Run full pipeline: conversion + ISOFIT atmospheric correction.

```bash
tanager-isofit process INPUT_H5 OUTPUT_DIR [OPTIONS]

Options:
  -n, --n-cores INTEGER   Number of CPU cores for ISOFIT (default: 4)
  --empirical-line/--no-empirical-line  Use empirical line correction (default: on)
  -s, --subset TEXT       Subset as 'row_start,row_end,col_start,col_end'
  --skip-isofit           Only convert, skip ISOFIT processing
  --sensor TEXT           ISOFIT sensor configuration name (default: tanager)
  --surface-path PATH     Path to surface model file (.mat) - REQUIRED
  --emulator-base PATH    Path to sRTMnet .h5 file ('auto' uses default, 'none' for MODTRAN)
```

### `tanager-isofit check`

Validate ENVI file and print statistics.

```bash
tanager-isofit check output/radiance
```

### `tanager-isofit find-emit`

Search for coincident NASA EMIT data.

```bash
tanager-isofit find-emit input.h5 --time-window 24
```

### `tanager-isofit validate`

Compare Tanager reflectance against EMIT L2A.

```bash
tanager-isofit validate tanager_refl emit_l2a.nc --output report.html
```

## How It Works

The pipeline follows these steps:

### 1. HDF5 to ENVI Conversion

```
Tanager HDF5 → Radiance + Location + Observation (ENVI format)
```

- Extracts TOA radiance from `/HDFEOS/SWATHS/HYP/Data Fields/toa_radiance`
- Extracts geolocation (lat/lon) from `/HDFEOS/SWATHS/HYP/Geolocation Fields/`
- Computes observation geometry (solar/sensor angles, path length)
- Writes ISOFIT-compatible ENVI files with proper headers

### 2. ISOFIT Atmospheric Correction

```
Radiance + Atmosphere Model → Surface Reflectance
```

- Builds look-up tables (LUTs) using 6S radiative transfer
- Uses sRTMnet neural network to emulate atmospheric properties
- Performs optimal estimation inversion per pixel
- Retrieves surface reflectance and atmospheric state

### 3. Output Products

| File | Description |
|------|-------------|
| `*_rfl` | Surface reflectance (0-1 scale) |
| `*_state` | Retrieved atmospheric state (H2O, AOD) |
| `*_uncert` | Uncertainty estimates |

## Python API

```python
from tanager_isofit import convert_tanager_to_envi, run_isofit_pipeline

# Convert only
result = convert_tanager_to_envi(
    "input.h5",
    "output/",
    subset=(0, 100, 0, 100)  # Optional subset
)

# Full pipeline
result = run_isofit_pipeline(
    "input.h5",
    "output/",
    n_cores=4,
    empirical_line=True,
    surface_path="/path/to/surface.mat",
    emulator_base="auto"  # Uses default sRTMnet location
)

# Access outputs
print(result['isofit_outputs']['rfl'])  # Reflectance path
```

## Output Files

### Conversion Output

| File | Bands | Description |
|------|-------|-------------|
| `radiance` | 426 | TOA radiance (uW/cm^2/sr/nm) |
| `loc` | 3 | Longitude, Latitude, Elevation (deg, deg, m) |
| `obs` | 10 | Observation geometry |
| `wavelengths.txt` | - | Wavelength table (index, center, FWHM) |

### Observation File Bands

| Band | Name | Units |
|------|------|-------|
| 0 | path_length | meters |
| 1 | to_sensor_azimuth | degrees |
| 2 | to_sensor_zenith | degrees |
| 3 | to_sun_azimuth | degrees |
| 4 | to_sun_zenith | degrees |
| 5 | phase_angle | degrees |
| 6 | slope | degrees |
| 7 | aspect | degrees |
| 8 | cosine_i | - |
| 9 | utc_time | decimal hours |

## Tanager HDF5 Structure

The package expects Tanager basic radiance HDF5 files:

```
/HDFEOS/SWATHS/HYP/
├── Data Fields/
│   ├── toa_radiance              (426, lines, samples)
│   ├── sensor_to_ground_path_length
│   ├── sensor_zenith
│   ├── sensor_azimuth
│   ├── sun_zenith
│   └── sun_azimuth
└── Geolocation Fields/
    ├── Latitude
    ├── Longitude
    └── Time
```

Wavelengths and FWHM are read from `toa_radiance` dataset attributes.

## Known Limitations

- **Flat terrain assumption**: slope=0, aspect=0 (suitable for water/coastal scenes)
- **Surface model wavelengths**: Must match Tanager bands (426 channels)
- **Memory usage**: Large scenes require subsetting or increased RAM
- **Processing time**: Full scenes can take hours depending on CPU cores

See [ISSUES.md](tanager-isofit-ISSUES.md) for detailed troubleshooting and known issues.

## Test Data

Sample Tanager data from Planet's open data program:

- **Abu Dhabi Coastal**: `20250511_074311_00_4001_basic_radiance.h5` (558 MB)
- URL: https://storage.googleapis.com/open-cogs/planet-stac/release1-basic-radiance/20250511_074311_00_4001_basic_radiance.h5

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_convert.py -v
```

## Package Structure

```
tanager_isofit/
├── __init__.py          # Package init and exports
├── config.py            # Constants, HDF5 paths, sensor specs
├── utils.py             # ENVI I/O helpers
├── convert.py           # HDF5 → ENVI conversion
├── geometry.py          # Solar/sensor geometry calculations
├── dem.py               # Elevation utilities
├── isofit_runner.py     # ISOFIT wrapper
├── validate.py          # EMIT comparison & metrics
└── cli.py               # Command-line interface
```

## License

MIT License

## Acknowledgments

- [ISOFIT](https://github.com/isofit/isofit) - Imaging Spectrometer Optimal FITting
- [Planet](https://www.planet.com/) - Tanager hyperspectral satellite
- [NASA EMIT](https://earth.jpl.nasa.gov/emit/) - Earth Surface Mineral Dust Source Investigation
- [6S](http://6s.ltdri.org/) - Second Simulation of the Satellite Signal in the Solar Spectrum
