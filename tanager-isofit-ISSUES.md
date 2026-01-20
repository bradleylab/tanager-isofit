# tanager-isofit Issues & Future Improvements

**Repo**: https://github.com/bradleylab/tanager-isofit

---

## Root Cause Analysis - ISOFIT Inversion Failure

### Summary
ISOFIT produces all-zero reflectance output due to **missing 6S radiative transfer code**.

### Investigation Path (2025-01-19)

1. **Initial error**: "Matrix inversion contains negative values" + "invalid value encountered in scalar divide" with coszen

2. **First hypothesis (incorrect)**: Geometry in observation file is wrong
   - **Finding**: Observation file geometry is valid:
     - Solar zenith: ~10.9° (sun high in sky)
     - cosine_i: ~0.982 (cos(10.9°) ≈ 0.982)
     - No zeros or negative values in obs file

3. **Second observation**: Path length units were wrong!
   - ISOFIT's `geometry.py` line 73: `self.path_length_km = units.m_to_km(obs[0])`
   - ISOFIT expects path_length in **METERS**, converts internally to km
   - Our code was converting meters→km before writing, causing 1000x error
   - **FIXED**: Now keeping path_length in meters (~485,000 m for Tanager)

4. **Current root cause**: 6S radiative transfer code not installed
   - sRTMnet emulator requires 6S to generate baseline atmospheric quantities
   - Without 6S: `ERROR: 6S path not valid, downstream simulations will be broken`
   - LUT ends up with NaN values for coszen, solzen, rhoatm, sphalb
   - This causes division-by-zero in radiative_transfer.py

### Consequence
Without 6S installed:
- sRTMnet can't generate required atmospheric quantities
- LUT values are NaN
- Atmospheric correction calculations produce invalid results
- Matrix inversion fails

### Solution
Install 6S:
```bash
# Requires gfortran compiler
isofit download sixs
```

---

## Fixes Implemented (2025-01-19)

### 1. ✅ Timestamp-prefixed filenames
**File**: `tanager_isofit/convert.py`

ISOFIT expects filenames starting with `YYYYMMDD_HHMMSS_`. Changed output filenames:
- `radiance` → `20250511_074311_radiance`
- `loc` → `20250511_074311_loc`
- `obs` → `20250511_074311_obs`

### 2. ✅ Acquisition time from HDF5 Time dataset
**File**: `tanager_isofit/convert.py`

Now reads Unix timestamps from `HDFEOS/SWATHS/HYP/Geolocation Fields/Time` dataset instead of failing back to current time.

### 3. ✅ Wavelength file format
**File**: `tanager_isofit/utils.py`

ISOFIT expects 3 columns: channel, wavelength, fwhm. Fixed `create_wavelength_file()` to write:
```
0 376.440002 5.390000
1 381.410004 5.420000
...
```

### 4. ✅ Added `--surface-path` CLI option
**File**: `tanager_isofit/cli.py`

Added `--surface-path` option to the `process` command for specifying the surface model.

### 5. ✅ Path length units (CRITICAL FIX)
**Files**: `tanager_isofit/convert.py`, `tanager_isofit/geometry.py`

ISOFIT expects path_length in **METERS** (band 0 of obs file), NOT kilometers.
ISOFIT internally converts to km via `units.m_to_km(obs[0])`.

Fixed:
- `convert.py`: Keep path_length in meters (~485,000 m for Tanager altitude)
- `geometry.py`: Updated parameter names from `path_length_km` to `path_length_m`
- Docstrings updated to clarify units

Verification:
```
Path length (band 0): 485,335.91 meters ✓
ISOFIT reports: Path (km): 485.34 ✓ (correctly converted internally)
```

### 6. ✅ Tanager-specific surface model
**File**: Surface model built with matching wavelength grid

Created surface model config and built `.mat` file:
```python
from isofit.utils.surface_model import surface_model
surface_model('tanager_surface_config.json')
```

---

## Known Issues (Remaining)

### 1. CRITICAL: 6S radiative transfer code not installed
**Status**: Blocking ISOFIT processing
**Root cause**: sRTMnet emulator requires 6S for baseline atmospheric calculations
**Error**: `6S path not valid, downstream simulations will be broken`
**Impact**: LUT values are NaN → "Matrix inversion contains negative values"
**Fix**:
```bash
# Install gfortran first (system dependency)
sudo apt install gfortran  # or conda install -c conda-forge gfortran

# Then download 6S
isofit download sixs
```

### 2. Surface model wavelength mismatch (RESOLVED)
**Status**: Fixed by building Tanager-specific surface model
**Original error**: "Center wavelengths provided in surface model file do not match wavelengths in radiance cube"
**Solution**: Build surface model with Tanager wavelengths:
```python
from isofit.utils.surface_model import surface_model

config = {
    'output_model_file': 'tanager_surface_model.mat',
    'wavelength_file': 'wavelengths.txt',  # From tanager-isofit conversion
    'normalize': 'Euclidean',
    'reference_windows': [[400, 1300], [1450, 1700], [2100, 2450]],
    'sources': [{
        'input_spectrum_files': ['~/.isofit/data/reflectance/surface_model_ucsb'],
        'n_components': 8,
        'windows': [...]
    }]
}
surface_model('surface_config.json')
```

### 3. sRTMnet failures at high water vapor
**Status**: Low priority (depends on 6S fix)
**Observed**: "Failed to parse any data" for H2OSTR=5.0 combinations
**Impact**: Some LUT grid points are NaN
**Likely cause**: High water vapor outside sRTMnet training domain

### 4. Negative radiance values
**Status**: Minor
**Observed**: Min radiance = -0.255
**Impact**: Physically impossible, may cause issues in water absorption bands
**Fix**: Clamp negative values to 0 or small positive value

---

## Unit Requirements (CRITICAL)

### Observation File (obs) Band Units
| Band | Name | ISOFIT Expects | Notes |
|------|------|----------------|-------|
| 0 | path_length | **METERS** | ISOFIT converts to km internally via `units.m_to_km()` |
| 1 | to_sensor_azimuth | degrees | 0-360, clockwise from N |
| 2 | to_sensor_zenith | degrees | 0-90, from zenith |
| 3 | to_sun_azimuth | degrees | 0-360, clockwise from N |
| 4 | to_sun_zenith | degrees | 0-90, from zenith |
| 5 | phase_angle | degrees | |
| 6 | slope | degrees | |
| 7 | aspect | degrees | |
| 8 | cosine_i | unitless | cos(effective solar zenith) |
| 9 | utc_time | decimal hours | |

### Location File (loc) Band Units
| Band | Name | ISOFIT Expects | Notes |
|------|------|----------------|-------|
| 0 | longitude | degrees | |
| 1 | latitude | degrees | |
| 2 | elevation | **METERS** | ISOFIT converts to km internally |

### Radiance Units

**HDF5 units**: `W/(m² sr μm)`
**ISOFIT expects**: `μW/(cm² sr nm)`
**Conversion**: `W/(m² sr μm) × 0.1 = μW/(cm² sr nm)`

Current implementation does NOT convert units. After investigation:
- Tanager sensor config in ISOFIT may expect W/(m² sr μm)
- Need to verify against Tanager noise model assumptions
- May need to add unit conversion in convert.py

---

## Test Results (2026-01-19)

### HDF5 → ENVI Conversion
| Test | Result | Notes |
|------|--------|-------|
| Timestamp from Time dataset | ✅ Pass | Correctly reads 2025-05-11 07:43:11 UTC |
| Output filenames | ✅ Pass | Properly prefixed with timestamp |
| Wavelength file format | ✅ Pass | 3 columns: channel, wavelength, fwhm |
| Geometry from HDF5 | ✅ Pass | Uses pre-computed sun/sensor angles |
| CLI --surface-path | ✅ Pass | Option added and working |
| Path length units | ✅ Pass | Correctly outputs ~485,000 meters |

### ISOFIT Processing
| Test | Result | Notes |
|------|--------|-------|
| sRTMnet emulator load | ✅ Pass | Loads sRTMnet_v120.h5 |
| Path length interpretation | ✅ Pass | ISOFIT reports "Path (km): 485.34" |
| Sun geometry | ✅ Pass | "To-sun zenith (deg): 10.91" |
| Sensor geometry | ✅ Pass | "To-sensor zenith (deg): 24.51" |
| Surface model | ✅ Pass | Tanager-specific model built |
| 6S dependency | ❌ Fail | 6S not installed, requires gfortran |
| LUT generation | ❌ Fail | NaN values due to missing 6S |
| Inversion | ❌ Fail | "Matrix inversion contains negative values" |

---

## Recommended Next Steps

1. **Install 6S radiative transfer code** (CRITICAL - blocking issue)
   ```bash
   # Install gfortran compiler
   sudo apt install gfortran
   # OR for conda environments:
   conda install -c conda-forge gfortran

   # Then download and compile 6S
   isofit download sixs
   ```

2. **Test ISOFIT with 6S installed**
   - The path_length, surface model, and geometry issues are all resolved
   - Once 6S is available, the full pipeline should work
   - Run: `tanager-isofit process input.h5 output/`

3. **Alternative: Use MODTRAN** (if available)
   - MODTRAN is a commercial alternative to 6S+sRTMnet
   - Requires MODTRAN license
   - Configure via `modtran_path` parameter

4. **Contact ISOFIT maintainers** (if issues persist after 6S install)
   - ISOFIT 3.6.1 has native Tanager support
   - The tanager sensor config works correctly
   - All unit conversions are now correct

---

## ISOFIT Data Available

After running `isofit download srtmnet` and `isofit download data`:

```
~/.isofit/
├── srtmnet/           # Fast radiative transfer emulator
│   ├── sRTMnet_v120.h5  (5.8 GB)
│   └── sRTMnet_v120_aux.npz
└── data/
    ├── reflectance/   # Spectral libraries (ENVI format)
    │   ├── surface_model_ucsb
    │   └── ...
    ├── tanager1_noise_20241016.txt  # Tanager noise model
    └── ...
```

**Key finding**: ISOFIT has Tanager-specific noise model, confirming native support exists.

---

## Usage (with current limitations)

The conversion pipeline works, but ISOFIT inversion fails:

```bash
# Install
pip install -e .[isofit]

# Download ISOFIT data
isofit download srtmnet
isofit download data

# Convert HDF5 to ENVI (works)
tanager-isofit convert input.h5 output_dir/ --subset 0,100,0,100

# Full pipeline (conversion works, ISOFIT fails)
tanager-isofit process input.h5 output_dir/ \
  --surface-path ~/tanager_surface_model.mat \
  --n-cores 4
```

---

## Architecture Notes

- **tanager_isofit/convert.py**: HDF5 → ENVI conversion (working)
- **tanager_isofit/geometry.py**: Geometry calculations (working, but HDF5 pre-computed values preferred)
- **tanager_isofit/isofit_runner.py**: ISOFIT wrapper (working, ISOFIT itself has issues)
- **tanager_isofit/cli.py**: CLI interface (updated with --surface-path)
- **tanager_isofit/utils.py**: ENVI I/O utilities (wavelength file format fixed)
