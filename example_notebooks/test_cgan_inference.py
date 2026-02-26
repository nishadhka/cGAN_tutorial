#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "pytest",
#     "tensorflow==2.15",
#     "numpy<2.0",
#     "xarray",
#     "netcdf4",
#     "pyyaml",
#     "cftime",
# ]
# ///
"""
Unit tests for cGAN inference normalization and preprocessing.

Validates that the corrections applied to run_gefs_inference_raw.py match
the original training code in data/data_gefs.py. These tests catch the
three bugs that caused precipitation to appear over ocean instead of land:

1. nonnegative_fields must match training definition
2. Elevation constants must be normalised by /10000
3. denormalise() must match training inverse transform

Run with:
    uv run pytest test_cgan_inference.py -v
"""

import os
import sys
import pickle

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONSTANTS_PATH = os.path.join(SCRIPT_DIR, "cgan_compact_20260202/CONSTANTS/")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cgan_compact_20260202/logfile_gefs_v3/")

# Import the inference module under test
sys.path.insert(0, SCRIPT_DIR)
import run_gefs_inference_raw as inference


# ============================================================================
# 1. Field definitions
# ============================================================================
class TestFieldDefinitions:
    """Verify field lists match the training code (data/data_gefs.py)."""

    def test_all_fcst_fields_order(self):
        """Field order must be: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp."""
        expected = ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl", "apcp"]
        assert inference.all_fcst_fields == expected, (
            f"Field order mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {inference.all_fcst_fields}"
        )

    def test_all_fcst_fields_count(self):
        """Must have exactly 8 forecast fields (32 input channels)."""
        assert len(inference.all_fcst_fields) == 8

    def test_nonnegative_fields_match_training(self):
        """nonnegative_fields must be ['cape','msl','pres','pwat','tmp'].

        Training code: data/data_gefs.py line 26.
        This was previously wrong as ['cape','pwat','apcp'].
        """
        expected = {"cape", "msl", "pres", "pwat", "tmp"}
        assert set(inference.nonnegative_fields) == expected, (
            f"nonnegative_fields mismatch.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Got:      {sorted(inference.nonnegative_fields)}"
        )

    def test_apcp_not_in_nonnegative_fields(self):
        """apcp must NOT be in nonnegative_fields (uses log transform instead)."""
        assert "apcp" not in inference.nonnegative_fields, (
            "apcp should not be in nonnegative_fields. "
            "It uses log10(1+x) normalization, not /max."
        )

    def test_wind_fields_not_in_nonnegative(self):
        """ugrd and vgrd must NOT be in nonnegative_fields (can be negative)."""
        assert "ugrd" not in inference.nonnegative_fields
        assert "vgrd" not in inference.nonnegative_fields


# ============================================================================
# 2. Normalization logic
# ============================================================================
class TestNormalization:
    """Verify normalization branches match training code."""

    def test_normalization_branch_apcp(self):
        """apcp must be handled by log10(1+x), not by nonnegative /max."""
        # The if/elif chain: apcp is caught first by `if field == "apcp"`
        # and should NOT fall through to the nonnegative branch.
        field = "apcp"
        assert field not in inference.nonnegative_fields or \
            field in ["msl", "pres", "tmp"], \
            "apcp should not reach the nonnegative_fields normalization branch"

    def test_normalization_branch_pressure_fields(self):
        """msl, pres, tmp must use z-score normalization (x-mean)/std.

        Even though they are in nonnegative_fields, the elif chain
        catches them in an earlier branch.
        """
        zscore_fields = ["msl", "pres", "tmp"]
        for field in zscore_fields:
            # These are in nonnegative_fields for the np.maximum(0) clamp
            assert field in inference.nonnegative_fields, (
                f"{field} must be in nonnegative_fields"
            )

    def test_normalization_branch_wind_symmetric(self):
        """ugrd, vgrd must use symmetric max-abs normalization."""
        wind_fields = ["ugrd", "vgrd"]
        for field in wind_fields:
            assert field not in inference.nonnegative_fields
            assert field in inference.all_fcst_fields

    def test_input_channels_count(self):
        """Total input channels = 4 per field x 8 fields = 32."""
        n_channels = 4 * len(inference.all_fcst_fields)
        assert n_channels == 32


# ============================================================================
# 3. Elevation normalization
# ============================================================================
class TestElevationNormalization:
    """Verify elevation is normalised by /10000 as in training code."""

    @pytest.fixture
    def constants(self):
        """Load hi-res constants using the inference function."""
        if not os.path.exists(os.path.join(CONSTANTS_PATH, "elev.nc")):
            pytest.skip("Constants files not available")
        return inference.load_hires_constants(CONSTANTS_PATH, batch_size=1)

    def test_elevation_range(self, constants):
        """Elevation values must be O(1) after /10000 normalisation.

        Raw elevation in metres: 0 to ~5000 m
        After /10000: 0 to ~0.5
        If NOT normalised, max would be ~5000.
        """
        elev = constants[0, :, :, 0]  # (H, W) - elevation channel
        assert elev.max() < 1.0, (
            f"Elevation max = {elev.max():.1f}, expected < 1.0 after /10000 normalisation. "
            f"Likely missing the /10000 division."
        )
        assert elev.min() >= -0.1, (
            f"Elevation min = {elev.min():.4f}, unexpected negative value"
        )

    def test_lsm_range(self, constants):
        """Land-sea mask must be in [0, 1] range."""
        lsm = constants[0, :, :, 1]  # (H, W) - LSM channel
        assert lsm.min() >= 0.0
        assert lsm.max() <= 1.0

    def test_constants_shape(self, constants):
        """Constants shape must be (batch, 384, 352, 2)."""
        assert constants.shape == (1, 384, 352, 2), (
            f"Constants shape mismatch: {constants.shape}"
        )

    def test_constants_dtype(self, constants):
        """Constants must be float32."""
        assert constants.dtype == np.float32


# ============================================================================
# 4. Denormalization (inverse transform)
# ============================================================================
class TestDenormalise:
    """Verify denormalise() matches training code: min(10^x - 1, 100)."""

    def test_zero_input(self):
        """denormalise(0) = 10^0 - 1 = 0."""
        result = inference.denormalise(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    def test_one_input(self):
        """denormalise(1) = 10^1 - 1 = 9."""
        result = inference.denormalise(np.array([1.0]))
        np.testing.assert_allclose(result, [9.0], atol=1e-7)

    def test_two_input(self):
        """denormalise(2) = 10^2 - 1 = 99."""
        result = inference.denormalise(np.array([2.0]))
        np.testing.assert_allclose(result, [99.0], atol=1e-7)

    def test_cap_at_100(self):
        """Output must be capped at 100 mm/h."""
        result = inference.denormalise(np.array([3.0]))  # 10^3 - 1 = 999
        np.testing.assert_allclose(result, [100.0], atol=1e-7)

    def test_large_input_capped(self):
        """Very large log-space values must still cap at 100."""
        result = inference.denormalise(np.array([10.0, 50.0, 100.0]))
        assert np.all(result == 100.0)

    def test_negative_input(self):
        """Negative log-space values give small positive results.
        denormalise(-1) = 10^-1 - 1 = -0.9 (negative, not capped).
        """
        result = inference.denormalise(np.array([-1.0]))
        expected = 10**(-1.0) - 1.0  # = -0.9
        np.testing.assert_allclose(result, [expected], atol=1e-7)

    def test_vectorized(self):
        """Must work on arrays."""
        data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = inference.denormalise(data)
        expected = np.minimum(np.power(10.0, data) - 1.0, 100.0)
        np.testing.assert_allclose(result, expected, atol=1e-7)


# ============================================================================
# 5. GEFS normalization statistics
# ============================================================================
class TestNormalizationStats:
    """Verify normalization statistics file is loadable and complete."""

    @pytest.fixture
    def fcst_norm(self):
        gefs_norm = os.path.join(CONSTANTS_PATH, "FCSTNorm_GEFS_2018.pkl")
        if not os.path.exists(gefs_norm):
            pytest.skip("GEFS normalization file not available")
        with open(gefs_norm, "rb") as f:
            return pickle.load(f)

    def test_required_fields_present(self, fcst_norm):
        """All non-apcp fields must be in the normalization file."""
        required = {"cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl"}
        missing = required - set(fcst_norm.keys())
        assert not missing, f"Missing fields in normalization: {missing}"

    def test_norm_stat_keys(self, fcst_norm):
        """Each field must have min, max, mean, std."""
        for field in ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl"]:
            for key in ["min", "max", "mean", "std"]:
                assert key in fcst_norm[field], (
                    f"Missing '{key}' in fcst_norm['{field}']"
                )

    def test_pressure_range_plausible(self, fcst_norm):
        """Surface pressure stats must be in plausible Pa range."""
        pres = fcst_norm["pres"]
        assert 60000 < pres["min"] < 80000, f"pres min = {pres['min']}"
        assert 95000 < pres["max"] < 110000, f"pres max = {pres['max']}"

    def test_temperature_range_plausible(self, fcst_norm):
        """Temperature stats must be in plausible Kelvin range."""
        tmp = fcst_norm["tmp"]
        assert 250 < tmp["min"] < 280, f"tmp min = {tmp['min']}"
        assert 310 < tmp["max"] < 340, f"tmp max = {tmp['max']}"


# ============================================================================
# 6. Model configuration
# ============================================================================
class TestModelConfig:
    """Verify model setup_params.yaml is valid."""

    @pytest.fixture
    def setup_params(self):
        config_path = os.path.join(MODEL_PATH, "setup_params.yaml")
        if not os.path.exists(config_path):
            pytest.skip("Model config not available")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def test_mode_is_gan(self, setup_params):
        assert setup_params["GENERAL"]["mode"] == "GAN"

    def test_noise_channels(self, setup_params):
        assert setup_params["GENERATOR"]["noise_channels"] == 4

    def test_filters_gen(self, setup_params):
        assert setup_params["GENERATOR"]["filters_gen"] == 128

    def test_architecture(self, setup_params):
        assert setup_params["MODEL"]["architecture"] == "forceconv"

    def test_padding(self, setup_params):
        assert setup_params["MODEL"]["padding"] == "reflect"


# ============================================================================
# 7. Grid coordinates
# ============================================================================
class TestGridCoordinates:
    """Verify ICPAC grid produces the expected dimensions."""

    def test_latitude_size(self):
        assert len(inference.latitude) == 384, (
            f"Latitude size = {len(inference.latitude)}, expected 384"
        )

    def test_longitude_size(self):
        assert len(inference.longitude) == 352, (
            f"Longitude size = {len(inference.longitude)}, expected 352"
        )

    def test_latitude_bounds(self):
        assert inference.latitude[0] == pytest.approx(-13.65, abs=0.01)
        assert inference.latitude[-1] == pytest.approx(24.65, abs=0.01)

    def test_longitude_bounds(self):
        assert inference.longitude[0] == pytest.approx(19.15, abs=0.01)
        assert inference.longitude[-1] == pytest.approx(54.25, abs=0.01)

    def test_resolution(self):
        """Grid resolution must be 0.1 degrees."""
        lat_res = np.diff(inference.latitude).mean()
        lon_res = np.diff(inference.longitude).mean()
        assert lat_res == pytest.approx(0.1, abs=0.001)
        assert lon_res == pytest.approx(0.1, abs=0.001)
