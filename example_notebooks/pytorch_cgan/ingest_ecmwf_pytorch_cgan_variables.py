#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "dask",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "icechunk",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "coiled",
#     "distributed",
#     "bokeh>=3.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
ECMWF IFS — PyTorch EP-cGAN input variables to source.coop Icechunk
====================================================================

Streams the input-channel set required by the PyTorch EP-cGAN
(Xu et al. 2026, DOI 10.1175/WAF-D-24-0199.1) from the AWS S3 ECMWF
Open Data feed into a materialised Icechunk store on source.coop.

Adapted from ``bn-airquality/ingest_ecmwf_fog_variables.py`` (same
pattern: HF parquet refs -> S3 byte-range GRIB fetches via Coiled/Dask
-> Icechunk write).

Channel set — **surface-only pilot configuration (5 channels)**.

The paper uses 11 channels (5 surface + 5 pressure-level + 1 derived).
This pilot temporarily drops the pressure-level set because the current
GIK parquet exposes a different hPa level per lead-time step under the
same key (see ``GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`` in this folder).
Surface vars are unaffected.

  PyTorch EP channel  Source field    Notes
  ──────────────────  ─────────────   ──────────────────────────────────
  tp                  tp              total precipitation (3-h accum diff)
  pad                 (derived)       tp resized 768 km -> 256 km — computed
                                      at training time, NOT stored here
  pw                  tcwv            precipitable water (column water vap.)
  msl                 msl             mean sea-level pressure
  sp                  sp              surface pressure
  cp_proxy            sf              snowfall used as cp proxy
                                      (cp absent from ENS open data)
  u, v, ub, vb, gh    pl              ❌ DISABLED — re-enable once the GIK
                                      parquet exposes per-level keys; see
                                      GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md
  cape_proxy          (mucape)        future — needs MU-CAPE in parquet

The paper's ``lsp`` channel is dropped (not in ENS open data, redundant
with tp-cp). The paper's ``cape`` uses MU-CAPE proxy because standard
CAPE is not in the ENS open data product.

Pipeline (identical to bn-airquality fog ingest):
  HF parquets (E4DRR/gik-ecmwf-par)
    -> Coiled/Dask workers fetch GRIB byte-ranges from s3://ecmwf-forecasts (anon)
    -> Decode with gribberish, subset to bbox
    -> Coordinator writes per-(date) regions to source.coop S3 Icechunk

Credentials:
  Workers     : none (HF public, ECMWF S3 anonymous)
  Coordinator : AWS STS temporary credentials in .env (1-hour lifetime).

Forecast lead-time selection:
  cGAN training only needs IFS leads 9–30 h at 3-h spacing (8 timesteps,
  matching the paper's per-lead-time model schedule). The default
  LEAD_TIME_HOURS reflects this — much shorter than the fog ingest
  (which goes 0–168 h). This cuts ~85% of S3 fetches per init date.

Subcommands:
  init          — Create empty template store on source.coop
  fill          — Populate with real data via Dask/Coiled
  verify        — Inspect store contents (anonymous read)
  probe-levels  — Decode one (u,v,gh)/pl GRIB to print actual hPa levels
                  (REQUIRED ONCE before init — confirms parquet exposes
                  the levels we want)

Usage:
    # 0. probe what levels the parquet exposes for u/v/gh
    uv run ingest_ecmwf_pytorch_cgan_variables.py probe-levels --date 20260301

    # 1. Create .env with fresh STS credentials for source.coop
    cat > .env << 'EOF'
    export AWS_ACCESS_KEY_ID="ASIA..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_SESSION_TOKEN="..."
    export AWS_DEFAULT_REGION="us-west-2"
    EOF

    # 2. One-time template (3 months MAM 2026)
    uv run ingest_ecmwf_pytorch_cgan_variables.py init  \
        --start-date 20260301 --end-date 20260531

    # 3. Fill (deterministic IFS only — fastest for training data)
    uv run ingest_ecmwf_pytorch_cgan_variables.py fill  \
        --start-date 20260301 --end-date 20260531 \
        --n-workers 30 --members-mode deterministic

    # 4. (optional) Fill ensemble as well if you want ENS-conditioned inputs
    uv run ingest_ecmwf_pytorch_cgan_variables.py fill  \
        --start-date 20260301 --end-date 20260531 \
        --n-workers 30 --members-mode ensemble

    # 5. Verify
    uv run ingest_ecmwf_pytorch_cgan_variables.py verify

Author: ICPAC GIK / PyTorch EP-cGAN port
"""

import json
import logging
import os
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingest_ecmwf_pytorch_cgan.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_dotenv_into_environ():
    """Parse SCRIPT_DIR/.env into os.environ if it exists."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_into_environ()

S3_BUCKET = os.environ.get("S3_BUCKET", "us-west-2.opendata.source.coop")
S3_PREFIX = os.environ.get("S3_PREFIX", "e4drr-project/forecasts/ecmwf_pytorch_cgan_ifs")
S3_REGION = os.environ.get("S3_REGION", "us-west-2")

# Alternative backend: GCS (used by --backend gcs). Auth via
# GOOGLE_APPLICATION_CREDENTIALS or --gcs-creds path to a service-account JSON.
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gik-ecmwf-aws-tf")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "pytorch_cgan_ifs")

HF_BASE_URL = (
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/run_par_ecmwf"
)
HF_COMBINED_URL = (
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/combined"
)

# Lead times: the PyTorch EP-cGAN trains 8 per-lead-time models at IFS
# +9, +12, +15, +18, +21, +24, +27, +30 h (paper §2.b — these correspond
# to effective forecast leads +3..+24 h after the 6-h operational delay).
# We fetch +6 h as well so the +9..+30 h diff-accumulation for tp/sf has
# the prior boundary.
LEAD_TIME_HOURS = [6, 9, 12, 15, 18, 21, 24, 27, 30]

MEMBER_IDS_DETERMINISTIC = ["control"]
MEMBER_IDS_ENSEMBLE = ["control"] + [f"ens_{i:02d}" for i in range(1, 51)]

# ECMWF global grid (0.25°)
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# Extended East Africa training domain. Wide enough to capture the
# Somali jet, IOD signature, Congo air boundary, and ITCZ migration —
# the synoptic drivers of EA rainfall (see
# docs/east_africa_kenya_training_plan.md §1).
LAT_MIN, LAT_MAX = -15, 25
LON_MIN, LON_MAX =  20, 53
_lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
_lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
EA_LATS = ECMWF_LATS[LAT_INDICES[0]: LAT_INDICES[-1] + 1]
EA_LONS = ECMWF_LONS[LON_INDICES[0]: LON_INDICES[-1] + 1]
N_LAT = len(EA_LATS)
N_LON = len(EA_LONS)
N_STEPS = len(LEAD_TIME_HOURS)

# ─── Variable definitions ───────────────────────────────────────────────────

# Surface variables (single-channel each, ECMWF name -> stored output name).
SURFACE_VARS: Dict[str, str] = {
    "tp":   "tp",     # total precipitation
    "tcwv": "pw",     # precipitable water (== column water vapour)
    "msl":  "msl",    # mean sea-level pressure
    "sp":   "sp",     # surface pressure
    "sf":   "cp_proxy",  # snowfall used as convective-precip proxy
}
SURFACE_VAR_ATTRS: Dict[str, Dict[str, str]] = {
    "tp":       {"long_name": "Total precipitation", "units": "m",
                 "note": "Accumulated; differentiate across lead_time for 3-h bucket."},
    "pw":       {"long_name": "Precipitable water (column water vapour)", "units": "kg m-2"},
    "msl":      {"long_name": "Mean sea level pressure", "units": "Pa"},
    "sp":       {"long_name": "Surface pressure", "units": "Pa"},
    "cp_proxy": {"long_name": "Snowfall (proxy for convective precipitation)",
                 "units": "m",
                 "note": "Used because convective precipitation 'cp' is not in "
                         "the ECMWF ENS Open Data product. Train and infer on "
                         "the same proxy."},
}

# Pressure-level variables — DISABLED.
#
# The GIK parquet (E4DRR/gik-ecmwf-par) currently exposes only one `pl`
# reference per (variable, step) and the underlying GRIB messages encode
# DIFFERENT pressure levels at different lead times. Probe across 9 steps
# on 20260301 confirmed:
#   gh/pl steps 6,9,12,15,18,21,24,27,30 -> 400, 300, 300, 400, 300, 400, 1000, 1000, 1000 hPa
#   u/pl  same steps                     -> 250, 500, 500, 500, 250, 250, 250, 250, 500 hPa
#   v/pl  same steps                     -> 250, 500, 500, 500, 250, 250, 250, 250, 500 hPa
#
# A single (var, step) -> array tensor would therefore mix levels along
# the lead_time axis — physically meaningless. Surface vars are unaffected.
#
# This is a GIK parquet-construction issue, not an ECMWF Open Data
# limitation. See GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md (same folder) for
# the full write-up and the requested upstream fix in
# https://github.com/icpac-igad/grib-index-kerchunk.
#
# Re-enable by setting these dicts to the entries documented in the MD
# once per-level keys land in the parquet, e.g.
#   ("u", 700): "u", ("v", 700): "v", ("u", 925): "ub", ...
PRESSURE_VARS: Dict[Tuple[str, int], str] = {}
PRESSURE_VAR_ATTRS: Dict[str, Dict[str, str]] = {}

ALL_OUT_NAMES = list(SURFACE_VARS.values()) + list(PRESSURE_VARS.values())

CHUNK_SHAPE = (1, 1, N_STEPS, N_LAT, N_LON)

CREDENTIAL_TIMEOUT_SECONDS = 45 * 60


# ─── Credential / storage helpers ──────────────────────────────────────────


def load_s3_credentials():
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS credentials not found. Create a .env with:\n"
            '  export AWS_ACCESS_KEY_ID="ASIA..."\n'
            '  export AWS_SECRET_ACCESS_KEY="..."\n'
            '  export AWS_SESSION_TOKEN="..."\n'
            '  export AWS_DEFAULT_REGION="us-west-2"'
        )
    return {
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "session_token": session_token,
    }


def make_s3_storage(creds=None, anonymous=False):
    import icechunk

    if anonymous:
        return icechunk.s3_storage(
            bucket=S3_BUCKET, prefix=S3_PREFIX, region=S3_REGION, anonymous=True,
        )
    if creds is None:
        creds = load_s3_credentials()
    return icechunk.s3_storage(
        bucket=S3_BUCKET, prefix=S3_PREFIX, region=S3_REGION,
        access_key_id=creds["access_key_id"],
        secret_access_key=creds["secret_access_key"],
        session_token=creds.get("session_token"),
    )


def make_gcs_storage(anonymous: bool = False):
    """Build an Icechunk GCS storage handle.

    Auth resolution order:
      1. env var GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account JSON
      2. Application Default Credentials (gcloud auth application-default login)
    icechunk reads these via the underlying google-cloud-storage client.
    """
    import icechunk

    if anonymous:
        return icechunk.gcs_storage(
            bucket=GCS_BUCKET, prefix=GCS_PREFIX, from_env=True,
        )
    return icechunk.gcs_storage(
        bucket=GCS_BUCKET, prefix=GCS_PREFIX, from_env=True,
    )


def make_storage(local: str = None, anonymous: bool = False, backend: str = "s3"):
    if local:
        import icechunk
        return icechunk.local_filesystem_storage(path=local)
    if backend == "gcs":
        return make_gcs_storage(anonymous=anonymous)
    return make_s3_storage(anonymous=anonymous)


def setup_gcs_credentials(creds_path: str = None):
    """Point GOOGLE_APPLICATION_CREDENTIALS at the service-account JSON.
    Called by init/fill/verify when --backend gcs and --gcs-creds was given."""
    if creds_path:
        p = Path(creds_path).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"GCS creds file not found: {p}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
        logger.info(f"  Using GCS service account: {p}")
    elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        logger.info(
            f"  Using GCS creds from env: "
            f"{os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"
        )
    else:
        logger.info("  Using GCS Application Default Credentials")


def build_date_list(start_date: str, end_date: str) -> List[str]:
    return [d.strftime("%Y%m%d")
            for d in pd.date_range(start_date, end_date, freq="D")]


def select_members(members_mode: str) -> List[str]:
    if members_mode == "deterministic":
        return MEMBER_IDS_DETERMINISTIC
    elif members_mode == "ensemble":
        return MEMBER_IDS_ENSEMBLE
    else:
        raise ValueError(f"unknown members_mode={members_mode!r}; "
                         "use 'deterministic' or 'ensemble'")


# ─── Phase 0: probe-levels ─────────────────────────────────────────────────


def probe_levels(args):
    """Decode one u/pl, v/pl, gh/pl GRIB byte-range from the parquet to
    print the actual hPa level encoded in each message. Run once before
    init to confirm which levels the parquet exposes."""
    import fsspec
    import xarray as xr

    date_str = args.date
    parquet_url = (
        f"{HF_BASE_URL}/{date_str[:4]}/{date_str[4:6]}/{date_str}/00z/"
        f"{date_str}00z-control.parquet"
    )
    logger.info(f"Reading parquet: {parquet_url}")
    df = pd.read_parquet(parquet_url)

    zstore = {}
    for _, row in df.iterrows():
        k, v = row["key"], row["value"]
        if isinstance(v, bytes):
            try:
                d = v.decode("utf-8")
                v = json.loads(d) if d.startswith(("[", "{")) else d
            except Exception:
                pass
        elif isinstance(v, str) and v.startswith(("[", "{")):
            try:
                v = json.loads(v)
            except Exception:
                pass
        zstore[k] = v

    # Surface key patterns we expect to see in the parquet
    logger.info("\n=== Surface keys observed at step_009 ===")
    for ec_var in SURFACE_VARS:
        for pat in [f"step_009/{ec_var}/sfc/control/0.0.0",
                    f"step_009/{ec_var}/sfc/0.0.0",
                    f"step_009/{ec_var}/surface/control/0.0.0"]:
            if pat in zstore:
                logger.info(f"  ✅ found: {pat}")
                break
        else:
            logger.warning(f"  ❌ NOT found: any of step_009/{ec_var}/sfc/...")

    # Pressure-level: enumerate all keys matching step_009/{var}/pl* to see
    # whether the parquet has per-level keys or just a single entry per var.
    logger.info("\n=== Pressure-level keys observed at step_009 ===")
    for ec_var in {ev for (ev, _lvl) in PRESSURE_VARS}:
        pl_keys = [k for k in zstore if k.startswith(f"step_009/{ec_var}/pl")
                   and k.endswith("0.0.0")]
        logger.info(f"  {ec_var}/pl keys ({len(pl_keys)}): {pl_keys[:5]}")

    s3 = fsspec.filesystem("s3", anon=True)

    def decode_one(key):
        ref = zstore.get(key)
        if not ref:
            logger.warning(f"  key not found: {key}")
            return
        url, off, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"):
            url = url + ".grib2"
        logger.info(f"  fetching {length} bytes from {url} @ {off}")
        with s3.open(url, "rb") as f:
            f.seek(off)
            grib_bytes = f.read(length)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as t:
            t.write(grib_bytes)
            p = t.name
        try:
            ds = xr.open_dataset(p, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
            logger.info(f"  data_vars: {list(ds.data_vars)}")
            for c in ("isobaricInhPa", "level", "pressure"):
                if c in ds.coords:
                    logger.info(f"  level coord '{c}': {ds[c].values}")
            for v in ds.data_vars:
                logger.info(f"  attrs[{v}]: {dict(ds[v].attrs)}")
        finally:
            os.unlink(p)

    logger.info("\n=== Decoding first pl entry per var to identify level ===")
    for ec_var in {ev for (ev, _lvl) in PRESSURE_VARS}:
        first_pl = next(
            (k for k in zstore if k.startswith(f"step_009/{ec_var}/pl")
             and k.endswith("0.0.0")),
            None,
        )
        if first_pl:
            logger.info(f"\n--- {ec_var}/pl (key: {first_pl}) ---")
            decode_one(first_pl)
        else:
            logger.warning(f"  no pl key found for {ec_var}")


# ─── Phase 1: init ──────────────────────────────────────────────────────────


def init_store(args):
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: ECMWF PyTorch-cGAN vars Icechunk store")
    logger.info("=" * 60)
    start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    members = select_members(args.members_mode)
    n_dates = len(dates)
    n_members = len(members)

    logger.info(f"  Dates       : {n_dates} ({dates[0]} -> {dates[-1]})")
    logger.info(f"  Members mode: {args.members_mode} ({n_members} members)")
    logger.info(f"  Lead times  : {N_STEPS} ({LEAD_TIME_HOURS})")
    logger.info(f"  Spatial     : {N_LAT} lat x {N_LON} lon "
                f"({LAT_MIN}..{LAT_MAX}N, {LON_MIN}..{LON_MAX}E)")
    logger.info(f"  Surface vars: {list(SURFACE_VARS.values())}")
    logger.info(f"  Pressure vars (level-tagged): {list(PRESSURE_VARS.values())}")

    init_date = pd.to_datetime(dates).values.astype("datetime64[ns]")
    lead_time = np.array(LEAD_TIME_HOURS, dtype=np.int32)
    member = np.array(members, dtype="U10")

    shape = (n_dates, n_members, N_STEPS, N_LAT, N_LON)
    size_gb = np.prod(shape) * 4 * len(ALL_OUT_NAMES) / (1024 ** 3)
    logger.info(f"  Per-var shape: {shape}")
    logger.info(f"  Uncompressed total ({len(ALL_OUT_NAMES)} vars): {size_gb:.2f} GiB")

    data_vars = {}
    encoding = {}
    all_attrs = {**SURFACE_VAR_ATTRS, **PRESSURE_VAR_ATTRS}
    for out_name in ALL_OUT_NAMES:
        data_vars[out_name] = (
            ("init_date", "member", "lead_time", "lat", "lon"),
            da.zeros(shape, chunks=shape, dtype=np.float32),
            all_attrs[out_name],
        )
        encoding[out_name] = {"chunks": CHUNK_SHAPE, "fill_value": float("nan")}

    template = xr.Dataset(
        data_vars,
        coords={
            "init_date": ("init_date", init_date),
            "member":    ("member", member),
            "lead_time": ("lead_time", lead_time, {"units": "hours"}),
            "lat":       ("lat", EA_LATS, {"units": "degrees_north"}),
            "lon":       ("lon", EA_LONS, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF IFS — PyTorch EP-cGAN input variables (Greater Horn of Africa)",
            "source": "GIK parquet refs (E4DRR/gik-ecmwf-par) -> S3 GRIB byte-ranges -> gribberish",
            "institution": "ICPAC / PyTorch cGAN port",
            "region": "20-53E, 15S-25N (extended East Africa)",
            "reference": "Xu et al. 2026, Wea. Forecasting 41:381–401 (DOI 10.1175/WAF-D-24-0199.1)",
            "variables": ",".join(ALL_OUT_NAMES),
            "n_members": str(n_members),
            "members_mode": args.members_mode,
            "lead_time_hours": ",".join(str(h) for h in LEAD_TIME_HOURS),
            "channel_mapping": (
                "tp=total_precip; pw=column_water_vapour; msl=mslp; sp=sfcp; "
                "cp_proxy=snowfall(cp not in ENS open data). Pressure-level "
                "channels DISABLED — see GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md."
            ),
            "notes": (
                "Surface-only pilot configuration. The paper's u/v/ub/vb/gh "
                "pressure-level channels are temporarily dropped because the "
                "GIK parquet exposes a different hPa level per lead-time step "
                "under the same key. Paper channel 'pad' (768 km tp synoptic "
                "context) is computed at training time from tp — NOT stored. "
                "Paper channel 'lsp' is dropped (not in ENS open data). Paper "
                "channel 'cape' may need a MU-CAPE proxy fetched separately."
            ),
            "storage": (
                f"gs://{GCS_BUCKET}/{GCS_PREFIX}/"
                if args.backend == "gcs"
                else f"s3://{S3_BUCKET}/{S3_PREFIX}/"
            ),
        },
    )
    logger.info(f"  Template:\n{template}")

    storage = make_storage(args.local, backend=args.backend)
    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    session = repo.writable_session("main")
    template.to_zarr(session.store, compute=False, mode="w",
                     encoding=encoding, consolidated=False)
    session.commit("initialize PyTorch EP-cGAN inputs template")

    logger.info("=" * 60)
    logger.info(f"INIT COMPLETE in {time.time() - start:.1f}s")
    logger.info(f"  Target: s3://{S3_BUCKET}/{S3_PREFIX}/")
    logger.info("=" * 60)


# ─── Worker function ────────────────────────────────────────────────────────


def read_member_cgan_vars(
    date_str: str,
    member_id: str,
    lead_time_hours: List[int],
    surface_vars: Dict[str, str],
    pressure_vars: Dict[Tuple[str, int], str],
    hf_base_url: str,
    grid_shape: Tuple[int, int],
    lat_idx_start: int,
    lat_idx_end: int,
    lon_idx_start: int,
    lon_idx_end: int,
    hf_combined_url: str = None,
):
    """One Dask task = one (date, member). Fetches every cGAN variable
    across all 8 lead times."""
    import json
    import os
    import re
    import tempfile
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import fsspec
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    try:
        import gribberish
        has_gribberish = True
    except ImportError:
        has_gribberish = False

    n_steps = len(lead_time_hours)
    n_lat = lat_idx_end - lat_idx_start
    n_lon = lon_idx_end - lon_idx_start
    year = date_str[:4]
    month = date_str[4:6]

    df = None
    if hf_combined_url:
        try:
            combined_url = f"{hf_combined_url}/ecmwf_gik_00z.parquet"
            cache_dir = os.path.join(tempfile.gettempdir(), "gik_combined_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "ecmwf_gik_00z.parquet")
            if not os.path.exists(cache_path):
                import urllib.request
                urllib.request.urlretrieve(combined_url, cache_path)
            import pyarrow.parquet as pq
            table = pq.read_table(
                cache_path,
                filters=[("date", "==", date_str), ("member", "==", member_id)],
            )
            df = table.to_pandas()
            del table
        except Exception:
            df = None

    if df is None or df.empty:
        parquet_url = (
            f"{hf_base_url}/{year}/{month}/{date_str}/00z/"
            f"{date_str}00z-{member_id}.parquet"
        )
        df = pd.read_parquet(parquet_url)

    zstore = {}
    for _, row in df.iterrows():
        key = row["key"]
        value = row["value"]
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
                if decoded.startswith("[") or decoded.startswith("{"):
                    value = json.loads(decoded)
                else:
                    value = decoded
            except Exception:
                pass
        elif isinstance(value, str):
            if value.startswith("[") or value.startswith("{"):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
        zstore[key] = value
    del df

    member_key = member_id.replace("_", "")  # 'ens_01' -> 'ens01'

    all_out_names = list(surface_vars.values()) + list(pressure_vars.values())
    out_data = {
        out_name: np.full((n_steps, n_lat, n_lon), np.nan, dtype=np.float32)
        for out_name in all_out_names
    }

    def find_ref_exact(patterns):
        for p in patterns:
            if p in zstore:
                v = zstore[p]
                if isinstance(v, list) and len(v) >= 3:
                    return v
        return None

    def find_ref_pl(ec_var: str, step_h: int, target_level: int):
        """Find a pl key for ec_var that decodes to target_level hPa.
        Two patterns are tried, in order:
          1. Per-level key: step_NNN/{var}/pl/{level}/...
          2. Single-level-per-var key: step_NNN/{var}/pl/{member}/...
             (caller is responsible for confirming the level via probe-levels)
        """
        # Per-level pattern attempts
        for level_token in (str(target_level), f"{target_level}hPa", f"isobaricInhPa{target_level}"):
            for pat in [
                f"step_{step_h:03d}/{ec_var}/pl/{level_token}/{member_key}/0.0.0",
                f"step_{step_h:03d}/{ec_var}/pl/{level_token}/0.0.0",
            ]:
                ref = find_ref_exact([pat])
                if ref is not None:
                    return ref
        # Single-level fallback (bn-airquality-style)
        return find_ref_exact([
            f"step_{step_h:03d}/{ec_var}/pl/{member_key}/0.0.0",
            f"step_{step_h:03d}/{ec_var}/pl/0.0.0",
        ])

    work: List[Tuple[str, int, list]] = []
    for s_idx, step_h in enumerate(lead_time_hours):
        for ec_var, out_name in surface_vars.items():
            ref = find_ref_exact([
                f"step_{step_h:03d}/{ec_var}/sfc/{member_key}/0.0.0",
                f"step_{step_h:03d}/{ec_var}/sfc/0.0.0",
                f"step_{step_h:03d}/{ec_var}/surface/{member_key}/0.0.0",
            ])
            if ref is not None:
                work.append((out_name, s_idx, ref))

        for (ec_var, level_hPa), out_name in pressure_vars.items():
            ref = find_ref_pl(ec_var, step_h, level_hPa)
            if ref is not None:
                work.append((out_name, s_idx, ref))
    del zstore

    if not work:
        return {"date_str": date_str, "member_id": member_id, "data": out_data}

    s3_fs = fsspec.filesystem("s3", anon=True)

    def _fetch_one(out_name, s_idx, ref):
        url, offset, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"):
            url = url + ".grib2"
        with s3_fs.open(url, "rb") as f:
            f.seek(offset)
            grib_bytes = f.read(length)

        arr = None
        if has_gribberish:
            try:
                flat = gribberish.parse_grib_array(grib_bytes, 0)
                arr = flat.reshape(grid_shape)
            except Exception:
                pass
        if arr is None:
            import xarray as xr
            with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
                tmp.write(grib_bytes)
                tmp_path = tmp.name
            try:
                ds = xr.open_dataset(tmp_path, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                arr = ds[list(ds.data_vars)[0]].values.copy()
                ds.close()
            finally:
                os.unlink(tmp_path)

        ea_arr = arr[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end].astype(np.float32)
        return out_name, s_idx, ea_arr

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_one, on, si, ref) for on, si, ref in work]
        for fut in as_completed(futs):
            try:
                out_name, s_idx, ea_arr = fut.result()
                out_data[out_name][s_idx] = ea_arr
            except Exception:
                pass

    return {"date_str": date_str, "member_id": member_id, "data": out_data}


# ─── Phase 2a: local-fill (smoke test) ─────────────────────────────────────


def local_fill(args):
    """Single-machine fill — no Coiled. Smoke-tests one day end-to-end
    against the target store."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("LOCAL-FILL: PyTorch cGAN vars Icechunk (no Coiled)")
    logger.info("=" * 60)
    overall_start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    members = select_members(args.members_mode)
    if args.limit_members:
        members = members[:args.limit_members]
    logger.info(f"  Dates  : {len(dates)} ({dates[0]} -> {dates[-1]})")
    logger.info(f"  Members: {len(members)} ({members[0]} .. {members[-1]})")
    logger.info(f"  Member-parallelism: {args.member_workers}")

    target_storage = make_storage(args.local, backend=args.backend)
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default()
    )

    session_ro = target_repo.readonly_session("main")
    template_ds = xr.open_zarr(session_ro.store, consolidated=False)
    init_dates = pd.to_datetime(template_ds["init_date"].values)
    date_to_idx = {d.strftime("%Y%m%d"): i for i, d in enumerate(init_dates)}
    template_ds.close()

    lat_idx_start = int(LAT_INDICES[0])
    lat_idx_end = int(LAT_INDICES[-1]) + 1
    lon_idx_start = int(LON_INDICES[0])
    lon_idx_end = int(LON_INDICES[-1]) + 1

    total_written = 0
    for date_str in dates:
        if date_str not in date_to_idx:
            logger.error(f"  date {date_str} not in template — skip")
            continue
        date_idx = date_to_idx[date_str]
        t_date = time.time()
        logger.info(f"\n  -- {date_str}  (init_date idx {date_idx}) --")

        member_data: Dict[int, dict] = {}
        n_ok = 0
        n_fail = 0

        with ThreadPoolExecutor(max_workers=args.member_workers) as pool:
            futs = {
                pool.submit(
                    read_member_cgan_vars,
                    date_str, member_id, LEAD_TIME_HOURS,
                    SURFACE_VARS, PRESSURE_VARS, HF_BASE_URL,
                    ECMWF_GRID_SHAPE,
                    lat_idx_start, lat_idx_end, lon_idx_start, lon_idx_end,
                    HF_COMBINED_URL,
                ): (m_idx, member_id)
                for m_idx, member_id in enumerate(members)
            }
            done = 0
            for fut in as_completed(futs):
                m_idx, member_id = futs[fut]
                done += 1
                try:
                    member_data[m_idx] = fut.result()
                    n_ok += 1
                    logger.info(f"    [{done:2d}/{len(members):2d}] {member_id} OK")
                except Exception as e:
                    n_fail += 1
                    logger.error(f"    [{done:2d}/{len(members):2d}] {member_id} FAILED: {e}")

        if n_ok == 0:
            logger.error(f"  {date_str}: all members failed")
            continue

        n_m = len(members)
        arrs = {
            out_name: np.full(
                (n_m, N_STEPS, N_LAT, N_LON), np.nan, dtype=np.float32
            )
            for out_name in ALL_OUT_NAMES
        }
        for m_i, res in member_data.items():
            for out_name in ALL_OUT_NAMES:
                arrs[out_name][m_i] = res["data"][out_name]

        session = target_repo.writable_session("main")
        ds_vars = {
            out_name: (
                ("init_date", "member", "lead_time", "lat", "lon"),
                arrs[out_name][np.newaxis],
            )
            for out_name in ALL_OUT_NAMES
        }
        ds_write = xr.Dataset(ds_vars)
        ds_write.to_zarr(
            session.store,
            region={"init_date": slice(date_idx, date_idx + 1),
                    "member": slice(0, n_m)},
            consolidated=False,
        )
        session.commit(
            f"fill date {date_idx} ({date_str}): {n_ok}/{len(members)} members [local-fill]"
        )
        total_written += 1
        logger.info(f"  {date_str}: committed in {time.time() - t_date:.1f}s")

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"LOCAL-FILL COMPLETE: {total_written}/{len(dates)} dates "
                f"in {elapsed/60:.1f} min")
    logger.info("=" * 60)


# ─── Phase 2: fill (Coiled/Dask) ────────────────────────────────────────────


def fill_store(args):
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: PyTorch cGAN vars Icechunk store on source.coop")
    logger.info("=" * 60)
    overall_start = time.time()
    session_start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    members = select_members(args.members_mode)
    n_members = len(members)
    n_dates = len(dates)
    logger.info(f"  Dates  : {n_dates} ({dates[0]} -> {dates[-1]})")
    logger.info(f"  Members: {n_members} (mode={args.members_mode})")

    target_storage = make_storage(args.local, backend=args.backend)
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default()
    )

    completed_indices = set()
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill date "):
                try:
                    idx_str = msg.split("fill date ")[1].split(" ")[0]
                    completed_indices.add(int(idx_str))
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    start_idx = max(completed_indices) + 1 if completed_indices else 0
    if start_idx > 0:
        logger.info(f"  Resuming from date index {start_idx} "
                    f"({len(completed_indices)} dates already done)")

    remaining = [(i, d) for i, d in enumerate(dates) if i >= start_idx]
    if not remaining:
        logger.info("  All dates already filled.")
        return
    logger.info(f"  Remaining: {len(remaining)} dates")

    timeout = args.credential_timeout
    logger.info(f"  Credential timeout: {timeout}s ({timeout/60:.0f} min)")

    cluster = coiled.Cluster(
        name=f"ecmwf-cgan-{int(time.time()) % 10000}",
        n_workers=args.n_workers,
        worker_vm_types=args.worker_vm_types,
        package_sync=True,
        region=args.coiled_region,
        idle_timeout="30 minutes",
        workspace=args.workspace,
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(10, args.n_workers), timeout=600)
    logger.info(f"  Cluster ready: {client.dashboard_link}")

    lat_idx_start = int(LAT_INDICES[0])
    lat_idx_end = int(LAT_INDICES[-1]) + 1
    lon_idx_start = int(LON_INDICES[0])
    lon_idx_end = int(LON_INDICES[-1]) + 1

    total_written = 0
    total_failed = 0
    failed_dates: List[str] = []
    timed_out = False

    for date_idx, date_str in remaining:
        elapsed_session = time.time() - session_start
        if elapsed_session > timeout:
            logger.warning(
                f"  Credential timeout ({timeout}s) reached after "
                f"{elapsed_session:.0f}s. Refresh .env and rerun to resume."
            )
            timed_out = True
            break

        date_t0 = time.time()
        logger.info(
            f"\n  -- date {date_idx} ({date_str}) — submitting {n_members} "
            f"member fetches  ({total_written}/{len(remaining)} done)"
        )

        futures = {}
        for m_idx, member_id in enumerate(members):
            future = client.submit(
                read_member_cgan_vars,
                date_str, member_id, LEAD_TIME_HOURS,
                SURFACE_VARS, PRESSURE_VARS, HF_BASE_URL,
                ECMWF_GRID_SHAPE,
                lat_idx_start, lat_idx_end, lon_idx_start, lon_idx_end,
                HF_COMBINED_URL,
                key=f"d{date_idx}-m{m_idx:02d}",
            )
            futures[future] = (m_idx, member_id)

        session = target_repo.writable_session("main")
        n_ok = 0
        n_fail = 0
        write_failed = False
        for future in distributed.as_completed(list(futures.keys())):
            m_idx, member_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                n_fail += 1
                logger.error(f"    member {member_id} (m_idx={m_idx}) FETCH FAILED: {e}")
                continue

            try:
                ds_vars = {
                    out_name: (
                        ("init_date", "member", "lead_time", "lat", "lon"),
                        result["data"][out_name][np.newaxis, np.newaxis],
                    )
                    for out_name in ALL_OUT_NAMES
                }
                xr.Dataset(ds_vars).to_zarr(
                    session.store,
                    region={"init_date": slice(date_idx, date_idx + 1),
                            "member": slice(m_idx, m_idx + 1)},
                    consolidated=False,
                )
                n_ok += 1
                del result, ds_vars
                if n_ok % 10 == 0 or n_ok == n_members:
                    logger.info(
                        f"    [{n_ok + n_fail:2d}/{n_members}] "
                        f"member {member_id} written "
                        f"(ok={n_ok}, failed={n_fail}, "
                        f"{time.time() - date_t0:.0f}s elapsed)"
                    )
            except Exception as e:
                write_failed = True
                logger.error(f"    member {member_id} WRITE FAILED: {e}")
                break

        if write_failed or n_ok == 0:
            total_failed += 1
            failed_dates.append(date_str)
            logger.error(f"    Date {date_idx} ({date_str}) FAILED")
            continue

        try:
            session.commit(
                f"fill date {date_idx} ({date_str}): {n_ok}/{n_members} members"
            )
            total_written += 1
            logger.info(
                f"    Committed date {date_idx} ({date_str}) "
                f"[{n_ok}/{n_members} members] in {time.time() - date_t0:.0f}s"
            )
        except Exception as e:
            total_failed += 1
            failed_dates.append(date_str)
            logger.error(f"    Date {date_idx} ({date_str}) COMMIT FAILED: {e}")

    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE" + (" (timed out — resume)" if timed_out else ""))
    logger.info(f"  Dates written: {total_written}/{n_dates}")
    logger.info(f"  Failed: {total_failed} -- {failed_dates[:20]}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    if args.backend == "gcs":
        logger.info(f"  Store: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    else:
        logger.info(f"  Store: s3://{S3_BUCKET}/{S3_PREFIX}/")
    logger.info("=" * 60)

    results = {
        "status": "timed_out" if timed_out else ("success" if not failed_dates else "partial"),
        "dates_written": total_written,
        "dates_total": n_dates,
        "failed_dates": failed_dates,
        "timed_out": timed_out,
        "elapsed_min": elapsed / 60,
        "members_mode": args.members_mode,
        "n_members": n_members,
    }
    out = SCRIPT_DIR / f"ecmwf_pytorch_cgan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Results: {out}")


# ─── Phase 3: verify ────────────────────────────────────────────────────────


def verify_store(args):
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: PyTorch cGAN vars Icechunk store")
    logger.info("=" * 60)

    storage = make_storage(args.local, anonymous=not args.local, backend=args.backend)
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")
    for dim in ["init_date", "member", "lead_time", "lat", "lon"]:
        if dim in ds.dims:
            v = ds[dim].values
            logger.info(f"  {dim}: {ds.sizes[dim]} [{v[0]} .. {v[-1]}]")

    for var in ds.data_vars:
        d = ds[var]
        logger.info(f"\nVariable '{var}': dtype={d.dtype}, shape={d.shape}")

    if args.spot_check:
        logger.info("\nSpot-check: first date, first member...")
        for var in ds.data_vars:
            try:
                sample = ds[var].isel(init_date=0, member=0).load()
                vals = sample.values
                n_valid = int((~np.isnan(vals)).sum())
                pct = 100 * n_valid / vals.size if vals.size else 0
                line = f"  {var}: {n_valid}/{vals.size} valid ({pct:.1f}%)"
                if n_valid > 0:
                    good = vals[~np.isnan(vals)]
                    line += (f"  min={float(good.min()):.4g}"
                             f"  max={float(good.max()):.4g}"
                             f"  mean={float(good.mean()):.4g}")
                logger.info(line)
            except Exception as e:
                logger.error(f"  {var}: spot-check failed: {e}")

    try:
        commits = list(repo.ancestry(branch="main"))
        logger.info(f"\nCommits ({len(commits)}):")
        for c in commits[:10]:
            logger.info(f"  {c.message}")
        if len(commits) > 10:
            logger.info(f"  ... and {len(commits) - 10} more")
    except Exception:
        pass

    logger.info("\nVerification complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ECMWF IFS PyTorch EP-cGAN input variables to source.coop Icechunk",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--local", type=str, default=None,
                        help="Local Icechunk path (overrides remote backend)")
    common.add_argument("--backend", type=str, default="s3", choices=["s3", "gcs"],
                        help="Remote storage backend: 's3' (source.coop) or 'gcs'")
    common.add_argument("--gcs-creds", type=str, default=None,
                        help="Path to GCS service-account JSON (sets "
                             "GOOGLE_APPLICATION_CREDENTIALS for icechunk). "
                             "Only used when --backend gcs.")
    common.add_argument("--gcs-bucket", type=str, default=None,
                        help="Override GCS bucket (default: $GCS_BUCKET or 'gik-ecmwf-aws-tf')")
    common.add_argument("--gcs-prefix", type=str, default=None,
                        help="Override GCS prefix (default: $GCS_PREFIX or 'pytorch_cgan_ifs')")
    common.add_argument("--members-mode", type=str, default="deterministic",
                        choices=["deterministic", "ensemble"],
                        help="'deterministic' = control only (1 member, fastest for "
                             "training); 'ensemble' = control + 50 ENS members.")

    p_probe = sub.add_parser("probe-levels",
                             help="Decode pl entries to identify exposed hPa levels")
    p_probe.add_argument("--date", type=str, default="20260301")

    p_init = sub.add_parser("init", parents=[common],
                            help="Create empty template store")
    p_init.add_argument("--start-date", type=str, default="20260301")
    p_init.add_argument("--end-date",   type=str, default="20260531")

    p_local = sub.add_parser("local-fill", parents=[common],
                             help="Single-machine smoke test (no Coiled)")
    p_local.add_argument("--start-date",     type=str, default="20260301")
    p_local.add_argument("--end-date",       type=str, default="20260301")
    p_local.add_argument("--limit-members",  type=int, default=None)
    p_local.add_argument("--member-workers", type=int, default=4)

    p_fill = sub.add_parser("fill", parents=[common],
                            help="Fill store from HF parquets + S3 GRIB byte-range reads")
    p_fill.add_argument("--start-date",      type=str, default="20260301")
    p_fill.add_argument("--end-date",        type=str, default="20260531")
    p_fill.add_argument("--n-workers",       type=int, default=20)
    p_fill.add_argument("--worker-vm-types", type=str, default="e2-standard-4")
    p_fill.add_argument("--coiled-region",   type=str, default="us-east1")
    p_fill.add_argument("--workspace",       type=str, default=None)
    p_fill.add_argument("--credential-timeout", type=int,
                        default=CREDENTIAL_TIMEOUT_SECONDS)

    p_verify = sub.add_parser("verify", parents=[common],
                              help="Inspect store contents")
    p_verify.add_argument("--spot-check",    dest="spot_check",
                          action="store_true",  default=True)
    p_verify.add_argument("--no-spot-check", dest="spot_check",
                          action="store_false")

    args = parser.parse_args()

    # Apply backend/creds overrides globally before subcommands run.
    if getattr(args, "gcs_bucket", None):
        globals()["GCS_BUCKET"] = args.gcs_bucket
    if getattr(args, "gcs_prefix", None):
        globals()["GCS_PREFIX"] = args.gcs_prefix
    if getattr(args, "backend", "s3") == "gcs":
        setup_gcs_credentials(getattr(args, "gcs_creds", None))

    if args.command == "probe-levels":
        probe_levels(args)
    elif args.command == "init":
        init_store(args)
    elif args.command == "local-fill":
        local_fill(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
