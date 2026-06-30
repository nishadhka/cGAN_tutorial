#!/usr/bin/env bash
# prep_year.sh — chain runbook steps 3a→3e for ONE year, disk-frugal.
#
#   3a download raw IFS (14 fields) for the year   (~97 GB, resumable)
#   3b gen_fcst_norm                               (2018 only, once)
#   3c write_data(year)  -> TFRecords              (~8 GB)
#   3d upload --year to source.coop + verify       (then deletes local TFRecords)
#   3e rm the year's raw IFS                        (reclaim ~97 GB)
#
# Run once per year:  ./prep_year.sh 2018  ;  ./prep_year.sh 2019 ; ...
#
# Env knobs (override as needed):
#   CGAN_PY     python with tensorflow+xarray+netCDF4 (REQUIRED for 3c)
#   REPO_DIR    path to .../24h_accumulations/cGAN/dsrnngan
#   DATA_ROOT   where IFS_training/, rfe_tfrecords/, cGAN_data/ live
#   UPLOADER    path to cgan_tfrecords_source_coop.py
# Flags:
#   --skip-download  --skip-upload  --keep-raw  --skip-norm
set -euo pipefail

YEAR="${1:?usage: prep_year.sh <YEAR> [--skip-download|--skip-upload|--keep-raw|--skip-norm]}"
shift || true
SKIP_DL=0; SKIP_UP=0; KEEP_RAW=0; SKIP_NORM=0
for a in "$@"; do case "$a" in
  --skip-download) SKIP_DL=1 ;; --skip-upload) SKIP_UP=1 ;;
  --keep-raw) KEEP_RAW=1 ;; --skip-norm) SKIP_NORM=1 ;;
  *) echo "unknown flag $a"; exit 2 ;;
esac; done

CGAN_PY="${CGAN_PY:-python3}"
REPO_DIR="${REPO_DIR:-/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan}"
DATA_ROOT="${DATA_ROOT:-/data/CGAN}"
UPLOADER="${UPLOADER:-/scratch/notebook/cGAN_tutorial/cgan_tfrecords_source_coop.py}"

BASE="https://rain.physics.ox.ac.uk/ICPAC/training/IFS"
VARS=(cape cp mcc sp ssr t2m tciw tclw tcrw tcw tcwv tp u700 v700)
IFS_DIR="$DATA_ROOT/IFS_training/$YEAR"

echo "════════════════════════════════════════════════════════════"
echo " prep_year $YEAR   python=$CGAN_PY   data_root=$DATA_ROOT"
echo "════════════════════════════════════════════════════════════"

# ── 3a download ────────────────────────────────────────────────────────────
if [[ "$SKIP_DL" == 0 ]]; then
  echo ">> 3a download 14 IFS fields for $YEAR"
  mkdir -p "$IFS_DIR"
  for v in "${VARS[@]}"; do
    out="$IFS_DIR/$v.nc"
    echo "   - $v.nc"
    curl -fL -C - --retry 10 --retry-delay 5 --retry-all-errors \
         -o "$out" "$BASE/$YEAR/$v.nc"
  done
else
  echo ">> 3a skipped"
fi

# ── 3b gen_fcst_norm (2018 only) ───────────────────────────────────────────
if [[ "$YEAR" == 2018 && "$SKIP_NORM" == 0 ]]; then
  echo ">> 3b gen_fcst_norm (writes FCSTNorm2018.pkl)"
  ( cd "$REPO_DIR" && "$CGAN_PY" -c "from data import gen_fcst_norm; gen_fcst_norm(2018)" )
else
  echo ">> 3b skipped (norm only runs for 2018)"
fi

# ── 3c write_data ──────────────────────────────────────────────────────────
echo ">> 3c write_data($YEAR) -> TFRecords"
( cd "$REPO_DIR" && "$CGAN_PY" -c "from tfrecords_generator import write_data; write_data($YEAR)" )

# ── 3d upload + clean ──────────────────────────────────────────────────────
if [[ "$SKIP_UP" == 0 ]]; then
  echo ">> 3d upload year $YEAR to source.coop (+clean local TFRecords)"
  # exit code 2 = credential budget hit; refresh .env and re-run this same line.
  ( cd "$(dirname "$UPLOADER")" && "$CGAN_PY" "$UPLOADER" upload --year "$YEAR" ) || {
      rc=$?
      [[ $rc == 2 ]] && { echo "!! credential budget reached — refresh .env then re-run: $0 $YEAR --skip-download --skip-norm"; exit 2; }
      exit $rc; }
else
  echo ">> 3d skipped"
fi

# ── 3e reclaim raw IFS disk ────────────────────────────────────────────────
if [[ "$KEEP_RAW" == 0 ]]; then
  echo ">> 3e rm raw IFS for $YEAR ($IFS_DIR)"
  rm -rf "$IFS_DIR"
else
  echo ">> 3e kept raw (--keep-raw)"
fi

echo ">> DONE year $YEAR"
