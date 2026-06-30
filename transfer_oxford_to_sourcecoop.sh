#!/usr/bin/env bash
# transfer_oxford_to_sourcecoop.sh
# ─────────────────────────────────────────────────────────────────────────────
# One routine: turn the ~388 GB of raw Oxford IFS (14 fields x 4 years) into
# TFRecords and publish them to source.coop, NEVER holding more than ~1 year of
# raw data on disk at once.
#
#   for YEAR in 2018 2019 2020 2021:
#       download 14 IFS fields (~97 GB)         [prep_year.sh 3a]
#       gen_fcst_norm (2018 only)               [3b]
#       write_data(YEAR) -> TFRecords (~3.5-5 GB) [3c]
#       upload --year YEAR to source.coop       [3d]  (resumable, 1h-STS aware)
#       delete that year's TFRecords (verified) [3d --clean]
#       rm that year's raw IFS                  [3e]
#
# Publishes to:
#   s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/cgan_tfrecords_ou/
#   (https://source.coop/e4drr-project/forecasts/cgan_tfrecords_ou)
#
# Peak disk ≈ one year raw (~97 GB) + that year's TFRecords (~5 GB) ≈ ~102 GB.
#
# PREREQ:
#   * tf215gpu env (TensorFlow 2.15)            -> see RFE2_cGAN_RUNBOOK.md §2.0
#   * source.coop publisher STS creds in .env   -> AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN
#   * s3fs installed in the env running the uploader
#
# Usage:
#   ./transfer_oxford_to_sourcecoop.sh                 # all 4 years
#   ./transfer_oxford_to_sourcecoop.sh 2020 2021       # subset of years
#   DATA_ROOT=/mnt/big ./transfer_oxford_to_sourcecoop.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── config (override via env) ──
export CGAN_PY="${CGAN_PY:-micromamba run -n tf215gpu python}"
export REPO_DIR="${REPO_DIR:-/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan}"
export DATA_ROOT="${DATA_ROOT:-/data/CGAN}"
export UPLOADER="${UPLOADER:-$HERE/cgan_tfrecords_source_coop.py}"
export SC_BUCKET="${SC_BUCKET:-us-west-2.opendata.source.coop}"
export SC_PREFIX="${SC_PREFIX:-e4drr-project/forecasts/cgan_tfrecords_ou}"
export TFRECORDS_DIR="${TFRECORDS_DIR:-$DATA_ROOT/rfe_tfrecords}"
export CONSTANTS_DIR="${CONSTANTS_DIR:-$DATA_ROOT/cGAN_data}"
export NORM_PKL="${NORM_PKL:-$DATA_ROOT/FCSTNorm2018.pkl}"

YEARS=("$@"); [[ ${#YEARS[@]} -eq 0 ]] && YEARS=(2018 2019 2020 2021)
PREP="$HERE/prep_year.sh"

echo "════════════════════════════════════════════════════════════════════"
echo " Oxford IFS -> TFRecords -> source.coop"
echo "   years      : ${YEARS[*]}"
echo "   python     : $CGAN_PY"
echo "   data_root  : $DATA_ROOT  (peak ~102 GB; raw deleted per year)"
echo "   dest       : s3://$SC_BUCKET/$SC_PREFIX/"
echo "════════════════════════════════════════════════════════════════════"

# ── preflight ──
preflight_fail=0
[[ -f "$HERE/.env" ]] || { echo "✗ missing $HERE/.env (source.coop STS creds)"; preflight_fail=1; }
$CGAN_PY -c "import tensorflow, xarray, netCDF4, s3fs" 2>/dev/null \
  || { echo "✗ env missing tensorflow/xarray/netCDF4/s3fs ($CGAN_PY)"; preflight_fail=1; }
[[ -x "$PREP" ]] || { echo "✗ prep_year.sh not found/executable at $PREP"; preflight_fail=1; }
avail_gb=$(df -PBG "$DATA_ROOT" 2>/dev/null | awk 'NR==2{gsub("G","",$4);print $4}' || echo 0)
[[ "${avail_gb:-0}" -ge 120 ]] || echo "⚠ only ${avail_gb}G free at $DATA_ROOT (need ~120G headroom)"
[[ $preflight_fail -eq 0 ]] || { echo "Preflight failed — fix the above and re-run."; exit 1; }
echo "✓ preflight OK (${avail_gb}G free)"

mkdir -p "$DATA_ROOT" "$TFRECORDS_DIR" "$CONSTANTS_DIR"

# ── per-year loop ──
for Y in "${YEARS[@]}"; do
  echo; echo "########## YEAR $Y ##########"
  # prep_year.sh exits 2 when the 1-hour STS budget is hit mid-upload.
  # We retry the SAME year (download/norm already cached, upload resumes) after
  # the operator refreshes .env. We pause and re-read .env automatically.
  for attempt in 1 2 3 4 5; do
    set +e
    if [[ $attempt -eq 1 ]]; then
      "$PREP" "$Y"                                   # full: download->norm->write->upload->rm
    else
      "$PREP" "$Y" --skip-download --skip-norm        # resume: raw cached; upload resumes (idempotent)
    fi
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      echo ">> year $Y complete"; break
    elif [[ $rc -eq 2 ]]; then
      echo ">> year $Y paused (STS credential budget) — attempt $attempt."
      echo ">> Refresh $HERE/.env with fresh source.coop creds, then press ENTER to resume (Ctrl-C to abort)."
      read -r _ || true
    else
      echo "✗ year $Y failed (rc=$rc)"; exit $rc
    fi
  done
done

echo; echo ">> ALL YEARS DONE. Verifying published store (anonymous)…"
$CGAN_PY "$UPLOADER" verify || true
echo ">> Published: https://source.coop/$SC_PREFIX"
