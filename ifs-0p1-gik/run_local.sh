#!/bin/bash
# The SEWAA cGAN products on ONE machine: no uploads, no Ceph, no EWC workers,
# no cluster (step4 --no-cluster: block tasks on threads in one process).
#   [step1 only with BUILD_VD=1] -> step4 realize the lead times the products
#   need from the PUBLIC virtual store (read anonymously) into a local store ->
#   step5 check -> step6 SEWAA NetCDF per product -> step7 cGAN inference
#
#   bash run_local.sh 20261026                       # 00z, jurre-6h + mvua-7d
#   PRODUCTS=jurre-6h bash run_local.sh 2026102612   # 12z, 6 h model only
#   MEMBERS=5 PRODUCTS=jurre-6h bash run_local.sh 20261026   # a small trial
#
# What each product needs (config.PRODUCTS; vendor/PROVENANCE.md):
#   jurre-6h   +30..54 h,  5 lead times x 14 fields x 51 = 3,570 messages
#   mvua-7d    +0..174 h, 30 lead times x 14 fields x 51 = 21,420 messages (00z/12z only)
#   both       the union, 30 lead times (jurre-6h's are a subset)
#
# Env:
#   PRODUCTS            "jurre-6h mvua-7d" (default) or one of them
#   VD                  virtual store spec (default: the public one on source.coop)
#   BUILD_VD=1          build/append the cycle in a LOCAL virtual store first
#                       (step1) and read that instead; needs no upload either
#   IFS0P1_LOCAL_ROOT   stores (default ~/icechunk-local); OUTDIR (./nc); GAN_OUTDIR (./gan)
#   THREADS (12 block threads), MEMBERS (all 51), NC_MEMBERS (all), DDOF (1)
#   CLUSTER=1           step4 on a Frisky local cluster instead: WORKERS (2) x THREADS (4)
#   TIMING              per-stage seconds appended here (default $D/logs/timing_local.tsv)
#   JURRE_MODEL_FOLDER, JURRE_CHECKPOINT (316800), JURRE_CONSTANTS
#   MVUA_MODEL_FOLDER,  MVUA_CHECKPOINT (300800),  MVUA_CONSTANTS
#                       a product whose model folder is unset gets step7 --dry-run
#   UNTRAINED=1         TEST ONLY: step7 with random weights (needs *_MODEL_FOLDER
#                       for setup_params.yaml)
#   GAN_MEMBERS (50), TFPY (python with TensorFlow 2.15, default $D/.venv-infer)
#
# The NetCDFs land in $OUTDIR/cgan-ifs-6h-ens/ and $OUTDIR/cgan-ifs-7d-ens/,
# the folder names the SEWAA jobs read, so they can be dropped into
# sewaa-forecasts-package data/jobs/ or the IFS server's Operational dirs as is.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/.venv/bin/python}
TFPY=${TFPY:-$D/.venv-infer/bin/python}
c=${1:?YYYYMMDD[HH]}; [ ${#c} = 8 ] && c=${c}00
DATE=${c:0:8}; RUN=${c:8:2}
PRODUCTS=${PRODUCTS:-jurre-6h mvua-7d}
ROOT=${IFS0P1_LOCAL_ROOT:-$HOME/icechunk-local}
OUT=${OUTDIR:-$D/nc}
GAN_OUT=${GAN_OUTDIR:-$D/gan}
cfg() { $PY -c "import sys; sys.path.insert(0, '$D'); import config; print($1)"; }
TAG=$(cfg config.BOX_TAG)
PTAG=$(echo $PRODUCTS | tr ' ' '\n' | sort | tr '\n' '+' | sed 's/+$//')
SINK=$ROOT/ecmwf-ifs-0p1-cgan14-$TAG/${DATE}T${RUN}z-$PTAG
F='WARN|\.rs:|^\s*$'
mkdir -p "$D/logs" "$ROOT"
TIMING=${TIMING:-$D/logs/timing_local.tsv}
[ -s "$TIMING" ] || printf 'cycle\tproducts\tstage\tseconds\tfinished_utc\n' > "$TIMING"
T0=$(date +%s); T=$T0
tick() {   # stage name -> seconds since the last tick, to stdout and $TIMING
  local now; now=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "${DATE}T${RUN}z" "$PTAG" "$1" $((now - T)) \
    "$(date -u +%FT%TZ)" >> "$TIMING"
  echo "    [time] $1 $((now - T))s"; T=$now
}
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_ENDPOINT_URL || true
case $RUN in 06|18) [[ " $PRODUCTS " == *" mvua-7d "* ]] && {
  echo "${RUN}z stops at 144 h: mvua-7d needs +174 h, use 00z or 12z"; exit 2; } ;; esac

if [ "${BUILD_VD:-0}" = 1 ]; then
  VDDIR=$ROOT/ecmwf-ifs-0p1-icechunk-vd; VD=local:$VDDIR
  echo "=== step1 local virtual store $VDDIR"
  if ! $PY - "$VDDIR" "$DATE" "$RUN" <<'EOF'
import sys, numpy as np, icechunk as ic, zarr
vd, d, r = sys.argv[1:]
st = ic.local_filesystem_storage(vd)
if not ic.Repository.exists(st):
    sys.exit(1)
g = zarr.open_group(ic.Repository.open(st).readonly_session("main").store,
                    path=f"0p1/{r}z", mode="r")
want = (np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}T{r}") - np.datetime64("1970-01-01T00")) // np.timedelta64(1, "h")
sys.exit(0 if want in g["time"][:] else 1)
EOF
  then
    $PY "$D/step1_build_virtual.py" --date $DATE --run $RUN --resolution 0p1 --store "$VDDIR" \
        --verify 8 2>&1 | grep --line-buffered -v -E "$F"
  else
    echo "    ${DATE} ${RUN}z already in the local virtual store"
  fi
  tick step1_virtual
else
  VD=${VD:-$(cfg 'config.vd_spec("sourcecoop")')}
  echo "=== virtual store $VD (public, anonymous; nothing built or uploaded here)"
fi

EXTRA=()
[ -n "${MEMBERS:-}" ] && EXTRA+=(--members $MEMBERS)
if [ "${CLUSTER:-0}" = 1 ]; then
  EXTRA+=(--workers ${WORKERS:-2} --threads ${THREADS:-4})
else
  EXTRA+=(--no-cluster --threads ${THREADS:-12})
fi
echo "=== step4 realize $PRODUCTS -> $SINK"
$PY -u "$D/step4_realize.py" --date $DATE --run $RUN --vd "$VD" --sink "local:$SINK" \
    --product $PRODUCTS "${EXTRA[@]}" 2>&1 | grep --line-buffered -v -E "$F"
tick step4_realize

echo "=== step5 check"
$PY -u "$D/step5_check.py" --store "local:$SINK" --vd "$VD" --samples 8 2>&1 | grep --line-buffered -v -E "$F"
tick step5_check

echo "=== step6 SEWAA NetCDF -> $OUT"
$PY -u "$D/step6_sewaa_netcdf.py" --store "local:$SINK" --outdir "$OUT" --product $PRODUCTS \
    --members ${NC_MEMBERS:-all} --ddof ${DDOF:-1} 2>&1 | grep --line-buffered -v -E "$F"
tick step6_netcdf

f=IFS_${DATE}_${RUN}Z.nc
for p in $PRODUCTS; do
  case $p in
    jurre-6h) MF=${JURRE_MODEL_FOLDER:-}; CK=${JURRE_CHECKPOINT:-316800}
              CO=${JURRE_CONSTANTS:-$D/../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS}
              NAME=jurre-brishti-ens ;;
    mvua-7d)  MF=${MVUA_MODEL_FOLDER:-}; CK=${MVUA_CHECKPOINT:-300800}
              CO=${MVUA_CONSTANTS:-$D/../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS}
              NAME=mvua-kubwa-ens ;;
  esac
  IN=$OUT/$(cfg "config.PRODUCTS['$p']['dir']")/$f
  if [ -n "$MF" ]; then
    W=(--checkpoint $CK); [ "${UNTRAINED:-0}" = 1 ] && W=(--untrained)
    echo "=== step7 $p cGAN inference -> $GAN_OUT/$NAME"
    $TFPY -u "$D/step7_infer.py" --nc "$IN" --constants "$CO" --model-folder "$MF" "${W[@]}" \
        --outdir "$GAN_OUT/$NAME" --ensemble-members ${GAN_MEMBERS:-50} 2>&1 \
        | grep --line-buffered -E 'input|model|window|wrote|Error|error'
  else
    echo "=== step7 $p dry run (${p%%-*} model folder not set: inputs checked, no inference)"
    $PY -u "$D/step7_infer.py" --nc "$IN" --constants "$CO" --dry-run 2>&1 \
        | grep --line-buffered -E 'input|window|dry run|non-finite|must'
  fi
  tick "step7_$p"
done
T=$T0; tick total
