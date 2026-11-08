#!/bin/bash
# STEP 1 (+2) -- wait for 0.1 deg cycles to be published on GCS, append each to
# the LOCAL virtual store, and -- only with PUBLISH=1 -- copy the store to
# source.coop. One cycle per argument, YYYYMMDDHH.
#
#   bash step1_append_when_published.sh 2026102700                 # local only
#   PUBLISH=1 bash step1_append_when_published.sh 2026102700       # + source.coop
#
# Env: VD_LOCAL (default ~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd),
#      SC_ENV (source.coop .env, PUBLISH=1 only), PY (default ./.venv/bin/python)
set -u
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/.venv/bin/python}
VD=${VD_LOCAL:-$HOME/icechunk-local/ecmwf-ifs-0p1-icechunk-vd}
G=https://storage.googleapis.com/ecmwf-open-data
for c in "$@"; do
  d=${c:0:8}; r=${c:8:2}; last=$([ $r = 00 -o $r = 12 ] && echo 360 || echo 144)
  echo "=== $c waiting for the last step's 4 .index and 4 .grib2 files $(date -u +%FT%TZ)"
  for i in $(seq 1 120); do          # up to 10 h, every 5 min
    n=0
    # the .grib2 too: on 2026-11-08 GCS had the 360 h .index a minute before
    # its .grib2, and step1 --verify then failed on a reference into it
    for f in enfo/${d}${r}0000-${last}h-enfo-ef-pl enfo/${d}${r}0000-${last}h-enfo-ef-sfc \
             oper/${d}${r}0000-${last}h-oper-fc-pl oper/${d}${r}0000-${last}h-oper-fc-sfc; do
      for x in index grib2; do
        [ "$(curl -s -o /dev/null -w '%{http_code}' -I $G/$d/${r}z/ifs/0p1/$f.$x)" = 200 ] && n=$((n+1))
      done
    done
    [ $n = 8 ] && n=4 && break
    sleep 300
  done
  [ $n = 4 ] || { echo "=== $c NOT PUBLISHED after 10 h"; continue; }
  t0=$(date +%s)
  $PY "$D/step1_build_virtual.py" --date $d --run $r --resolution 0p1 --store "$VD" --verify 12 \
      2>&1 | grep -v -E 'WARN|\.rs:|^\s*$|  equal$'
  rc=${PIPESTATUS[0]}
  echo "=== $c build exit=$rc wall=$(( $(date +%s)-t0 ))s"
  [ $rc = 0 ] || continue
  if [ "${PUBLISH:-0}" = 1 ]; then
    ( set -a; . "${SC_ENV:?set SC_ENV to the source.coop .env}"; set +a
      eval "$(source-coop creds --format env)"
      $PY -u "$D/publish_local_to_source_coop.py" --src "$VD" \
          --dest forecasts/ecmwf-ifs-0p1-icechunk-vd --threads 12 ) 2>&1 | grep -v '^  retry'
  fi
  echo "=== $c done $(date -u +%FT%TZ)"
done
