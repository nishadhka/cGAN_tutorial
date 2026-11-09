"""Compare our SEWAA IFS NetCDF(s) with the operational one SEWAA downloads.

icpac-igad/SEWAA-forecasts run_forecast.py (IFS_SOURCES) fetches
IFS_<date>_<HH>Z.nc first from Oxford:

    https://rain.physics.ox.ac.uk/ICPAC/operational/{6,24}h_accumulations/IFS_forecast_data/<year>/IFS_<date>_<HH>Z.nc

For every field: the median over valid times (min-max) of sd_oxford / sd_ours,
taken where sd_ours is above its median, and the median of mean_oxford /
mean_ours. The last index of the accumulated fields is skipped: Oxford stores 0
there, we store NaN, and neither operational script reads it.

    curl -fLo ox6.nc https://rain.physics.ox.ac.uk/ICPAC/operational/6h_accumulations/IFS_forecast_data/2026/IFS_20261026_00Z.nc
    python runs/compare_with_oxford.py ox6.nc  new nc/cgan-ifs-6h-ens/IFS_20261026_00Z.nc  [label2 file2 ...]
"""
import sys

import netCDF4 as nc
import numpy as np

FIELDS = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw", "tcrw", "tcw", "tcwv",
          "tp", "u700", "v700"]
ACCUMULATED = ("tp", "cp", "ssr")


def ratios(ox_path, ours_path):
    o, m = nc.Dataset(ox_path), nc.Dataset(ours_path)
    nt = o.dimensions["valid_time"].size
    if m.dimensions["valid_time"].size != nt:
        sys.exit(f"{ours_path}: {m.dimensions['valid_time'].size} valid times, Oxford has {nt}")
    out = {}
    for f in FIELDS:
        sr, mr = [], []
        for k in range(nt - (1 if f in ACCUMULATED else 0)):
            def get(d, kind):
                return np.ma.filled(d[f"{f}_ensemble_{kind}"][k], np.nan).astype(float)
            so, sm = get(o, "standard_deviation"), get(m, "standard_deviation")
            xo, xm = get(o, "mean"), get(m, "mean")
            q = np.isfinite(so) & np.isfinite(sm) & (sm > np.nanpercentile(sm, 50))
            if q.sum() > 100:
                sr.append(np.nanmedian(so[q] / sm[q]))
            ok = np.isfinite(xo) & np.isfinite(xm)
            # ssr is ~0 at night, where a ratio of means says nothing
            if f != "ssr" and ok.any() and abs(np.nanmean(xm[ok])) > 0:
                mr.append(np.nanmean(xo[ok]) / np.nanmean(xm[ok]))
        out[f] = (np.median(sr), np.min(sr), np.max(sr), np.median(mr) if mr else np.nan)
    return out


def main():
    if len(sys.argv) < 4 or len(sys.argv) % 2:
        sys.exit(__doc__)
    ox = sys.argv[1]
    res = {lab: ratios(ox, p) for lab, p in zip(sys.argv[2::2], sys.argv[3::2])}
    labs = list(res)
    print("sd_oxford / sd_ours: median over valid times (min-max)")
    print("field  " + "  ".join(f"{lab:>26s}" for lab in labs) + "   mean ratio")
    for f in FIELDS:
        print(f"{f:5s}  " + "  ".join(f"{res[lab][f][0]:.4f} ({res[lab][f][1]:.3f}-{res[lab][f][2]:.3f})"
                                      for lab in labs)
              + f"   {res[labs[0]][f][3]:.4f}")
    for lab in labs:
        print(f"{lab}: median over fields {np.median([res[lab][f][0] for f in FIELDS]):.4f}")


if __name__ == "__main__":
    main()
