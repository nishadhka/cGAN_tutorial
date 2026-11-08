"""run_local.sh log(s) + timing TSV -> one Markdown report, a section per cycle.

    python runs/make_local_run_report.py --log /srv/state/ifs0p1/run9.log \\
        --timing /srv/state/ifs0p1/timing9.tsv --out runs/2026-10-30-local-9day-endtoend.md \\
        --title "..." --commit 185b789

Everything in the report is read from the log, the TSV, the stores on disk and
the step4 prefix-read counters (CGAN_DAG_HOME/prefix-stats-<pid>.json, matched
to a cycle by modification time); nothing is typed in by hand.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess

STAGES = ["step1_virtual", "step4_realize", "step5_check", "step6_netcdf",
          "step7_jurre-6h", "step7_mvua-7d"]
LABEL = {"step1_virtual": "step1 virtual store", "step4_realize": "step4 realize",
         "step5_check": "step5 check", "step6_netcdf": "step6 NetCDF",
         "step7_jurre-6h": "step7 jurre-6h", "step7_mvua-7d": "step7 mvua-7d"}


def sections(log):
    """{date8: (start_utc, [lines])} from the '################ <date> 00z <utc>' markers."""
    out, cur = {}, None
    for ln in open(log, errors="replace"):
        ln = ln.rstrip("\n")
        # the last section of a cycle wins (a rerun after a failure)
        m = re.match(r"#{16} (\d{8}) (\d\d)z.*?(\d{4}-\d\d-\d\dT\S+Z)\s*$", ln)
        if m:
            cur = m.group(1)
            out[cur] = (m.group(3), [])
            continue
        if cur and not re.search(r"WARN|\.rs:", ln) and ln.strip():
            out[cur][1].append(ln)
    return out


def grab(lines, pat, group=1, default="—"):
    for ln in lines:
        m = re.search(pat, ln)
        if m:
            return m.group(group)
    return default


def du_gb(path):
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True)
                   .stdout.split()[0]) / 1e9
    except Exception:                                          # noqa: BLE001
        return float("nan")


def counters(start, end, home):
    """step4's prefix-read counter file written between start and end (UTC)."""
    best = None
    for f in glob.glob(os.path.join(home, "prefix-stats-*.json")):
        t = dt.datetime.fromtimestamp(os.path.getmtime(f), dt.timezone.utc)
        if start <= t <= end:
            d = json.load(open(f))
            if best is None or d["msgs"] > best["msgs"]:
                best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--commit", required=True, help="cGAN_tutorial commit the run used")
    ap.add_argument("--intro", default="", help="Markdown file inserted after the title")
    ap.add_argument("--stats-home", default=os.environ.get("CGAN_DAG_HOME", "/tmp/frisky-cgan-ea"))
    a = ap.parse_args()

    secs = sections(a.log)
    tim = {}
    for r in csv.DictReader(open(a.timing), delimiter="\t"):
        tim.setdefault(r["cycle"][:8], {})[r["stage"]] = (int(r["seconds"]), r["finished_utc"])

    out = [f"# {a.title}", ""]

    # summary table
    out += ["## Summary, seconds per stage", "",
            "| Cycle | " + " | ".join(LABEL[s] for s in STAGES) + " | **total** | step4 msg/s | fetched GB | step5 |",
            "|---|" + "---:|" * (len(STAGES) + 3) + "---|"]
    totals = []
    for d in sorted(secs):
        t = tim.get(d, {})
        if "total" not in t:
            continue
        lines = secs[d][1]
        tot = t["total"][0]
        totals.append(tot)
        out.append(f"| {d} 00z | " + " | ".join(str(t.get(s, ('—',))[0]) for s in STAGES)
                   + f" | **{tot}** ({tot / 60:.1f} min) | {grab(lines, r'^wall\s+\d+s\s+([\d.]+) msg/s')}"
                   + f" | {grab(lines, r'^bytes\s+fetched ([\d.]+) of')} | {grab(lines, r'^VERDICT\s+(\w+)')} |")
    if totals:
        out += ["", f"{len(totals)} cycles: total mean **{sum(totals) / len(totals) / 60:.1f} min**, "
                f"min {min(totals) / 60:.1f}, max {max(totals) / 60:.1f}; "
                f"all {sum(totals) / 3600:.2f} h.", ""]

    if a.intro:
        out += [open(a.intro).read().rstrip(), ""]

    # per cycle
    for d in sorted(secs):
        start, lines = secs[d]
        t = tim.get(d, {})
        if "total" not in t:
            out += [f"## {d} 00z", "", "Not finished when this report was written.", ""]
            continue
        t0 = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        t1 = dt.datetime.fromisoformat(t["total"][1].replace("Z", "+00:00"))
        sink = grab(lines, r"^=== step4 realize .* -> (\S+)")
        c = counters(t0, t1, a.stats_home)
        out += [f"## {d} 00z", "",
                f"Run {start} → {t['total'][1]}, **{t['total'][0]} s ({t['total'][0] / 60:.1f} min)**.", "",
                "| Stage | s | What happened |", "|---|---:|---|"]
        out.append(f"| step1_build_virtual.py | {t.get('step1_virtual', ('—',))[0]} | "
                   f"{grab(lines, r'(\d+ refs from \d+ .index files on GCS in [\d.]+s)')}; "
                   f"virtual snapshot `{grab(lines, r'-> (\w{20}) ')}`; "
                   f"read-back {grab(lines, r'read-back: (\d+/\d+) chunks')} chunks equal a direct decode |")
        extra = ""
        if c:
            extra = (f"; {c['gets']:,} GETs ({c['topups']} top-ups), "
                     f"{c['src_gcs']:,} of the first {c['msgs']:,} messages (counter saved every 200) "
                     f"read from the GCS copy after AWS 503 Slow Down")
        out.append(f"| step4_realize.py `--no-cluster --threads 12` | {t.get('step4_realize', ('—',))[0]} | "
                   f"{grab(lines, r'^work\s+(.*), prefix read')}; "
                   f"{grab(lines, r'^wall\s+\d+s\s+([\d.]+ msg/s)')}; "
                   f"fetched {grab(lines, r'^bytes\s+fetched (.*?) whole-message')} whole-message bytes "
                   f"(saved {grab(lines, r'saved ([\d.]+%)')}){extra}; "
                   f"store `{grab(lines, r'^committed (\w+)')}`, {du_gb(sink):.2f} GB on disk |")
        out.append(f"| step5_check.py `--samples 8` | {t.get('step5_check', ('—',))[0]} | "
                   f"{grab(lines, r'complete\s+(.*)')}; {grab(lines, r'^\s+(\d+/\d+ bit-exact)')} "
                   f"vs whole messages through the virtual store; **{grab(lines, r'^VERDICT\s+(\w+)')}** |")
        nc6 = grab(lines, r"cgan-ifs-6h-ens/IFS_\d+_\d+Z.nc\s+(\d+ MB)")
        nc7 = grab(lines, r"cgan-ifs-7d-ens/IFS_\d+_\d+Z.nc\s+(\d+ MB)")
        out.append(f"| step6_sewaa_netcdf.py | {t.get('step6_netcdf', ('—',))[0]} | "
                   f"`cgan-ifs-6h-ens/IFS_{d}_00Z.nc` {nc6}, `cgan-ifs-7d-ens/IFS_{d}_00Z.nc` {nc7} "
                   f"(51 members, sd ddof 0) |")
        dr = [ln.strip() for ln in lines if ln.startswith("dry run")]
        for p, k in (("jurre-6h", 0), ("mvua-7d", 1)):
            out.append(f"| step7_infer.py {p} `--dry-run` | {t.get('step7_' + p, ('—',))[0]} | "
                       f"{dr[k] if k < len(dr) else '—'} |")
        out.append("")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
