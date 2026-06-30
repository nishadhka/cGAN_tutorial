#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "s3fs",
#     "tqdm",
# ]
# ///
"""
RFE2 cGAN TFRecords — upload to source.coop, then free local disk
=================================================================

Adapted from `grib-index-kerchunk/ecmwf/dev-test/ecmwf_ea_tp_source_coop_zarr.py`
(same source.coop S3 + STS-credential pattern), but for the cGAN training
TFRecords instead of a Zarr store.

Why this shape (carried over from the zarr routine):
  * source.coop write credentials are AWS STS temporary tokens that live for
    ONLY ~1 hour. So the uploader is fully resumable and stops ~15 min before
    expiry; you refresh `.env` and re-run to continue exactly where it left off.
  * Every TFRecord is an independent S3 object -> no transactions, re-runnable.

Disk-frugal by design:
  The prep box is space-constrained. After a file is uploaded AND its remote
  size is confirmed to match the local file, the local copy is DELETED
  (`--clean`, on by default). Run this immediately after each year's
  `write_data(...)` so the ~33 GB of 14-field TFRecords never piles up.

Layout written to source.coop:
  s3://us-west-2.opendata.source.coop/<PREFIX>/
      rfe_tfrecords/2018_1.0.tfrecords ...   (the training data)
      cGAN_data/elev.nc, lsm.nc              (constants, required on GPU)
      FCSTNorm2018.pkl                       (normalisation, required on GPU)
      SHA256SUMS.txt                         (integrity manifest)

Subcommands:
  upload    Upload (resumable) + verify + delete-local. Refresh .env, re-run.
  verify    Anonymous read-back: list objects, counts, sizes, manifest check.
  download  Pull everything to the GPU node's LOCAL disk for training.

Usage:
    # 1. Put fresh STS creds in .env next to this script (shell-export syntax):
    #    export AWS_ACCESS_KEY_ID="ASIA..."
    #    export AWS_SECRET_ACCESS_KEY="..."
    #    export AWS_SESSION_TOKEN="..."
    #    export AWS_DEFAULT_REGION="us-west-2"

    # 2. After each year's write_data(), upload that year and reclaim its disk:
    uv run cgan_tfrecords_source_coop.py upload --year 2018
    #    ... refresh .env if it warns about the credential budget, then ...
    uv run cgan_tfrecords_source_coop.py upload          # uploads whatever's left

    # 3. Check it landed (no credentials needed):
    uv run cgan_tfrecords_source_coop.py verify

    # 4. On the GPU node, pull to local NVMe:
    uv run cgan_tfrecords_source_coop.py download --dest /scratch/CGAN

Author: ICPAC SEWAA / GIK team
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# ─── Config (edit to taste) ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# source.coop target — a real AWS S3 bucket in us-west-2 (same as the zarr routine)
S3_BUCKET = os.getenv("SC_BUCKET", "us-west-2.opendata.source.coop")
S3_PREFIX = os.getenv("SC_PREFIX", "e4drr-project/cgan-rfe2/tfrecords-14field-v1")

# Local sources on the prep box
TFRECORDS_DIR = Path(os.getenv("TFRECORDS_DIR", "/data/CGAN/rfe_tfrecords"))
CONSTANTS_DIR = Path(os.getenv("CONSTANTS_DIR", "/data/CGAN/cGAN_data"))   # elev.nc, lsm.nc
NORM_PKL      = Path(os.getenv("NORM_PKL", "/data/CGAN/FCSTNorm2018.pkl"))

# Stop this long before the 1-hour STS token dies, to never write with an
# expired credential. 45 min budget == 15 min safety margin (same as zarr).
CREDENTIAL_TIMEOUT_SECONDS = int(os.getenv("SC_CRED_TIMEOUT", 45 * 60))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(SCRIPT_DIR / "cgan_tfrecords_source_coop.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("cgan-sc")


# ─── Credentials (1-hour STS tokens from .env) ───────────────────────────────
def load_s3_credentials():
    """Load source.coop STS credentials from a shell-export .env file."""
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    st = os.getenv("AWS_SESSION_TOKEN")
    if not ak or not sk:
        raise RuntimeError(
            "No AWS credentials. Create .env next to this script with:\n"
            '  export AWS_ACCESS_KEY_ID="ASIA..."\n'
            '  export AWS_SECRET_ACCESS_KEY="..."\n'
            '  export AWS_SESSION_TOKEN="..."\n'
            '  export AWS_DEFAULT_REGION="us-west-2"')
    return {"access_key_id": ak, "secret_access_key": sk, "session_token": st}


def make_s3fs(anonymous=False):
    import s3fs
    if anonymous:
        return s3fs.S3FileSystem(anon=True)
    c = load_s3_credentials()
    return s3fs.S3FileSystem(key=c["access_key_id"],
                             secret=c["secret_access_key"],
                             token=c.get("session_token"))


def remote_key(rel: str) -> str:
    return f"{S3_BUCKET}/{S3_PREFIX}/{rel}"


def sha256_of(path: Path, buf=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Build the upload list ───────────────────────────────────────────────────
def gather_files(year_filter):
    """Return [(local_path, rel_key), ...] for everything we publish."""
    items = []
    if TFRECORDS_DIR.is_dir():
        pat = f"{year_filter}_*.tfrecords" if year_filter else "*.tfrecords"
        for p in sorted(TFRECORDS_DIR.glob(pat)):
            items.append((p, f"rfe_tfrecords/{p.name}"))
    # constants + norm are tiny; (re)upload them unless filtering a single year
    if not year_filter:
        if CONSTANTS_DIR.is_dir():
            for p in sorted(CONSTANTS_DIR.glob("*.nc")):
                items.append((p, f"cGAN_data/{p.name}"))
        if NORM_PKL.is_file():
            items.append((NORM_PKL, NORM_PKL.name))
    return items


# ─── upload ──────────────────────────────────────────────────────────────────
def cmd_upload(args):
    fs = make_s3fs(anonymous=False)
    items = gather_files(args.year)
    if not items:
        log.error("Nothing to upload. Check TFRECORDS_DIR=%s (year=%s)",
                  TFRECORDS_DIR, args.year)
        sys.exit(1)

    n = len(items)
    total_bytes = sum(p.stat().st_size for p, _ in items)
    log.info("=" * 60)
    log.info("UPLOAD -> s3://%s/%s/", S3_BUCKET, S3_PREFIX)
    log.info("  files: %d   size: %.2f GB   clean-after: %s",
             n, total_bytes / 1e9, args.clean)
    log.info("  credential budget: %d s (refresh .env + re-run to resume)",
             args.credential_timeout)
    log.info("=" * 60)

    manifest = SCRIPT_DIR / "SHA256SUMS.txt"
    done_lines = {}
    if manifest.exists():
        for ln in manifest.read_text().splitlines():
            if ln.strip():
                hsh, name = ln.split(maxsplit=1)
                done_lines[name.strip()] = hsh

    session_start = time.time()
    uploaded = skipped = cleaned = 0
    timed_out = False

    for i, (local, rel) in enumerate(items, 1):
        if time.time() - session_start > args.credential_timeout:
            log.warning("Credential budget reached. Stopping cleanly. "
                        "Refresh .env with fresh STS creds and re-run `upload`.")
            timed_out = True
            break

        key = remote_key(rel)
        lsize = local.stat().st_size

        # Resume: if remote already has it at the same byte size, treat as done.
        already = False
        try:
            info = fs.info(key)
            already = int(info.get("size", -1)) == lsize
        except FileNotFoundError:
            already = False
        except Exception as e:  # transient listing error -> attempt upload
            log.debug("info(%s) failed: %s", key, e)

        if already:
            log.info("[%d/%d] skip (already uploaded, %.1f MB)  %s",
                     i, n, lsize / 1e6, rel)
            skipped += 1
        else:
            if args.dry_run:
                log.info("[%d/%d] DRY-RUN would upload %.1f MB  %s",
                         i, n, lsize / 1e6, rel)
                continue
            t0 = time.time()
            fs.put_file(str(local), key)
            # verify by remote size before we dare delete anything
            rsize = int(fs.info(key)["size"])
            if rsize != lsize:
                log.error("[%d/%d] SIZE MISMATCH %s (local %d != remote %d) "
                          "-- NOT cleaning", i, n, rel, lsize, rsize)
                continue
            rate = lsize / 1e6 / max(time.time() - t0, 1e-3)
            log.info("[%d/%d] uploaded %.1f MB @ %.0f MB/s  %s",
                     i, n, lsize / 1e6, rate, rel)
            uploaded += 1

        # record checksum once (cheap insurance; used by `verify`)
        if rel not in done_lines:
            done_lines[rel] = sha256_of(local)
            with open(manifest, "a") as mf:
                mf.write(f"{done_lines[rel]}  {rel}\n")

        # delete local copy now that it's safely on source.coop
        if args.clean and not args.dry_run:
            try:
                local.unlink()
                cleaned += 1
                log.info("        cleaned local %s", local)
            except OSError as e:
                log.warning("        could not delete %s: %s", local, e)

    # push the manifest itself (best-effort)
    if not args.dry_run and manifest.exists():
        try:
            fs.put_file(str(manifest), remote_key("SHA256SUMS.txt"))
        except Exception as e:
            log.warning("manifest upload failed: %s", e)

    log.info("=" * 60)
    log.info("UPLOAD %s: uploaded=%d skipped=%d cleaned=%d",
             "PAUSED (resume after refreshing creds)" if timed_out else "COMPLETE",
             uploaded, skipped, cleaned)
    log.info("=" * 60)
    sys.exit(2 if timed_out else 0)   # exit 2 => "re-run me with fresh creds"


# ─── verify (anonymous) ──────────────────────────────────────────────────────
def cmd_verify(args):
    fs = make_s3fs(anonymous=True)
    base = f"{S3_BUCKET}/{S3_PREFIX}"
    log.info("VERIFY (anonymous) s3://%s/", base)
    try:
        objs = fs.find(f"{base}/rfe_tfrecords")
    except Exception as e:
        log.error("Cannot list store (%s). Is the repo published/public yet?", e)
        sys.exit(1)
    tfr = [o for o in objs if o.endswith(".tfrecords")]
    total = 0
    bins = {0: 0, 1: 0, 2: 0, 3: 0}
    for o in tfr:
        sz = fs.info(o)["size"]
        total += sz
        # name pattern: <year>_<lead>.<bin>.tfrecords
        try:
            b = int(Path(o).name.split(".")[1])
            bins[b] = bins.get(b, 0) + 1
        except Exception:
            pass
    log.info("  tfrecords objects: %d   total: %.2f GB", len(tfr), total / 1e9)
    log.info("  per class-bin counts: %s", bins)
    if len(tfr) < 16:
        log.warning("  expected >=16 (4 years x 4 bins); found %d", len(tfr))
    for extra in ("FCSTNorm2018.pkl", "cGAN_data/elev.nc", "cGAN_data/lsm.nc",
                  "SHA256SUMS.txt"):
        ok = fs.exists(f"{base}/{extra}")
        log.info("  %-22s %s", extra, "present" if ok else "MISSING")


# ─── download (to GPU local disk) ────────────────────────────────────────────
def cmd_download(args):
    fs = make_s3fs(anonymous=not args.authed)
    base = f"{S3_BUCKET}/{S3_PREFIX}"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("DOWNLOAD s3://%s/  ->  %s  (local NVMe, NOT a network mount!)",
             base, dest)
    keys = fs.find(base)
    if not keys:
        log.error("Nothing found at %s", base)
        sys.exit(1)
    got = 0
    for k in keys:
        rel = k[len(base) + 1:]
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == int(fs.info(k)["size"]):
            continue
        fs.get_file(k, str(out))
        got += 1
        log.info("  pulled %s", rel)
    log.info("DONE: %d new files into %s. Point data_paths.yaml here.", got, dest)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="resumable upload + verify + clean-local")
    up.add_argument("--year", type=str, default=None,
                    help="Only upload this year's tfrecords (e.g. 2018)")
    up.add_argument("--credential-timeout", type=int,
                    default=CREDENTIAL_TIMEOUT_SECONDS,
                    help="Stop before STS token expires (default 2700s = 45 min)")
    up.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Keep local files after upload (default: delete them)")
    up.add_argument("--dry-run", action="store_true")
    up.set_defaults(clean=True)

    sub.add_parser("verify", help="anonymous read-back of the published store")

    dl = sub.add_parser("download", help="pull store to GPU local disk")
    dl.add_argument("--dest", type=str, default="/scratch/CGAN")
    dl.add_argument("--authed", action="store_true",
                    help="Use .env creds (needed if repo not yet public)")

    args = ap.parse_args()
    {"upload": cmd_upload, "verify": cmd_verify,
     "download": cmd_download}[args.command](args)


if __name__ == "__main__":
    main()
