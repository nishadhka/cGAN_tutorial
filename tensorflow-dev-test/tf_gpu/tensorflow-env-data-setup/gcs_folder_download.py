
#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import os
from pathlib import Path
from typing import Tuple

from google.api_core.retry import Retry
from google.cloud import storage
"""
python gcs_folder_download.py \
  gs://geosfm/geosfm_input_icpac_pc/20250818 \
  --creds /path/to/your-service-account.json \
  --dest /path/to/download/here \
  --skip-existing \
  --workers 16
"""

def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI like gs://bucket/path/to/prefix into (bucket, prefix).
    Ensures the returned prefix always ends with a single '/' (unless empty).
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    remainder = gcs_uri[5:]
    parts = remainder.split("/", 1)
    bucket = parts[0]
    prefix = ""
    if len(parts) == 2:
        prefix = parts[1].strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def should_skip(local_path: Path, blob) -> bool:
    """
    Skip download if a local file exists with the same size.
    """
    try:
        if local_path.exists() and local_path.is_file():
            return local_path.stat().st_size == blob.size
    except Exception:
        pass
    return False


def download_blob(args):
    (blob, base_prefix, dest_dir, skip_existing, chunk_size) = args
    # GCS can have "directory marker" objects ending with '/'
    if blob.name.endswith("/"):
        return f"DIR  : {blob.name} (skipped marker)"

    rel = blob.name[len(base_prefix):].lstrip("/")
    local_path = Path(dest_dir) / rel
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and should_skip(local_path, blob):
        return f"SKIP : {rel} (exists, same size)"

    # Optional: set a chunk size for large files (e.g., 8 MiB)
    if chunk_size:
        blob._chunk_size = chunk_size  # pylint: disable=protected-access

    # Robust retries on transient errors
    retry = Retry(initial=1.0, maximum=30.0, multiplier=2.0, deadline=300.0)
    blob.download_to_filename(str(local_path), retry=retry)
    return f"OK   : {rel}"


def main():
    p = argparse.ArgumentParser(description="Download a GCS prefix (folder) recursively.")
    p.add_argument("gcs_uri", help="GCS URI, e.g., gs://bucket/path/to/folder")
    p.add_argument(
        "--creds",
        required=True,
        help="Path to JSON service account key file (e.g., service-account.json)",
    )
    p.add_argument(
        "--dest",
        default=".",
        help="Local destination directory (default: current directory)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 2) * 4),
        help="Number of parallel downloads (default: a sensible value)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally with the same size",
    )
    p.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Download chunk size in MiB (helpful for large files). Set 0 to use default.",
    )
    args = p.parse_args()

    bucket_name, prefix = parse_gcs_uri(args.gcs_uri)
    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client.from_service_account_json(args.creds)
    bucket = client.bucket(bucket_name)

    # List all blobs with the prefix
    print(f"Listing objects in gs://{bucket_name}/{prefix}")
    blobs_iter = bucket.list_blobs(prefix=prefix)

    # Materialize the list so we can show counts and parallelize
    blobs = list(blobs_iter)
    if not blobs:
        print("No objects found for given prefix.")
        return

    print(f"Found {len(blobs)} objects. Starting download to {dest_dir} ...")

    work = []
    chunk_size = args.chunk_size_mb * 1024 * 1024 if args.chunk_size_mb > 0 else None
    for b in blobs:
        work.append((b, prefix, dest_dir, args.skip_existing, chunk_size))

    completed = 0
    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for result in ex.map(download_blob, work, chunksize=10):
            if result.startswith("OK"):
                completed += 1
            elif result.startswith("SKIP"):
                print(result)
            elif result.startswith("DIR"):
                pass
            else:
                errors += 1
            # Print minimal progress lines
            if result.startswith(("OK", "SKIP")):
                print(result)

    print(f"\nDone. Successful: {completed}, Errors: {errors}, Total listed: {len(blobs)}")


if __name__ == "__main__":
    main()
