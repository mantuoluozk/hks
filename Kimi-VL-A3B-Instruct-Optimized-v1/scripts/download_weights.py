"""Download pinned publisher files directly on the server, resume and verify.

No local relay, duplicate HF cache, model execution or global configuration.
"""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "weights"
CACHE = ROOT / "cache" / "download-20261028"
MANIFEST = json.loads((ROOT / "sources/publisher-files.json").read_text())
REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
assert MANIFEST["sha"] == REVISION
FILES = [f for f in MANIFEST["siblings"]
         if "/" not in f["rfilename"] and not f["rfilename"].startswith(".")]
ENDPOINT = "https://hf-mirror.com"
verified = {}
lock = threading.Lock()


def check(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError("size mismatch")
    h = hashlib.sha256() if entry.get("lfs") else hashlib.sha1()
    if not entry.get("lfs"):
        h.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    expected = entry["lfs"]["sha256"] if entry.get("lfs") else entry["blobId"]
    if h.hexdigest() != expected:
        raise ValueError("publisher hash mismatch")
    return {"bytes": entry["size"], "hash": h.hexdigest(),
            "algorithm": "sha256" if entry.get("lfs") else "git-blob-sha1"}


def download(entry):
    name = entry["rfilename"]
    path = DEST / name
    partial = DEST / (name + ".partial")
    url = f"{ENDPOINT}/{MANIFEST['id']}/resolve/{REVISION}/{quote(name)}?download=true"
    for attempt in range(8):
        try:
            if path.exists():
                proof = check(path, entry)
            else:
                size = partial.stat().st_size if partial.exists() else 0
                if size > entry["size"]:
                    raise ValueError("partial too large")
                if size != entry["size"]:
                    headers = {"Accept-Encoding": "identity"}
                    if size:
                        headers["Range"] = f"bytes={size}-"
                    with requests.get(url, headers=headers, stream=True,
                                      timeout=(15, 60)) as response:
                        if response.status_code not in (200, 206):
                            raise RuntimeError(f"HTTP {response.status_code}")
                        if response.status_code == 206:
                            expected_range = f"bytes {size}-"
                            if not response.headers.get("Content-Range", "").startswith(expected_range):
                                raise ValueError("incorrect resume offset")
                        else:
                            size = 0  # A server ignoring Range must never be appended.
                        with partial.open("ab" if size else "wb") as stream:
                            for chunk in response.iter_content(64 * 1024):
                                if not chunk:
                                    continue
                                if size + len(chunk) > entry["size"]:
                                    raise ValueError("response larger than expected file")
                                stream.write(chunk)
                                size += len(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                proof = check(partial, entry)
                partial.rename(path)
            with lock:
                verified[name] = proof
            print(json.dumps({"verified": name, **proof}), flush=True)
            return
        except Exception as error:
            # Avoid persisting signed redirect URLs or other HTTP internals.
            reason = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            print(json.dumps({"file": name, "attempt": attempt + 1, "error": reason}), flush=True)
            if isinstance(error, ValueError):
                bad = path if path.exists() else partial
                if bad.exists():
                    bad.rename(bad.with_name(bad.name + f".invalid-{time.time_ns()}"))
                raise RuntimeError(f"Integrity failure: {name} ({reason}); preserved invalid file") from None
            if attempt == 7:
                raise RuntimeError(f"Download failed: {name} ({reason})") from None
            time.sleep(min(30, 2 ** attempt))


def progress(state, started):
    sizes = {}
    for entry in FILES:
        path = DEST / entry["rfilename"]
        partial = path.with_name(path.name + ".partial")
        size = 0
        for candidate in (partial, path):
            try:
                size = max(size, candidate.stat().st_size)
            except FileNotFoundError:
                pass
        sizes[path.name] = size
    with lock:
        checked = dict(verified)
    record = {"state": state, "pid": os.getpid(), "revision": REVISION,
              "destination": str(DEST), "elapsed_seconds": time.monotonic() - started,
              "downloaded_bytes": sum(sizes.values()),
              "expected_bytes": sum(f["size"] for f in FILES),
              "verified_files": len(checked), "total_files": len(FILES), "file_bytes": sizes,
              "publisher_hash_checks": checked}
    tmp = CACHE / "status.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(CACHE / "status.json")
    return record


def main():
    assert Path("/root/private_data").is_mount(), "Shared storage is not mounted"
    CACHE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    with (CACHE / "download.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        missing = sum(max(0, f["size"] - max(
            (DEST/f["rfilename"]).stat().st_size if (DEST/f["rfilename"]).exists() else 0,
            (DEST/(f["rfilename"]+".partial")).stat().st_size if (DEST/(f["rfilename"]+".partial")).exists() else 0)) for f in FILES)
        assert shutil.disk_usage(DEST).free > missing + 5 * 1024**3, "Insufficient storage"
        started = time.monotonic()
        failed = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(download, entry) for entry in FILES]
            while not all(f.done() for f in futures):
                progress("downloading", started)
                time.sleep(10)
            for future in futures:
                if future.exception() is not None:
                    failed = True
            result = progress("failed" if failed else "complete_verified", started)
        if failed:
            raise SystemExit(1)
        output = ROOT / "results" / "weights-download-20261028.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        with (CACHE / "backup.log").open("w") as log:
            subprocess.run(["python", str(ROOT.parents[1]/"platform/scnet-notebook/backup_shared.py")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
