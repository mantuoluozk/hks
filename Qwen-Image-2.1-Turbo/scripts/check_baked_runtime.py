#!/usr/bin/env python3
"""Verify baked image files and installed wheel RECORD hashes (stdlib only)."""
import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

MOVE = "runtime/diffusers-py310/"


class Fail(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def rel(name, wheel=False):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise Fail(f"invalid relative path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise Fail(f"path escape rejected: {name}")
    if wheel and path.parts[0].endswith(".data"):
        raise Fail(f"unsupported wheel .data relocation: {name}")
    return path


def inside(base, name):
    path = (base / Path(*rel(name).parts)).resolve()
    if not path.is_relative_to(base.resolve()):
        raise Fail(f"path escape rejected: {name}")
    return path


def digest(value):
    if isinstance(value, dict):
        value = value.get("sha256") or value.get("hash") or value.get("digest")
    if not isinstance(value, str):
        raise Fail("invalid manifest digest")
    value = value.strip().lower()
    if value.startswith(("sha256:", "sha256=")):
        value = value[7:]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Fail(f"invalid SHA256 digest: {value!r}")
    return value


def manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("files"), (dict, list)):
        data = data["files"]
    if isinstance(data, dict):
        return list(data.items())
    if isinstance(data, list):
        entries = []
        for item in data:
            if not isinstance(item, dict):
                raise Fail("invalid SHA256.json entry")
            name = item.get("path") or item.get("file")
            if not isinstance(name, str):
                raise Fail("invalid SHA256.json path")
            entries.append((name, item))
        return entries
    raise Fail("SHA256.json must be an object or list")


def image_file(root, name):
    name = rel(name).as_posix()
    if name.startswith(MOVE):
        name = "diffusers-py310/" + name[len(MOVE):]
    return inside(root, name)


def check_manifest(root, result):
    for name, value in manifest(root / "SHA256.json"):
        path = image_file(root, name)
        if not path.is_file():
            raise Fail(f"missing manifest file: {name}")
        if sha256(path).hex() != digest(value):
            raise Fail(f"SHA256 mismatch: {name}")
        result["manifest_files_checked"] += 1


def check_wheel(wheel, dest, base, result):
    with zipfile.ZipFile(wheel) as archive:
        records = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise Fail(f"{wheel.name}: expected one .dist-info/RECORD")
        with archive.open(records[0]) as raw:
            rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for row in rows:
                if len(row) != 3:
                    raise Fail(f"{wheel.name}: malformed RECORD row")
                name, recorded = row[0], row[1]
                safe = rel(name, wheel=True)
                if not recorded:
                    result["record_entries_without_hash"] += 1
                    continue
                if not recorded.startswith("sha256="):
                    raise Fail(f"unsupported RECORD hash: {name}")
                encoded = recorded[7:]
                expected = base64.urlsafe_b64decode(
                    encoded + "=" * (-len(encoded) % 4)
                )
                path = (dest / Path(*safe.parts)).resolve()
                if not path.is_relative_to(base):
                    raise Fail(f"path escape rejected: {name}")
                if not path.is_file():
                    raise Fail(f"missing installed file: {name}")
                if sha256(path) != expected:
                    raise Fail(f"installed SHA256 mismatch: {name}")
                result["installed_files_checked"] += 1


def check_wheels(directory, count, dest, result):
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != count:
        raise Fail(f"{directory}: expected {count} wheels, found {len(wheels)}")
    base = dest.resolve()
    for wheel in wheels:
        check_wheel(wheel, dest, base, result)
        result["wheels_checked"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/opt/qwen-image21")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    result = {
        "manifest_files_checked": 0,
        "wheels_checked": 0,
        "installed_files_checked": 0,
        "record_entries_without_hash": 0,
        "pass": False,
        "error": None,
    }
    code = 0
    try:
        check_manifest(root, result)
        check_wheels(root / "runtime" / "wheels", 2, root / "site-packages", result)
        check_wheels(
            root / "base-extra" / "wheels", 8, Path("/opt/qwen3-asr-deps"), result
        )
        result["pass"] = True
    except Exception as exc:
        result["error"] = str(exc) if isinstance(exc, Fail) else f"{type(exc).__name__}: {exc}"
        code = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
