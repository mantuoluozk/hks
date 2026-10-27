import json
import math
from pathlib import Path


def evaluate_image(root, validation, args):
    keys = ("image_preflight", "image_install_check", "image_info")
    if all(args[key] is None for key in keys):
        return {"dedicated_image_revalidation": False}
    if any(args[key] is None for key in keys):
        raise ValueError("image revalidation needs all three paths")

    rel, docs = {}, {}
    for key in keys:
        path = (root / Path(args[key])).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{key} is outside root")
        rel[key] = path.relative_to(root).as_posix()
        docs[key] = json.loads(path.read_text(encoding="utf-8"))

    preflight = docs["image_preflight"]
    installed = docs["image_install_check"]
    info = docs["image_info"]
    launcher = preflight.get("launcher_integrity") or {}
    if (preflight.get("baked_runtime") is not True
            or preflight.get("runtime_root") != "/opt/qwen-image21"
            or not launcher.get("image_sha256")
            or launcher.get("image_sha256") != launcher.get("executed_source_sha256")
            or installed.get("pass") is not True
            or installed.get("wheels_checked") != 10
            or installed.get("manifest_files_checked") != 1093
            or any(not info.get(k) for k in ("image", "version", "image_page", "visibility", "region", "instance"))
            or "digest" not in info):
        raise ValueError("image environment check failed")

    status = json.loads((validation / "status.json").read_text(encoding="utf-8"))
    summaries = [
        json.loads((validation / f"round-{n}/smoke/summary.json").read_text(encoding="utf-8"))
        for n in (1, 2)
    ]

    def smoke_ok(data):
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list) or len(cases) != 8:
            return False
        by_name = {case.get("case"): case for case in cases if isinstance(case, dict)}
        return (all(isinstance(c, dict) and c.get("http_status") == 200 for c in cases)
                and (by_name.get("text512-repeat") or {}).get("same_seed_pixels_identical") is True
                and (by_name.get("text512-otherseed") or {}).get("different_seed_pixels_differ") is True)

    def perf_ok(rows):
        if not isinstance(rows, list) or len(rows) != 3:
            return False
        if {row.get("size") for row in rows if isinstance(row, dict)} != {512, 1024, 2048}:
            return False
        for row in rows:
            value = row.get("relative_difference_percent") if isinstance(row, dict) else None
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 20):
                return False
        return True

    rounds = status.get("rounds") if isinstance(status, dict) else []
    gate = (isinstance(status, dict) and status.get("status") == "completed"
            and isinstance(rounds, list) and len(rounds) == 2
            and all(isinstance(r, dict) and r.get("functional_cases") == 8 for r in rounds)
            and smoke_ok(summaries[0]) and smoke_ok(summaries[1])
            and perf_ok(status.get("performance_comparison")))
    return {
        "dedicated_image_revalidation": gate,
        "image_preflight": rel["image_preflight"],
        "image_install_check": rel["image_install_check"],
        "image_info": rel["image_info"],
    }
