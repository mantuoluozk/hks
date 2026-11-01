#!/usr/bin/env python3
"""Apply the Kimi-Linear vLLM 0.25.1 HCU runtime fix in an isolated image.

This script changes only three installed Python source files. It is idempotent,
checks exact source fragments, and saves originals beside the installed files.
Run before starting vLLM, never while a vLLM process is using the packages.
"""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path


CHANGES = {
    "vllm_hcu/models/kimi_k3/amd/kimi_gdn_linear_attn.py": [
        (
            '                        x.transpose(0, 1),\n                        weight,\n                        None,\n                        activation="silu",\n                        conv_states=state,',
            '                        x.transpose(0, 1).contiguous(),\n                        weight,\n                        None,\n                        activation="silu",\n                        conv_states=state,',
            1,
        ),
    ],
    "vllm_hcu/models/kimi_k3/amd/ops/third_party/kda/chunk.py": [
        ("        safe_gate=lower_bound is not None,", "        safe_gate=True,", 1),
    ],
    "boltops/fla/kda/triton/chunk_intra.py": [
        (
            "    b_gm = (b_g - b_gn).to(tl.float32)",
            "    b_gm = tl.clamp((b_g - b_gn).to(tl.float32), -80.0, 80.0)",
            2,
        ),
        (
            "    use_poly_solve = K == 128 and BT == 64 and BC == 16",
            "    use_poly_solve = False",
            1,
        ),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--check", action="store_true", help="Check that the patch is present without writing")
    args = parser.parse_args()
    updates: list[tuple[Path, str, str]] = []
    for relative, changes in CHANGES.items():
        target = args.site_packages / relative
        original = target.read_text()
        updated = original
        for before, after, count in changes:
            if updated.count(after) == count and updated.count(before) == 0:
                continue
            if updated.count(before) != count or updated.count(after) != 0:
                raise RuntimeError(f"Unexpected source in {target}: {before[:60]!r}")
            updated = updated.replace(before, after)
        if args.check and updated != original:
            raise RuntimeError(f"Patch missing: {target}")
        updates.append((target, original, updated))
    if not args.check:
        for target, original, updated in updates:
            if updated == original:
                print(f"already patched: {target}")
                continue
            backup = target.with_name(target.name + ".kimi-linear-original")
            if backup.exists():
                raise RuntimeError(f"Backup already exists; inspect before replacing: {backup}")
            backup.write_text(original)
            target.write_text(updated)
            print(f"patched: {target}; original: {backup}")
    else:
        for target, _, _ in updates:
            print(f"patch present: {target}")


if __name__ == "__main__":
    main()
