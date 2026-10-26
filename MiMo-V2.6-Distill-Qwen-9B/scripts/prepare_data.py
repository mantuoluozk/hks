#!/usr/bin/env python3
"""把 ModelScope 下载的 GSM8K / MMLU parquet 转成 verify.py 需要的 jsonl 子集。

官方「分题材料」尚未下发时，用它自建可复现的本地评测子集：
- 固定 seed 抽样，题数写死在 baselines/<模型>.json 里
- 生成的 data/*.jsonl 在整个验证过程中不再变动（与官方「不得换子集」要求一致）
"""
import argparse
import json
import os
import random
import sys

import pandas as pd


def build_gsm8k(src_dir, out_path, n, seed):
    pq = os.path.join(src_dir, "main", "test-00000-of-00001.parquet")
    if not os.path.exists(pq):
        # 兼容 modelscope 直接落在根目录的情况
        pq = os.path.join(src_dir, "test-00000-of-00001.parquet")
    if not os.path.exists(pq):
        sys.exit(f"找不到 GSM8K parquet: {pq}")
    df = pd.read_parquet(pq)
    # GSM8K 答案形如 "....#### 42"，取 #### 之后的数值
    items = []
    for _, r in df.iterrows():
        ans = str(r["answer"])
        if "####" not in ans:
            continue
        items.append(
            {
                "question": str(r["question"]).strip(),
                "answer": ans.split("####")[-1].strip().replace(",", ""),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n]
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"gsm8k: {len(items)} 题 -> {out_path}")


def build_mmlu(src_dir, out_path, n, seed):
    # modelscope/MMLU 的 data.tar 解压后是 data/test/*.csv（部分科目落在 data/val/）
    candidates = [
        os.path.join(src_dir, "data", "test"),
        os.path.join(src_dir, "test"),
        os.path.join(src_dir, "data", "val"),
        os.path.join(src_dir, "val"),
        src_dir,
    ]
    csv_dir = next(
        (
            d
            for d in candidates
            if os.path.isdir(d) and any(x.endswith(".csv") for x in os.listdir(d))
        ),
        None,
    )
    if csv_dir is None:
        sys.exit(f"找不到 MMLU CSV，已尝试: {candidates}")
    csvs = sorted(
        os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith(".csv")
    )
    if not csvs:
        sys.exit(f"找不到 MMLU test CSV: {csv_dir}")
    items = []
    for path in csvs:
        df = pd.read_csv(path, header=None)
        for _, r in df.iterrows():
            items.append(
                {
                    "question": str(r[0]).strip(),
                    "choices": [str(r[1]).strip(), str(r[2]).strip(), str(r[3]).strip(), str(r[4]).strip()],
                    "answer": str(r[5]).strip(),
                }
            )
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n]
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"mmlu: {len(items)} 题 -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k-dir", default="cache/datasets/gsm8k")
    ap.add_argument("--mmlu-dir", default="cache/datasets/mmlu")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--mmlu-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    build_gsm8k(a.gsm8k_dir, os.path.join(a.out_dir, "gsm8k.jsonl"), a.gsm8k_n, a.seed)
    build_mmlu(a.mmlu_dir, os.path.join(a.out_dir, "mmlu.jsonl"), a.mmlu_n, a.seed)
