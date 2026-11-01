# GSM8K 200 题固定样本：来源与重建

本仓库**不包含数据集文件**（评测数据与权重不随代码分发）。GSM8K 评测脚本需要你自己提供这份固定的 200 题文件。

## 文件是什么

- 文件名：`gsm8k.jsonl`
- 行数：**200**（严格校验，数量不符脚本会拒绝运行）
- 每行字段：`{"question": "...", "answer": "<数值字符串>"}`
- SHA256：`03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546`

该样本集与同批 MiMo-V2.6-Distill-Qwen-9B 评测使用同一份文件，因此两个模型的自评口径一致。

## 怎么提供

三种方式任选其一，优先级从高到低：

```bash
# 1) 命令行显式指定
python scripts/eval_gsm8k_vllm.py --output results/accuracy-gsm8k-200 --data /path/to/gsm8k.jsonl

# 2) 环境变量
export KIMI_GSM8K_PATH=/path/to/gsm8k.jsonl

# 3) 约定路径（脚本会自动按序查找）
#    ./data/gsm8k.jsonl
#    /root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B/data/gsm8k.jsonl
```

脚本启动时会计算 SHA256 并与上面的固定值比对：

- 一致 → 正常评测
- 不一致 → 默认**拒绝运行**（避免用错文件却得到一个看似正常的分数）
- 确知是有意替换 → 加 `--allow-sha-mismatch`，脚本会把 `dataset_sha_matches_pinned: false` 写进 `summary.json`，便于评审识别

## 从原始 GSM8K 重建固定 200 题

原始数据集为 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)（`main` split，即 test 集，共 1319 题）。本批固定 200 题是从中按固定随机种子抽取的稳定子集，抽取规则与同批 MiMo 评测一致（seed `20261023`），以保证两个模型用完全相同的 200 题。

重建步骤（需能访问原始数据集）：

```bash
python scripts/rebuild_gsm8k_200.py \
  --source /path/to/openai_gsm8k_main_test.parquet \
  --output data/gsm8k.jsonl \
  --seed 20261023
```

脚本会打印最终 SHA256，应与上面固定值一致。如果不一致，说明抽取规则或源数据版本与原始批次不同，此时**不要**用这份文件与已发布结果对比。

## 关于答案字段

原始 GSM8K 的 `answer` 字段是完整解题过程，格式为 `... #### 18`。本固定样本的 `answer` 字段只保留 `####` 之后的**最终数值字符串**，例如 `18`、`3,430`、`205`。评分时服务输出会用同样的规则抽取数值后做 Decimal 比较，容忍 `$`、千分位与加粗标记。
