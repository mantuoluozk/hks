# MMLU-Pro 测试准备

当前仅完成脚本、完整数据预处理和CPU检查，未启动MiMo推理，没有新的精度成绩。原GSM8K/普通MMLU结果保留。

选择发布者有公开成绩的一个数据集：**MMLU-Pro完整test 12,032题**。参考为[Qwen3.5-9B固定版本模型卡](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)的**82.5%**，按“不低于”比较，不增加2%容差，也不先四舍五入再判定。12,032题至少需正确9,927题才能达到该数值。

## 已准备内容

- 数据：`TIGER-Lab/MMLU-Pro`固定版本`b189ec765aa7ed75c8acfea42df31fdae71f97be`；两个Parquet约4.19MB，已校验发布者SHA256。完整test 12,032题、validation 70题，14个领域各5个CoT示例。
- 提示：按[MMLU-Pro官方实现](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_api.py)，固定同领域5-shot CoT、原顺序、原选项及`N/A`清理，最终格式为`The answer is (X)`。CPU检查直接调用官方函数核对提示一致。
- 计分：只解析最终`content`，不从`reasoning_content`猜答案。沿用官方提取规则，不随机回退；请求错误、解析失败及截断保留并计错，不删题、不根据对错重试。
- 采样：采用Qwen模型卡通用Thinking建议：temperature=1、top_p=0.95、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1，输出上限32768，seed42，并发1。
- 上下文：计划49152，包含输入与输出预算。实际执行前用本地原始tokenizer计算输入长度，并核对SGLang实际上下文；预算不足会拒绝执行。这个启动配置尚未进行GPU实测。
- 输出：原始完整响应、每题预测/错误/截断、总体及分领域分数、缺失数和配置指纹。断点续跑只处理未记录题目，配置变化拒绝混用。
- 调度：默认命令只展示计划，不发送推理请求；显式`--run`才执行。启动脚本和评测入口均会阻止在AuK生成/转写期间运行MiMo。

Qwen模型卡没有提供足以独立核实其82.5%成绩的全部逐项评测设置。因此上述设置是“官方MMLU-Pro提示 + Qwen公开通用建议”，**不宣称与该成绩的完整协议已完全相同**。汇总中`numeric_threshold_met`仅表示完整、零请求错误结果是否达到数值门槛；正式验收由主办方审核。

## 后续运行命令

在模型目录`/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B`执行。当前不执行启动或推理命令。

```bash
# CPU检查与数据准备；数据已位于data/mmlu-pro
python scripts/test_mmlu_pro.py
python scripts/prepare_mmlu_pro.py
python scripts/mmlu_pro_eval.py             # 只显示计划，无推理

# 待AuK任务完成且确认切换MiMo后，在终端1启动服务
bash scripts/run_mmlu_pro_server.sh

# 终端2先用8题检查链路、输出截断和耗时；不是达标成绩
bash run-scnet.sh python scripts/mmlu_pro_eval.py --run --limit 8 \
  --output "results/mmlu-pro/pilot-$(date -u +%Y%m%dT%H%M%SZ)"

# 完整test，使用新的输出目录
bash run-scnet.sh python scripts/mmlu_pro_eval.py --run \
  --output "results/mmlu-pro/full-$(date -u +%Y%m%dT%H%M%SZ)"

# 如中断，使用原目录和完全相同配置续跑
# bash run-scnet.sh python scripts/mmlu_pro_eval.py --run --resume --output <原目录>
```

CPU检查覆盖官方提示一致性、Unicode JSONL、错误分母、最终答案解析、截断计错、门槛边界、断点指纹及AuK并行保护。结果见`results/mmlu-pro-preparation/`；这些检查不是MiMo模型实测。
