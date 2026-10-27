# 国芯国模黑客松

## 模型目录

| 模型 | 环境 | 技术报告 | 功能与性能 | 精度实测 |
|---|---|---|---|---|
| Recon2Reason-Reasoning-4B | BW × 1 / SGLang 0.5.18 / BF16 | [README](Recon2Reason-Reasoning-4B/README.md) | 5/5 Smoke；12 轮、608 请求完成 | OCRBench 463/1000；MMMU 437/900 |
| RoboBrain2.5-8B-NV | BW × 1 / SGLang 0.5.18 / BF16 | [README](RoboBrain2.5-8B-NV/README.md) | 8/8 Smoke；12 轮、608 请求完成 | 公开 CV-Bench 全量 88.10（参考 94.58，差 -6.48 pt）；官方 GPU 动作/坐标一致率单列未判定 |
| MiMo-V2.6-Distill-Qwen-9B | BW × 1 / SGLang 0.5.18 / BF16 | [README](MiMo-V2.6-Distill-Qwen-9B/README.md) | 11/11 Smoke；12轮608请求 | 固定公开样本GSM8K 179/200、MMLU 160/200；正式精度待官方代码复验 |
| Qwen3-ASR-1.7B | BW × 1 / vLLM 0.25.1 / BF16 | [README](Qwen3-ASR-1.7B/README.md) | 在线6/6、PCM4/4；性能8项128请求；镜像复验最大偏差7.28% | 完整 test-clean 2620条 WER 1.6501%，零错误零截断；正式精度待官方代码 |
| Qwen3-ForcedAligner-0.6B | BW × 1 / vLLM 0.25.1 / BF16 | [README](Qwen3-ForcedAligner-0.6B/README.md) | 中英文2/2；每进程8项512请求；独立进程复现最大偏差7.18% | 正式AAS待官方代码；批次8存在80ms边界差异 |
| Trident-Qwen3VL-2B | BW × 1 / vLLM 0.25.1 / BF16 | [README](Trident-Qwen3VL-2B/README.md) | 多模态6条及HTTP embedding通过；每进程24格1536请求；复跑偏差3.88% | 完整ChartQA Origin 1250查询/1509图表，NDCG@10 69.4483（论文70.19）；正式精度待官方验收 |
| Qwen-Image-2.1-Turbo | BW × 1 / SGLang 0.5.18 generic Diffusers / BF16 | [README](Qwen-Image-2.1-Turbo/README.md) | 专用镜像1.0.0新Notebook复验；每进程8项功能；三档性能复跑最大偏差17.09% | 正式CLIP/FID及GPU参考待官方评测代码 |

每个模型的配置、启动脚本、精度与性能结果位于对应目录，报告按比赛八章模板组织。

首版作为性能基准；后续优化采用模型专属 tag，保持卡型、框架和负载一致。

Qwen3-ASR 与 Qwen-Image 专用镜像保持私有，固定摘要待正式提交前补齐。
