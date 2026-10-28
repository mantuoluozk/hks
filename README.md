# 国芯国模 · BW 模型适配

八个模型的 SCNet BW 单卡 BF16 技术报告、复现代码与原始测试证据。公开自评、功能、性能和官方验收分别记录。

| 模型 | 框架 | 功能 | 公开精度自评 | 首次性能与复现 |
|---|---|---|---|---|
| [Recon2Reason-Reasoning-4B](Recon2Reason-Reasoning-4B/README.md) | SGLang | 5/5 | OCRBench 463/1000；MMMU 437/900 | 12 轮、608 请求 |
| [RoboBrain2.5-8B-NV](RoboBrain2.5-8B-NV/README.md) | SGLang | 8/8 | CV-Bench 2638 题，88.10% | 12 轮、608 请求；完整矩阵独立复跑待补 |
| [MiMo-V2.6-Distill-Qwen-9B](MiMo-V2.6-Distill-Qwen-9B/README.md) | SGLang | 11/11 | GSM8K 179/200；MMLU 160/200 | 12 轮、608 请求 |
| [Qwen3-ASR-1.7B](Qwen3-ASR-1.7B/README.md) | vLLM | 在线 6/6；PCM 4/4 | test-clean 2620 条，WER 1.6501% | 128 请求；复跑最大偏差 7.28% |
| [Qwen3-ForcedAligner-0.6B](Qwen3-ForcedAligner-0.6B/README.md) | vLLM | 中英文 2/2 | 缺匹配公开标注，AAS 暂停；尚无实测成绩 | 每进程 512 请求；复跑最大偏差 7.18% |
| [Trident-Qwen3VL-2B](Trident-Qwen3VL-2B/README.md) | vLLM | 6 条多模态输入及 HTTP embedding | ChartQA Origin NDCG@10 ×100 = 69.4483 | 每进程 1536 请求；复跑最大偏差 3.88% |
| [Qwen-Image-2.1-Turbo](Qwen-Image-2.1-Turbo/README.md) | SGLang | 两个独立进程各 8/8 | CLIP score / FID 待官方评测 | 每进程 9 实测 + 3 预热；复跑最大偏差 17.09% |
| [Kimi-VL-A3B-Instruct](Kimi-VL-A3B-Instruct/README.md) | vLLM | 两个独立进程各8/8 | OCRBench 869/1000；发布者867 | 每进程32请求；吞吐偏差1.32%，全指标最大17.39% |

Recon2Reason 与 RoboBrain 按2026-10-28赛事更新不要求精度，保留既有自测，可准备提交；MiMo最低精度标准改为Qwen3.5-9B，尚未完成同协议对照。Qwen3-ASR 与 Trident 已整理发布者公开基线对照及完整基准实测，提供首版提交材料。其他模型的精度范围见各自报告。比赛验收结论由主办方审核；本仓库不将公开自评或材料提交记为已通过官方验收。

[镜像与复验范围总表](镜像与复验总表.md)列明三套镜像的复用关系、额外依赖及未完成项。

## 复现方式

进入相应模型目录，按技术报告第 4 章执行。默认在 SCNet Notebook 内运行，HKS_ROOT 指向包含 platform/scnet-notebook 的项目根目录；Notebook 已是容器。模型权重与评测数据不随代码分发，从模型报告标注的来源获取并按固定清单核验。

各报告统一为简介、环境、适配与配置、启动与复现、功能测试、精度测评、首次性能、复现与限制八章。各模型吞吐的输入、批次和计时范围不同，不直接横向比较。
