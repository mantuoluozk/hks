# 国芯国模 · BW 模型适配

十二个模型、十五个模型×框架版本的 SCNet BW 单卡/多卡技术报告、复现代码与原始测试证据。公开自评、功能、性能和官方验收分别记录。

2026-11-02 校验入口维护：18个已发布目录的默认证据校验入口、共32次CPU命令检查正常退出；Qwen-Image与MobileVLA报告清单已同步。校验成功返回0不代表赛事验收通过，详见[校验说明](校验入口维护-20261102.md)。

| 模型 | 框架 | 功能 | 公开精度自评 | 首次性能与复现 |
|---|---|---|---|---|
| [Recon2Reason-Reasoning-4B](Recon2Reason-Reasoning-4B/README.md) | SGLang | 5/5 | OCRBench 463/1000；MMMU 437/900 | 12 轮、608 请求 |
| [Recon2Reason-Reasoning-4B-vLLM](Recon2Reason-Reasoning-4B-vLLM/README.md) | vLLM | 新镜像两独立服务各5/5 | 赛事不要求精度 | 官方六档完整双进程1216请求；模板指标最大偏差9.65%；特殊token输出说明见报告 |
| [Recon2Reason-Reasoning-4B-vLLM · 优化版v1](Recon2Reason-Reasoning-4B-vLLM-Optimized-v1/README.md) | vLLM | 四服务各5/5 | 赛事不要求精度；短测50条请求/usage/输出一致 | 官方六档608/608；完整矩阵吞吐提升2.21%–347.89%，五档超过10%；单矩阵服务 |
| [RoboBrain2.5-8B-NV](RoboBrain2.5-8B-NV/README.md) | SGLang | 8/8 | CV-Bench 2638 题，88.10% | 12 轮、608 请求；完整矩阵独立复跑待补 |
| [RoboBrain2.5-8B-NV-vLLM](RoboBrain2.5-8B-NV-vLLM/README.md) | vLLM | 两个服务各8/8，功能文本逐项一致 | 赛事口径不要求额外精度；本轮未测完整基准 | 官方六档608/608，实际输出622592 token；单矩阵服务，独立进程完整复现null |
| [MiMo-V2.6-Distill-Qwen-9B](MiMo-V2.6-Distill-Qwen-9B/README.md) | SGLang | 11/11 | GSM8K 179/200；MMLU 160/200 | 12 轮、608 请求 |
| [Qwen3-ASR-1.7B](Qwen3-ASR-1.7B/README.md) | vLLM | 在线 6/6；PCM 4/4；官方条件识别 300/300 | test-clean 2620 条，WER 1.6501% | 128 请求；复跑最大偏差 7.28% |
| [Qwen3-ForcedAligner-0.6B](Qwen3-ForcedAligner-0.6B/README.md) | vLLM | 中英文 2/2；官方条件词级对齐 30/30 | — | 每进程 512 请求；复跑最大偏差 7.18% |
| [Qwen3-ForcedAligner-0.6B 优化版v1](Qwen3-ForcedAligner-0.6B-Optimized-v1/README.md) | vLLM | 中英文2/2；最终各批次与本轮单条一致 | 与首版历史边界最大差80ms；非AAS精度结论 | 512条对齐输入；相对首版历史吞吐提升17.77%–97.20%；单进程两轮 |
| [Trident-Qwen3VL-2B](Trident-Qwen3VL-2B/README.md) | vLLM | 6 条多模态输入及 HTTP embedding | ChartQA Origin NDCG@10 ×100 = 69.4483 | 每进程 1536 请求；复跑最大偏差 3.88% |
| [Trident-Qwen3VL-2B 优化版v1](Trident-Qwen3VL-2B-Optimized-v1/README.md) | vLLM | 6条多模态输入及HTTP embedding通过 | 首版ChartQA Origin 69.4483；优化版演示向量对照通过 | 1536条embedding输入；相对初版历史吞吐提升21.14%–346.72%；单进程两轮 |
| [Qwen-Image-2.1-Turbo](Qwen-Image-2.1-Turbo/README.md) | SGLang | 两个独立进程各 8/8；T03/T04 固定条件生成 180/180 | T03/T04 GenEval 图片已生成；未计算 CLIP/FID | 每进程 9 实测 + 3 预热；复跑最大偏差 17.09% |
| [Kimi-VL-A3B-Instruct](Kimi-VL-A3B-Instruct/README.md) | vLLM | 两个独立进程各8/8 | OCRBench 869/1000；发布者867 | 每进程32请求；吞吐偏差1.32%，全指标最大17.39% |
| [Kimi-VL-A3B-Instruct · 优化版v1](Kimi-VL-A3B-Instruct-Optimized-v1/README.md) | vLLM | 优化短负载与正式服务各8/8 | 首版OCRBench 869/1000；优化对照32条输出文本及长度逐条一致 | 官方六档608/608；同简单负载吞吐提升195.20%–384.94% |
| [AuK](AuK/README.md) | SGLang-Omni | 两独立服务各9/9；新镜像中文HTTP通过 | Seed-TTS-Eval英文1088条，WER 1.7137%；较发布者增加0.6937个百分点 | 每进程32实测+6预热；均值延迟/音频吞吐偏差1.1583% |
| [MobileVLA-R1](MobileVLA-R1/README.md) | RGB前端 + vLLM语言后端 | 两独立服务各12/12；两条单图参考token一致 | 未测（闭环导航需 Habitat 环境） | 每进程9实测+3预热；均值延迟偏差4.64%；官方六档矩阵未执行（见报告7.2） |
| [MiniCPM-V-4.7-35B-A3B](MiniCPM-V-4.7-35B-A3B/README.md) | vLLM，BW×2 TP2 | 两独立服务各8/8 | OCRBench 888/1000，88.80%；发布者16x/instruct90.40%，下降1.60 pt≤2 pt | 六格复跑最大偏差10.31%；同Notebook独立进程 |
| [Kimi-Linear-48B-A3B-Instruct-SGLang](Kimi-Linear-48B-A3B-Instruct-SGLang/README.md) | SGLang | 两独立服务各6/6 | GSM8K 185/200（92.50%） | 6档 ×2轮，最大偏差12.69% |
| [Kimi-Linear-48B-A3B-Instruct-vLLM](Kimi-Linear-48B-A3B-Instruct-vLLM/README.md) | vLLM | 历史两独立服务各6/6；发布入口 CPU 回归通过 | GSM8K 183/200（91.5%） | 历史6档 ×2轮（均含预热），最大偏差14.30%；官方矩阵待补 |

AuK T03/T04固定条件补测：120/120 条生成完成，交付ZIP及逐项反馈见 [补测材料](AuK/results/official-t03t04-20261030/README.md)。本次未计算质量分数。

Qwen3-ForcedAligner T03/T04固定条件：英文30条词级时间戳输出全部完成。材料见 [补测记录](Qwen3-ForcedAligner-0.6B/results/official-t03t04-20261030/README.md)。

Qwen3-ASR T03/T04固定条件：AISHELL-1、LibriSpeech test-clean、test-other 各100条，共300/300识别完成。材料见 [补测记录](Qwen3-ASR-1.7B/results/official-t03t04-20261030/README.md)。

Qwen-Image-2.1-Turbo T03/T04固定条件：GenEval 60条 prompt、每条3个seed，共180/180张2048×2048图像生成完成。材料见 [补测记录](Qwen-Image-2.1-Turbo/results/official-t03t04-20261030/README.md)。

[镜像与复验范围总表](镜像与复验总表.md)列明六套镜像的复用关系、额外依赖及未完成项。

## 复现方式

进入相应模型目录，按技术报告第 4 章执行。默认在 SCNet Notebook 内运行，HKS_ROOT 指向包含 platform/scnet-notebook 的项目根目录；Notebook 已是容器。模型权重与评测数据不随代码分发，从模型报告标注的来源获取并按固定清单核验。

各报告统一为简介、环境、适配与配置、启动与复现、功能测试、精度测评、首次性能、复现与限制八章。各模型吞吐的输入、批次和计时范围不同，不直接横向比较。
