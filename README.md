# 国芯国模 · BW 模型适配

十二个模型、十八个模型×框架版本的 SCNet BW 单卡/多卡技术报告、复现代码与原始测试证据。各优化版独立保留；公开自评、功能、性能和官方验收分别记录。

卡数为实际BW数量；TP1表示未启用跨卡张量并行。功能列按每次完整检查的通过项数列示，专项列为固定条件执行完成数，不代表精度分数；多进程及镜像复验范围见对应报告。精度列中的首版历史成绩、完整基准未测及评分未计算状态分别标明。

| 模型 | 框架 / 卡数 / TP | 功能验证 | 精度测试 | 首次性能与复现 |
|---|---|---|---|---|
| [Recon2Reason-Reasoning-4B](Recon2Reason-Reasoning-4B/README.md) | SGLang / 1卡 / TP1 | 功能 5/5 | OCRBench：46.30%（463/1000）；MMMU：48.56%（437/900） | 12 轮、608 请求 |
| [Recon2Reason-Reasoning-4B-vLLM](Recon2Reason-Reasoning-4B-vLLM/README.md) | vLLM / 1卡 / TP1 | 功能 5/5 | 赛事不要求精度测试 | 官方六档完整双进程1216请求；模板指标最大偏差9.65%；特殊token输出说明见报告 |
| [Recon2Reason-Reasoning-4B-vLLM · 优化版v1](Recon2Reason-Reasoning-4B-vLLM-Optimized-v1/README.md) | vLLM / 1卡 / TP1 | 功能 5/5 | 赛事不要求精度测试；优化一致性对照50条通过 | 官方六档608/608；完整矩阵吞吐提升2.21%–347.89%，五档超过10%；单矩阵服务 |
| [RoboBrain2.5-8B-NV](RoboBrain2.5-8B-NV/README.md) | SGLang / 1卡 / TP1 | 功能 8/8 | CV-Bench：88.10%（2638题） | 12 轮、608 请求；完整矩阵独立复跑待补 |
| [RoboBrain2.5-8B-NV-vLLM](RoboBrain2.5-8B-NV-vLLM/README.md) | vLLM / 1卡 / TP1 | 功能 8/8 | 赛事不要求额外精度测试；本版未测完整基准 | 官方六档608/608，实际输出622592 token；单矩阵服务，独立进程完整复现null |
| [MiMo-V2.6-Distill-Qwen-9B](MiMo-V2.6-Distill-Qwen-9B/README.md) | SGLang / 1卡 / TP1 | 功能 11/11 | GSM8K：89.50%（179/200）；MMLU：80.00%（160/200） | 12 轮、608 请求 |
| [MiMo-V2.6-Distill-Qwen-9B · vLLM](MiMo-V2.6-Distill-Qwen-9B-vLLM/README.md) | vLLM / 1卡 / TP1 | 功能 17/17 | GSM8K：90.50%（181/200，与SGLang相同固定子集） | 六档608/608；解码图短测吞吐提升126.24%–129.30%；单矩阵服务，c64延迟含排队 |
| [Qwen3-ASR-1.7B](Qwen3-ASR-1.7B/README.md) | vLLM / 1卡 / TP1 | 功能 6/6；PCM 4/4；专项 300/300 | test-clean：WER 1.6501%（2620条） | 128 请求；复跑最大偏差 7.28% |
| [Qwen3-ASR-1.7B · SGLang](Qwen3-ASR-1.7B-SGLang/README.md) | SGLang / 1卡 / TP1 | 功能 12/12；专项 300/300 | test-clean：WER 1.7501%、CER 0.6396%（2620条） | 音频性能128/128；图执行单并发短测提升283.16%，并发4冷编译另述；独立服务完整复现null |
| [Qwen3-ForcedAligner-0.6B](Qwen3-ForcedAligner-0.6B/README.md) | vLLM / 1卡 / TP1 | 功能 2/2；专项 30/30 | AAS未计算（专项未提供人工时间边界） | 每进程 512 请求；复跑最大偏差 7.18% |
| [Qwen3-ForcedAligner-0.6B 优化版v1](Qwen3-ForcedAligner-0.6B-Optimized-v1/README.md) | vLLM / 1卡 / TP1 | 功能 2/2；专项 30/30 | AAS未计算；相对首版29/30条边界一致，另1条最大差80ms | 512条对齐输入；相对首版历史吞吐提升17.77%–97.20%；独立进程复跑512条，吞吐偏差19.79% |
| [Trident-Qwen3VL-2B](Trident-Qwen3VL-2B/README.md) | vLLM / 1卡 / TP1 | 功能 6/6；HTTP embedding通过 | ChartQA Origin：NDCG@10 ×100 = 69.4483 | 每进程 1536 请求；复跑最大偏差 3.88% |
| [Trident-Qwen3VL-2B 优化版v1](Trident-Qwen3VL-2B-Optimized-v1/README.md) | vLLM / 1卡 / TP1 | 功能 6/6；HTTP embedding通过 | 本版未复测完整基准；首版ChartQA Origin 69.4483；演示向量对照通过 | 1536条embedding输入；相对初版历史吞吐提升21.14%–346.72%；单进程两轮 |
| [Qwen-Image-2.1-Turbo](Qwen-Image-2.1-Turbo/README.md) | SGLang / 1卡 / TP1 | 功能 8/8；专项 180/180 | 未计算质量分数（GenEval图片已生成；未计算CLIP/FID） | 每进程 9 实测 + 3 预热；复跑最大偏差 17.09% |
| [Kimi-VL-A3B-Instruct](Kimi-VL-A3B-Instruct/README.md) | vLLM / 1卡 / TP1 | 功能 8/8 | OCRBench：86.90%（869/1000；发布者86.70%） | 每进程32请求；吞吐偏差1.32%，全指标最大17.39% |
| [Kimi-VL-A3B-Instruct · 优化版v1](Kimi-VL-A3B-Instruct-Optimized-v1/README.md) | vLLM / 1卡 / TP1 | 功能 8/8 | 本版未复测完整基准；首版OCRBench 86.90%；优化对照32条文本及长度一致 | 官方六档608/608；同简单负载吞吐提升195.20%–384.94% |
| [Kimi-VL-A3B-Instruct · SGLang](Kimi-VL-A3B-Instruct-SGLang/README.md) | SGLang / 1卡 / TP1 | 功能 8/8 | OCRBench：87.20%（872/1000；发布者86.70%），上游评分逐题一致 | 六档608/608；图执行短测提升232.16%–401.47%；c64为客户端压力、运行上限8，TTFT/E2E含排队 |
| [AuK](AuK/README.md) | SGLang-Omni / 1卡 / TP1 | 功能 9/9；专项 120/120 | Seed-TTS-Eval英文：WER 1.7137%（1088条；较发布者增加0.6937个百分点） | 每进程32实测+6预热；均值延迟/音频吞吐偏差1.1583% |
| [MobileVLA-R1](MobileVLA-R1/README.md) | vLLM（RGB前端） / 1卡 / TP1 | 功能 12/12 | 未测试闭环导航精度（需Habitat环境） | 每进程9实测+3预热；均值延迟偏差4.64%；官方六档矩阵未执行（见报告7.2） |
| [MiniCPM-V-4.7-35B-A3B](MiniCPM-V-4.7-35B-A3B/README.md) | vLLM / 2卡 / TP2 | 功能 8/8 | OCRBench：88.80%（888/1000）；发布者90.40%，下降1.60个百分点≤2个百分点 | 六格复跑最大偏差10.31%；同Notebook独立进程 |
| [Kimi-Linear-48B-A3B-Instruct-SGLang](Kimi-Linear-48B-A3B-Instruct-SGLang/README.md) | SGLang / 4卡 / TP4 | 功能 6/6 | GSM8K：92.50%（185/200） | 6档 ×2轮，最大偏差12.69% |
| [Kimi-Linear-48B-A3B-Instruct-vLLM](Kimi-Linear-48B-A3B-Instruct-vLLM/README.md) | vLLM / 4卡 / TP4 | 功能 6/6 | GSM8K：91.50%（183/200） | 历史6档 ×2轮（均含预热），最大偏差14.30%；官方矩阵待补 |

[镜像与复验范围总表](镜像与复验总表.md)列明七套镜像的复用关系、额外依赖及未完成项。

## 复现方式

进入相应模型目录，按技术报告第 4 章执行。默认在 SCNet Notebook 内运行，HKS_ROOT 指向包含 platform/scnet-notebook 的项目根目录；Notebook 已是容器。模型权重与评测数据不随代码分发，从模型报告标注的来源获取并按固定清单核验。

各报告统一为简介、环境、适配与配置、启动与复现、功能测试、精度测评、首次性能、复现与限制八章。各模型吞吐的输入、批次和计时范围不同，不直接横向比较。
