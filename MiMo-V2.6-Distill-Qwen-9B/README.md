# MiMo-V2.6-Distill-Qwen-9B · BW × SGLang 技术报告

## 1. 简介

T01 文本模型适配，模型为 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`，BF16、单张 BW、TP1。权重来自[发布者 Hugging Face 仓库](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)，固定 revision `2367e865d009c13ac81713a2878291d33ab28177`，15 个运行文件及四个权重分片逐项 SHA256 校验通过。SCNet AI 社区克隆内容不完整，本次采用发布者固定快照，不将其标注为社区完整克隆。

## 2. 环境

SCNet Notebook，华中一区（A区），BW × 1，65520 MiB、80 CU，逻辑设备 0；CPU 15 核、内存 59 GiB。Ubuntu 22.04.5，Python 3.10.12，HIP 6.3.26113。

| 组件 | 实测版本 |
|---|---|
| PyTorch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| SGLang | 0.5.18+dtk2604.torch2110.2609231937.gc3bbbd |
| Transformers | 5.12.1 |
| Triton | 3.6.0+dtk2604.torch2110.2609231415.g8cce4a |

[实际使用的 SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114287084172722177)：`image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`，固定摘要 `sha256:98dca2a455d62b758a2455e674fb480bc39ac236a2b3b472de8a1525743ec8a3f`。环境与参数见 `results/scnet/validation-20261026T021625Z/runtime/`。

## 3. 适配与优化

采用 Triton 文本注意力，设置 `SGLANG_KV_LAYOUT_HCU_FA=false` 选择标准 KV 布局，保留 `SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS=1`，关闭调度重叠及 prefill 图执行。开启 full decode graph，最大图批次 64；上下文 32768、静态显存比例 0.85、最大运行请求 64、chunked prefill 4096、page size 1。使用原始 tokenizer、MiMo 模板与 `mimo` reasoning parser。

未修改权重、模型结构或图像处理器，不需要修改基础镜像框架源码。被测框架文件与基础镜像一致的检查见 `framework-source-check.json`。本次记录 SCNet 首版性能，不以不同硬件的成绩差异计算优化收益。

## 4. 服务启动脚本

将本模型目录部署至 `/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B`，配套的 `platform/scnet-notebook` 目录部署至 `/root/private_data/hks/platform/scnet-notebook`。Notebook 已是容器，不嵌套 Docker。将指定快照权重放在 `/root/private_data/hks/weights/MiMo-V2.6-Distill-Qwen-9B` 并保留完整性来源清单，或显式覆盖 `MODEL_PATH`。

```bash
cd /root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B
# 已有完整权重时仅校验并生成来源清单；需下载时去掉 --verify-only
python scripts/download_publisher.py --verify-only
export RUN_RESULTS_DIR="$PWD/results/scnet/reproduce/runtime"
export SMOKE_OUTPUT_DIR="$PWD/results/scnet/reproduce/smoke"
bash run-scnet.sh
# 另一个终端设置相同的两个输出目录后执行：
bash run-scnet.sh python scripts/smoke.py
BENCH_PROFILE=matrix bash run-scnet.sh bash bench.sh
```

服务为 `127.0.0.1:30124`，别名 `mimo-v2.6-9b`。启动时校验权重、检查端口、记录独立日志和实际参数。初始化完成须同时确认 `/health` 返回 200 与 `/v1/models` 包含该别名；仅模型列表可用不足以判定推理已就绪。

## 5. Smoke 测试

完整 11/11 通过：健康、模型列表、单轮、多轮、思考、普通流式、思考流式及四路并发。原始请求、响应及判定见 `results/scnet/validation-20261026T021625Z/smoke/`。

## 6. 精度测评

固定 GSM8K 与 MMLU 各 200 题公开自评样本，seed=20261023；来源固定版本和 SHA256 见 `sources/evaluation-files.json`，样本重建一致性见 `results/scnet/data-verification.json`。0-shot、greedy，temperature=0、top_p=1、开启思考、max_tokens=4096、并发 8。GSM8K 解析最终 `####` 数值，MMLU 解析最终 `ANSWER: A/B/C/D`；没有随机回退，错误、截断与解析失败均保留在分母。

| 数据集 | 正确 / 总数 | 自评准确率 | 请求错误 | 截断 | 未解析 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 179/200 | 89.50% | 0 | 0 | 2 |
| MMLU | 160/200 | 80.00% | 0 | 5 | 5 |

这些是固定公开样本成绩，不是完整测试集或发布者 GPU 基线。正式精度为**待官方测评代码复验**，`official_accuracy_pass=null`；不依据这组分数填写正式验收 PASS。原始逐题响应、输入摘要与完整协议见 `results/scnet/validation-20261026T021625Z/quality/`。

```bash
# 先按 sources/evaluation-files.json 获取两个固定来源文件到相应 cache 路径
bash run-scnet.sh python scripts/prepare_scnet_eval.py
bash run-scnet.sh python scripts/public_eval.py --output results/scnet/reproduce/quality --workers 8
```

## 7. 性能参考

4K/16K 输入 × 固定输出 1024 × 并发 1/8/64，共六档负载，每档连续两轮；请求数 max(8,2×并发)，每档清空 prefix cache，取第二轮。12/12 轮、608/608 请求完成，输入/输出 token 数与负载审计通过。

| 输入 tokens | 并发 | 请求数 | 输出 tokens/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 8 | 46.13 | 270.34 | 21.43 |
| 4096 | 8 | 16 | 381.14 | 1197.08 | 19.83 |
| 4096 | 64 | 128 | 1076.73 | 9382.85 | 50.28 |
| 16384 | 1 | 8 | 42.75 | 1230.96 | 22.21 |
| 16384 | 8 | 16 | 246.87 | 5643.67 | 26.90 |
| 16384 | 64 | 128 | 352.19 | 82191.24 | 73.78 |

原始两轮结果、参数、逐档日志与审计见 `bench/matrix-20261026T022326Z/`。另在 512 输入 / 128 输出下完成 4/4 轮短性能检查，单并发 46.21、四并发 225.23 tokens/s，见 `bench/smoke-20261026T021937Z/`。

## 8. 注意事项

功能、性能与公开自评串行测试。16K、64 并发存在排队，延迟如实记录。首次启动可能编译内核，须等真实健康检查通过后测试；每次复跑使用新的输出目录。权重、数据和编译缓存不随源码分发。

本镜像与启动参数已在当前 SCNet Notebook 完成上述测试；独立新 Notebook 的完整矩阵性能偏差尚未复验，不声明该项通过。镜像提供运行环境，源码和指定权重按上述路径部署；不依赖额外系统安装或框架补丁。

`python verify.py --selfcheck` 核对已保存的平台证据；`python verify.py` 记录正式精度待官方代码复验并返回 2，不使用占位基线计算 PASS。
