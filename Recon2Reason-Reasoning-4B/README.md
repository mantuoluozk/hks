# Recon2Reason-Reasoning-4B · BW × SGLang 技术报告

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Recon2Reason-Reasoning-4B)
- [发布者模型说明](https://huggingface.co/BAAI/Recon2Reason-Reasoning-4B)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114287084172722177)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Recon2Reason-Reasoning-4B/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 5/5 |
| 公开精度自评 | OCRBench 463/1000；MMMU 437/900 |
| 首次性能与复现 | 12 轮、608 请求 |
| 镜像验证范围 | 基础镜像上实测；未完成模型专用新镜像复验 |
| 赛事精度要求 | 官方不要求精度；已有自测保留，可按现有材料提交 |
<!-- report-resources:end -->

本项目完成 `BAAI/Recon2Reason-Reasoning-4B` 在超算互联网单张 BW 加速卡上的 SGLang BF16 部署与验证。模型采用标准 Qwen3-VL 架构，约 4.44B 参数，支持单图、多图和空间推理。


## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；设备与运行参数见下文 |
| 框架 | SGLang；实际组件版本见下文 |
| 镜像用途 | 共用基础镜像 |
| 镜像版本 | `0.5.18` |
| 镜像地址 | `image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18` |
| 镜像复验 | 基础镜像上实测；未完成模型专用新镜像复验 |
<!-- report-environment:end -->

SCNet Notebook，BW × 1，65520 MiB、80 CU；Ubuntu 22.04.5，驱动 6.3.31-V1.5.3.beta，Python 3.10，DTK 26.04，HIP 6.3.26113。部署精度为 BF16，TP=1，服务地址 `127.0.0.1:30123`。

| 组件 | 版本 |
|---|---|
| torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| triton | `3.6.0+dtk2604.torch2110.2609231415.g8cce4a` |
| sglang | `0.5.18+dtk2604.torch2110.2609231937.gc3bbbd` |
| sglang-kernel | `0.4.6.post1+dtk2604.torch2110.2609231956.gea1c29` |
| transformers | `5.12.1` |
| flash_attn | `2.8.4+dtk2604.torch2110.2609241509.g624d7b` |

镜像：`image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`


环境记录见 `env.log`。模型通过 SCNet AI 社区克隆获取，权重路径为 `/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B`。13 个关键文件通过 SHA256 校验，未修改权重、结构、tokenizer 或采样语义。

## 3. 适配与配置

直接使用镜像内推理框架和依赖，无额外模型运行包升级，无模型或框架算子补丁。

使用 Triton 文本注意力、SDPA 视觉注意力；上下文长度 32768，静态显存比例 0.70，最大运行请求数 64，chunked prefill 4096。启用 full decode graph，最大图批次为 32，关闭 prefill graph。

运行脚本加载 DTK 和驱动环境，使用 `/usr/bin/python` 与逻辑设备 0。完整参数见 `env.sh` 和 `results/scnet/runtime-config.json`。

## 4. 启动与复现

```bash
bash run.sh
```

实际执行命令：

```bash
python -m sglang.launch_server --model-path /root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B --served-model-name recon2reason-4b --host 127.0.0.1 --port 30123 --tp-size 1 --dtype bfloat16 --context-length 32768 --mem-fraction-static 0.70 --max-running-requests 64 --chunked-prefill-size 4096 --attention-backend triton --mm-attention-backend sdpa --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-decode 32 --enable-multimodal
```

模型路径可通过 `MODEL_PATH` 覆盖；服务启动后轮询 `/health`，启动日志写入 `server.log`。

## 5. 功能测试

```bash
bash smoke.sh
```

`/health` 返回 200，`/v1/models` 返回 `recon2reason-4b`。五项测试全部通过：

| 测试 | 实际输出 | 结果 |
|---|---|---|
| 纯文本计算 | `5` | PASS |
| 单图识别 | `red` | PASS |
| 多图识别 | `red, blue` | PASS |
| 多轮问答 | `blue` | PASS |
| 单图流式 | `red`，以 `[DONE]` 结束 | PASS |

原始请求、响应、图片与汇总见 `results/smoke-20261025T104608Z/`。

## 6. 精度测评

赛事最新要求不对本模型设置精度门槛。本节保留已完成的补充自测结果；这些成绩不再作为提交前置条件。

采用 OCRBench test 全量 1000 题、MMMU validation 全量 900 题。数据文件按固定版本下载并校验 SHA256；greedy 解码，temperature=0，top_p=1，max_tokens=2048，使用原始 processor，max_pixels=50176。

| 数据集 | 题数 | 正确数 / 得分 | 请求错误 | 截断 |
|---|---:|---:|---:|---:|
| OCRBench test | 1000 | 463/1000 | 0 | 9 |
| MMMU validation | 900 | 437/900（48.56%） | 0 | 0 |

截断题保留在分母。MMMU 随机回退为 0，严格计分同为 437/900。评分脚本五项自检通过。

本节为公开数据集补充实测，不进行赛事精度达标判定。汇总见 `verify_result.json`；逐题记录、评分结果和运行 manifest 见 `results/scnet/public-accuracy-20261025T115032Z/`。

## 7. 首次性能

单卡执行 4K/16K 输入 × 1K 输出 × 并发 1/8/64 的六档负载，每档连续两轮，第一轮预热、第二轮计入。每请求一张 1080p 图像，请求数为并发的两倍且至少为 8。12 轮共 608 个请求，均完成并通过固定输出 token 数校验。

| 输入 token | 输出 token | 并发 | 完成请求 | 输出 tokens/s | 平均 TTFT ms | 平均 TPOT ms |
|---|---|---:|---:|---:|---:|---:|
| 4096 | 1024 | 1 | 8 | 81.95 | 225.53 | 11.99 |
| 4096 | 1024 | 8 | 16 | 405.25 | 1688.97 | 16.80 |
| 4096 | 1024 | 64 | 128 | 725.79 | 21451.09 | 53.08 |
| 16384 | 1024 | 1 | 8 | 75.81 | 630.37 | 12.58 |
| 16384 | 1024 | 8 | 16 | 237.65 | 2551.70 | 27.31 |
| 16384 | 1024 | 64 | 128 | 230.45 | 164826.46 | 54.98 |

原始两轮 JSONL、日志、配置及汇总见 `bench/matrix-20261025T104835Z/`。本次结果作为首版性能基准，后续优化在同卡型、同框架、同负载下对比。

## 8. 复现与限制

以下命令在模型目录执行。首次部署时，将本模型目录放到 `/root/private_data/hks/models/Recon2Reason-Reasoning-4B`，并部署随代码提供的运行环境脚本：

```bash
mkdir -p /root/private_data/hks/platform/scnet-notebook
cp platform/scnet-notebook/env-scnet.sh /root/private_data/hks/platform/scnet-notebook/env-scnet.sh
bash env_check.sh
bash run.sh                 # 在独立终端保持运行
bash smoke.sh
source env.sh
python verify.py --output "results/scnet/public-accuracy-$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_PROFILE=matrix bash bench.sh
```

精度评测所用固定数据版本、文件路径与 SHA256 见 `sources/datasets-metadata.json`，下载脚本为 `scripts/download_public_data.py`。性能与精度测试串行执行。

- 16K 输入、64 并发时存在排队，延迟见性能原始记录。
- 首次启动需要编译内核，启动超时设置为 1800 秒。
- 权重通过 SCNet 模型库获取，不随代码仓库分发。
