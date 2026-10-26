# RoboBrain2.5-8B-NV on BW + SGLang 0.5.18

## 1. 简介

本项目完成 `BAAI/RoboBrain2.5-8B-NV` 在超算互联网单张 BW 加速卡上的 SGLang BF16 serving 适配。模型采用 Qwen3-VL 架构，支持文本、单图、多图输入，以及文本、坐标和轨迹输出；机器人实机任务不在本次部署范围。

- [模型下载](https://www.scnet.cn/ui/aihub/models/sugon_scnet/RoboBrain2.5-8B-NV)
- [发布者模型说明](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)
- [SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114287084172722177)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)，模型目录 `RoboBrain2.5-8B-NV/`。

## 2. 验证环境

SCNet Notebook，BW × 1，65520 MiB、80 CU、gfx936，逻辑设备 0，TP=1、BF16。Ubuntu 22.04.5，驱动 6.3.31-V1.5.3.beta，Python 3.10.12、DTK 26.04、HIP 6.3.26113。

| 组件 | 版本 |
|---|---|
| torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| sglang | `0.5.18+dtk2604.torch2110.2609231937.gc3bbbd` |
| sglang-kernel | `0.4.6.post1+dtk2604.torch2110.2609231956.gea1c29` |
| triton | `3.6.0+dtk2604.torch2110.2609231415.g8cce4a` |
| transformers | `5.12.1` |
| flash_attn | `2.8.4+dtk2604.torch2110.2609241509.g624d7b` |

区域：华中一区（A区）。Notebook 规格：BW × 1，显存 64GB；CPU 15 核，内存 59GB（平台显示值）；系统盘配额未显示，代码、权重和日志位于共享文件存储 /root/private_data。服务仅监听 `127.0.0.1:30125`，未使用外部服务映射。

基础镜像：`image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`。

固定摘要：`sha256:98dca2a455d62b758a2455e674fb480bc39ac236a2b3b472de8a1525743ec8a3f`。启动时自动重放第 3 节的固定源码补丁；未额外安装或升级推理依赖包。

权重通过 SCNet AI 社区克隆获取，路径 `/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`。13 个关键模型文件通过固定 SHA256 校验；说明文件和仓库属性文件不参与运行文件一致性校验。权重、结构、tokenizer 和图像处理配置未修改。环境及校验记录见 `env.log`、`results/scnet/runtime/`。

## 3. 适配优化方案

使用 Triton 文本注意力、SDPA 视觉注意力、确定性推理；上下文长度 32768，静态显存比例 0.75，最大运行请求数 64，chunked prefill 4096；完整解码图最大批次 32，关闭预填充图。

补丁作用于 `sglang/srt/batch_invariant_ops/batch_invariant_ops.py`。原确定性 MM/BMM 的流水配置申请 98304 字节共享内存，超过设备 65536 字节上限。修复根据 HIP 设备共享内存容量限制流水级数，并为 FP32 点积显式指定 IEEE 输入精度。高分辨率输入修复进一步使 BMM 在输出超过 2³¹ 个元素时使用 64 位批次偏移；边界回归覆盖 2,164,260,864 个输出元素并逐批检查全部值通过。

重放脚本为 `scripts/apply_deterministic_lds_patch.py`，实际差异及许可证位于 `patches/`。脚本校验补丁前 SHA256 `9a9fdb89d84f8c392ae9a46f46c0a846b332e78f4ae77c24f240f8db55e6a6e4`，补丁后 SHA256 `371f9816ae1614d944dce1616ee2e88a6d1b6dfed2ca660275ae28cb9785bb39`，拒绝修改其他版本的源码。

矩阵核检查中，BF16 GEMM 对 FP32 参考相对 L2 误差为 0.00166328，首行跨 batch 大小逐字节一致；BF16/FP32 BMM 数值及首批次一致性检查通过，记录见 `results/scnet/runtime/deterministic-gemm-check.json`。这些检查不能替代模型级 GPU 精度验收。

## 4. 服务启动脚本

```bash
bash run-scnet.sh
```

`run.sh` 校验权重、重放补丁、启动服务并轮询 `/health`。实际执行命令为：

```bash
python -m sglang.launch_server --model-path /root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV --served-model-name robobrain25-8b --host 127.0.0.1 --port 30125 --tp-size 1 --dtype bfloat16 --context-length 32768 --mem-fraction-static 0.75 --max-running-requests 64 --chunked-prefill-size 4096 --attention-backend triton --mm-attention-backend sdpa --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-decode 32 --enable-multimodal --enable-deterministic-inference
```

`MODEL_PATH` 可覆盖。日志、PID 和实际参数写入 `results/scnet/runtime/`；`stop.sh` 仅停止 PID、模型路径、服务名称和端口均匹配的本模型服务。

## 5. Smoke 测试

`/health` 返回 200，`/v1/models` 包含 `robobrain25-8b`。8 项功能和诊断检查全部通过：

| 测试 | 结果 |
|---|---|
| text | PASS |
| single_image | PASS |
| multi_image | PASS |
| multi_turn | PASS |
| stream_image | PASS |
| grounding | PASS |
| pointing | PASS |
| trajectory | PASS |

原始请求、响应、流式内容、图片与汇总见 `results/scnet/smoke-20261025T134807Z/`。坐标/轨迹检查验证格式和范围，不代表任务精度。

同一服务重复具身提示：3/3 组输出组内完全一致，逐次响应见 `results/scnet/stability-20261025T160055Z/`。

4 路并发响应与同服务串行结果逐字比较：24/24 一致，请求错误 0；原始记录见 `results/scnet/concurrency-20261025T160055Z/`。

## 6. 精度测评

T02 具身 serving 验收要求在公示图像（含深度图）输入集上，解析动作/坐标与 GPU 参考输出一致率 ≥98%。当前缺少对应输入集、逐例 GPU 参考输出及比较协议，`official_accuracy_pass=null`，不填写 PASS。

发布者论文中的空间理解与轨迹基准成绩不能替代逐例 GPU 一致率；同服务重复稳定性和功能格式检查也不作为该一致率分数。正式状态与缺失项见 `verify_result.json`；来源索引见 `官方精度基线查证.md`。

公开基准复测按照发布者采用的 FlagEvalMM 及公开数据开展。已固定 CV-Bench 完整 test 的 2,638 题、数据版本、提示和计分函数，正在全量运行，待完成后与论文 NV 8B 的 94.58 对照。高分辨率 BMM 修复后 8/8 预跑请求正常；这 8 题仅为运行诊断，不是全量精度。协议、其余基准状态与复现命令见 [公开基准复测](公开基准复测.md)。

本轮精度运行保留 BF16、确定性推理及 checkpoint 图像处理器，静态显存比例调整为 0.5，为高分辨率视觉临时矩阵留出空间，实际配置单独记录。

## 7. 性能参考

本节保留首版 `robobrain-bw-sglang-v1.0.0` 在静态显存比例 0.75、框架补丁 SHA256 `9668a00e1f8d068d3528c5f6fea55b7ec6994259ebf42fc2edf031525aa87a1b` 下的实测。本轮修复和精度配置尚未重跑此矩阵。

每请求一张 1080p 随机图，文本长度设置 4096/16384、固定输出 1024，并发 1/8/64，N=max(8,2C)。每档连续两轮，报告第二轮；每档开始前清空 prefix cache。12/12 轮、608/608 个请求完成，生成 token 数和配置校验通过。

图像和模板会增加实际输入 token 数，下表的文本长度设置不是固定总多模态输入长度。TTFT/TPOT 为客户端测得的平均值。

| 文本长度设置 | 并发 | 请求数 | 输出 tokens/s | TTFT ms | TPOT ms | 实际输入 token 总数 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 8 | 25.26 | 3285.35 | 36.42 | 50932 |
| 4096 | 8 | 16 | 126.98 | 15763.24 | 47.62 | 101932 |
| 4096 | 64 | 128 | 183.43 | 181088.22 | 111.36 | 814784 |
| 16384 | 1 | 8 | 21.51 | 6989.56 | 39.71 | 154656 |
| 16384 | 8 | 16 | 78.73 | 30050.46 | 72.29 | 309038 |
| 16384 | 64 | 128 | 84.95 | 516300.77 | 88.87 | 2469946 |

原始两轮 JSONL、逐档日志、参数和审计见 `bench/scnet-matrix-20261025T134821Z/`。本次结果作为平台首版性能基准，后续优化在相同卡型、框架和负载下比较。

## 8. 注意事项

在 SCNet Notebook 的共享存储中部署本模型目录，以及随代码提供的 `platform/scnet-notebook/env-scnet.sh`：

```bash
mkdir -p /root/private_data/hks/platform/scnet-notebook
cp platform/scnet-notebook/env-scnet.sh /root/private_data/hks/platform/scnet-notebook/
bash env_check.sh
bash run.sh                 # 在独立终端保持运行
bash smoke.sh
python verify.py            # 缺少官方参考时记录未判定并返回 2
BENCH_PROFILE=matrix bash bench.sh
```

性能和其他推理测试串行执行；16K、64 并发存在排队，延迟如实记录。首次启动会编译内核，启动超时为 1800 秒。尚未完成新 Notebook 中完整矩阵的性能偏差复现，不声明该项通过。权重不随代码仓库分发。
