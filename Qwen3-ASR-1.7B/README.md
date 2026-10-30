# Qwen3-ASR-1.7B · BW × vLLM 技术报告

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen3-ASR-1.7B)
- [发布者模型说明](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Qwen3-ASR-1.7B/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 在线 6/6；PCM 4/4 |
| T03/T04 官方条件识别 | 300/300：AISHELL-1、LibriSpeech test-clean、test-other 各 100 条 |
| 公开精度自评 | test-clean 2620 条，WER 1.6501% |
| 首次性能与复现 | 128 请求；复跑最大偏差 7.28% |
| 镜像验证范围 | 专用镜像已完成新 Notebook 复验 |
| 发布者精度对照 | LibriSpeech test-clean：发布者 WER 1.63%，实测 1.6501% |
<!-- report-resources:end -->

T04 语音识别适配，模型为 `Qwen/Qwen3-ASR-1.7B`，BF16、单张 BW。权重来自 [SCNet AI 社区](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen3-ASR-1.7B)，11 个运行文件逐项 SHA256 校验通过，含两片权重共 4,698,521,512 字节。模型卡和 `.gitattributes` 不属于运行文件，社区说明文件与发布者快照不同，未将这两项计入运行文件一致性检查。模型文件清单见 `sources/modelscope-files.json`，实际校验见 `results/scnet/validation-20261026T080803Z/runtime/weights-check.json`。

## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；设备与运行参数见下文 |
| 框架 | vLLM；实际组件版本见下文 |
| 镜像用途 | ASR 专用镜像；其他模型按报告复用 |
| 镜像版本 | `v1` |
| 镜像地址 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1` |
| 镜像复验 | 专用镜像已完成新 Notebook 复验 |
<!-- report-environment:end -->

SCNet Notebook，华中一区（A区），BW × 1，逻辑设备 0。Python 3.10.12，HIP 6.3.26113。

| 组件 | 版本 |
|---|---|
| PyTorch（模块版本） | 2.11.0 |
| vLLM | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| Transformers | 5.12.1 |
| OpenAI Python | 2.30.0 |

[ASR 专用镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)：`appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1`，版本 `v1`。八个补充依赖固定版本见 `requirements-scnet.txt`，wheel 来源及 SHA256 见 `sources/scnet-wheel-files.json`；镜像内安装文件逐项核对通过，见 `results/scnet/validation-20261026T080803Z/runtime/image-dependencies.json`。PyTorch、Transformers、vLLM 保持基础镜像版本。

## 3. 适配与配置

采用镜像原生 vLLM Qwen3-ASR 实现，未修改模型权重、结构、tokenizer 或框架源码。补齐 librosa、soxr、audioread、lazy-loader、pooch、scikit-learn、editdistance；OpenAI Python 固定为 2.30.0，提供 vLLM 导入所需的 `NamespaceTool`。补充包安装至镜像内 `/opt/qwen3-asr-deps`，由启动脚本显式加载。

BF16、上下文 8192、最大运行请求 16、显存比例 0.4，关闭 eager，采用默认编译与图执行。Prefix cache 与多模态处理缓存关闭；使用原始音频处理器、greedy 解码、自然 EOS，不强制补满输出 token。本版记录平台首版结果，不以不同设备或环境的成绩差异计算优化奖励。

## 4. 启动与复现

将模型目录部署至 `/root/private_data/hks/models/Qwen3-ASR-1.7B`，配套 `platform/scnet-notebook` 目录部署至 `/root/private_data/hks/platform/scnet-notebook`。Notebook 已是容器，无需嵌套 Docker。指定社区权重放在 `run-scnet.sh` 默认路径，或通过 `MODEL_PATH` 覆盖。镜像重建文件为 `Dockerfile.scnet`。

```bash
cd /root/private_data/hks/models/Qwen3-ASR-1.7B
export IMAGE='appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1'
export EXPECTED_IMAGE_ID=''
export RUN_RESULTS_DIR="$PWD/results/scnet/reproduce/runtime"
bash run-scnet.sh
# 另一个终端设置相同 RUN_RESULTS_DIR 后执行：
SMOKE_OUTPUT_DIR="$PWD/results/scnet/reproduce/smoke" bash run-scnet.sh python scripts/smoke_test.py
BENCH_OUTPUT_DIR="$PWD/results/scnet/reproduce/bench" bash run-scnet.sh python scripts/bench_audio.py
```

服务为 `127.0.0.1:30128`，模型别名 `qwen3-asr-1.7b`。启动时校验权重、检查依赖和端口，并记录实际参数。`stop.sh` 仅停止模型路径、服务名及端口与记录相符的 PID。

## 5. 功能测试

新镜像中英文 WAV 分别通过聊天音频输入、音频转写 API、SSE 输出，共 **6/6**；两种语言的三个接口转写分别一致。原始请求响应及判定见 `results/scnet/validation-20261026T080803Z/smoke/`。

输入分块另测 **4/4**（中英文 × 1000/4000 ms 输入步长），固定发布者五个流式方法和状态类，绑定原生离线 vLLM。二秒解码块、累计音频重送、回退五 token，原始中间输出见 `results/scnet/validation-20261026T080803Z/input-streaming/`。该项是离线 PCM 输入分块适配器；线上 SSE 是文本输出流式，不据此宣称 WebSocket 输入流式、增量编码器缓存或时间戳。

## 6. 精度测评

发布者公开 LibriSpeech test-clean WER 为 **1.63%**，来源和口径见 `官方精度基线查证.md`。完整 test-clean 共 2620 条，固定来源文件 SHA256 为 `7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`。BF16、greedy、最大输出 1024、不指定语种、并发 4；Whisper `EnglishTextNormalizer` 固定源码版本见 `sources/whisper-source.json`。

| 运行 | 完成数 | WER | CER（同归一化、去空格） | 请求错误 / 截断 |
|---|---:|---:|---:|---:|
| 首轮平台自测 | 2620 | 1.6520% | 0.5306% | 0 / 0 |
| 专用镜像新 Notebook 复验 | 2620 | 1.6501% | 0.5285% | 0 / 0 |

完整 test-clean 实测 WER **1.6501%**，相比发布者 **1.63%** 增加 **0.0201 个百分点**，相对增加约 **1.23%**。逐条预测、参考归一化摘要及编辑距离见两轮 `accuracy/`。本次采用上述 Whisper 归一化与评分实现；该对照反映所列评测协议下的数值差异，不表示两侧评分实现完全相同。中文演示音频仅用于功能检查，未据此声称中文 CER 达标。

### T03/T04 官方条件测试

按固定条件完成 300 条语音识别：AISHELL-1、LibriSpeech test-clean、LibriSpeech test-other 各 100 条。300 条全部完成且反馈状态均成功。推理使用 BF16、temperature=0、最大输出 1024、自动识别语种；模型请求未传入参考文本或语种标签。

逐条识别结果、官方反馈表、环境信息及汇总见 [`results/official-t03t04-20261030/`](results/official-t03t04-20261030/README.md)。

```bash
bash run-scnet.sh python scripts/prepare_librispeech_parquet.py
EVAL_OUTPUT_DIR="$PWD/results/scnet/reproduce/accuracy" bash run-scnet.sh python scripts/evaluate_librispeech.py
```

## 7. 首次性能

负载为发布者两个演示 WAV：15.051 秒英文 48 kHz、4.204 秒中文 16 kHz，交替请求；并发 1/4/8/16，每档 N=max(8,2C)，连续两轮，报告第二轮。每轮另有两条预热。新镜像 8/8 项、128/128 个计数请求完成，不强制固定输出长度。

| 并发 | 请求数 | 音频秒/墙钟秒 | 平均请求 RTF | 平均延迟秒 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 32.20 | 0.0367 | 0.298 |
| 4 | 8 | 53.63 | 0.0776 | 0.638 |
| 8 | 16 | 61.56 | 0.1358 | 1.087 |
| 16 | 32 | 68.86 | 0.2634 | 1.991 |

聚合音频秒/墙钟秒越高越好，RTF=请求耗时/音频时长，越低越好且包含排队。两个演示样本不能外推完整语音场景，也不替代官方 T04 性能协议。逐请求输出、计时和两轮结果见 `results/scnet/validation-20261026T080803Z/bench/`。

## 8. 复现与限制

专用镜像新建 Notebook 后，不额外安装补充依赖，直接从 `/opt/qwen3-asr-deps` 加载；功能、性能、完整公开集精度及离线 PCM 检查串行完成。与首轮平台测试的第二轮性能比较如下，使用相同 BW 卡型、框架、参数、音频和并发；不同 Notebook 的波动按实际值记录。

| 并发 | 首轮平台吞吐 | 新镜像复验吞吐 | 偏差 |
|---:|---:|---:|---:|
| 1 | 32.07 | 32.20 | +0.38% |
| 4 | 56.70 | 53.63 | -5.40% |
| 8 | 66.39 | 61.56 | -7.28% |
| 16 | 68.41 | 68.86 | +0.65% |

最大绝对偏差 **7.28%**；≤20% 检查为 **通过**。证据位于 `results/scnet/validation-20261026T064636Z/` 与 `results/scnet/validation-20261026T080803Z/`，汇总文件为 `results/scnet/validation-summary.json`。权重、音频、数据集和编译缓存不随代码分发。首次启动须等待真实健康检查通过；每次复跑使用新的输出目录。在线服务必须停止后再运行离线 PCM 检查，避免同卡进程争用。

`python verify.py --selfcheck` 校验已保存证据。
