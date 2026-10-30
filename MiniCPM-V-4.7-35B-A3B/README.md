# MiniCPM-V-4.7-35B-A3B · BW ×2 (TP2) × vLLM 技术报告

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/OpenBMB/MiniCPM-V-4.7-35B-A3B)
- [发布者模型说明（固定版本）](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/tree/ee2d0a272c9d55db52a8b6841b6b24c9d0dda0a3)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `MiniCPM-V-4.7-35B-A3B/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 两个独立服务进程各 8/8 项（文本/单图/大图/双图/八图/流式/多轮/非法参数 400） |
| 公开精度自评 | OCRBench 888/1000=88.80%；完整集独立复算通过 |
| 首次性能与复现 | 每轮 6 格图片负载（2 尺寸 × 3 并发 × 每并发 3 请求 + 1 预热）；复跑最大偏差 10.31%，≤20% 检查通过 |
| 镜像验证范围 | 复用 SCNet AuK 镜像（vLLM 0.25.1+das185）；模型局部插件 + 模型内隔离依赖，不覆盖全局推理栈 |
| 发布者精度对照 | OCRBench 16x/instruct 90.40%，本次88.80%，下降1.60 pt，≤2 pt数值检查通过；完整协议差异如实保留 |
<!-- report-resources:end -->

T2 视觉多模态模型，35.2B 总参数 / 约 3.5B 激活（系列首个 MoE），文本骨干 `qwen3_5_moe_text` 40 层、256 专家每 token 取 8；混合注意力：每 4 层 1 层 full attention、其余为 GDN 线性注意力（发布者要求 SSM 状态 float32）；视觉塔 27 层 SigLIP2 系，canvas 3D M-RoPE，4×/16× 视觉 token 混合压缩（默认 16×）；上下文 256K。BF16 权重 16 个 safetensors 分片合计 70,425,903,448 字节，逐片 SHA256 与发布者固定版本 `ee2d0a272c9d55db52a8b6841b6b24c9d0dda0a3` 清单（`sources/publisher-files.json`）核对全部通过。

现有 SCNet 推理栈为 vLLM 0.25.1（含 MiniCPM-V 4.6 实现），上游已合入 4.7 支持（PR #58674），两者差距通过**模型局部插件**回移固定上游提交 `c85616efc6f0eaa50ca1f1a09c32368aedffd3d8` 的实现补齐：通过独立 dist-info 以 vLLM general plugin 注册，不覆盖镜像内任何全局文件；transformers 5.18.0、tokenizers 0.23.1、openai 2.30.0 隔离安装在本模型 runtime 内，全局 torch / vLLM / transformers 保持镜像原样（`results/global-stack-after-setup.json` 记录全局版本与原 vLLM 基类文件 SHA256）。

## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 双 BW 实例；实例区域未独立记录 |
| 加速卡 / 精度 | BW（gfx936）× 2，张量并行 TP=2 / BF16 |
| 框架 | vLLM 0.25.1+das185（全局未改）+ 模型局部插件 |
| 镜像用途 | [复用 AuK 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2115345884192739329)，未另建 MiniCPM 专用镜像 |
| 隔离依赖 | transformers 5.18.0 / tokenizers 0.23.1 / openai 2.30.0（仅本模型 runtime） |
| 服务端口 | 30147 |
<!-- report-environment:end -->

全局组件版本（未修改）：

| 组件 | 版本 |
|---|---|
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| vllm | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| transformers（全局） | 5.12.1 |
| tokenizers（全局） | 0.22.2 |
| openai（全局） | 2.6.1 |

模型运行时隔离安装 transformers 5.18.0 / tokenizers 0.23.1（满足发布者 `transformers>=5.18.0` 要求，官方配置 `transformers_version` 5.18.0）与 openai 2.30.0；仅通过 `PYTHONPATH` 供本模型进程使用，其余模型不受影响。设备与版本指纹见 `results/runtime-check.json`、`results/global-stack-after-setup.json`。

## 3. 适配与配置

**插件回移**（`python/minicpm_adapter/`，经 general plugin 注册）：固定上游提交的 `MiniCPMV4_7ForConditionalGeneration` 模型实现与 `minicpmv` reasoning parser；为旧框架补齐上游接口差异——`_scoped_mm_kwarg`、`_drop_vision_last_layer`、`_merge_and_resolve_mm_processor_kwargs`（按实际 HF schema 在模型局部桥接）及 `process_images` 兼容签名；`ParserManager` 包装仅对 `minicpmv` 生效；保留 canvas 位置编码、分片/换行处理与 DAS 非思考 `is_reasoning_end` 钩子。当前回移文件 SHA256 见 `sources/overlay-files.json`，发布文件见 `sources/release-files.json`；镜像原 vLLM 基类文件未被修改（SHA256 `6a099f72…`，见 `results/global-stack-after-setup.json`）。

**权重组织**：`weights/` 为 24 项运行视图——6 份小配置（config / generation / preprocessor / processor / tokenizer_config / chat_template）取自固定官方版本，逐文件 Git blob 核对通过（`results/publisher-config-check.json`）；16 个 safetensors 分片、index 与 tokenizer.json 等复用社区目录原件，逐片 LFS SHA256 全部通过（`results/community-weights-check.json`、`results/weights-check.json`）。社区目录中两份旧版模板与本固定版本不同，原件不动、校验差异记录在 `results/community-weights-check.json`。

**服务参数**（`run.sh`）：TP=2、BF16、上下文 8192、`--max-num-seqs 4`、`--gpu-memory-utilization 0.85`、eager（关闭 CUDA Graph / torch.compile）、`--reasoning-parser minicpmv`、关闭 prefix caching、`--limit-mm-per-prompt '{"image":8,"video":0}'`、端口占用即退出、启动前强制权重校验。显式 `--mamba-ssm-cache-dtype float32`：发布者 `text_config.mamba_ssm_dtype=float32`，而默认 auto 会使 SSM 状态跟随 BF16 权重，GDN 预热报 `initial_state must be float32`（首轮失败记录保留）。

## 4. 启动与复现

在复用镜像中设置 `HKS_ROOT` 为包含 `platform/scnet-notebook` 的仓库根目录，再进入本模型目录。默认共享部署目录为 `/root/private_data/hks/models/MiniCPM-V-4.7-35B-A3B`。权重与数据从报告来源准备，代码仓库不包含权重或数据集。

```bash
export HKS_ROOT="$(git rev-parse --show-toplevel)"
cd "$HKS_ROOT/MiniCPM-V-4.7-35B-A3B"

# 1) 安装固定依赖到模型隔离目录，构建局部插件
bash setup-runtime.sh
# 离线回归（不加载权重）：37 项全部通过
bash offline-check.sh

# 2) 从已完整准备的模型目录创建24项软链接视图，再校验
python scripts/prepare_weights.py /path/to/existing/MiniCPM-V-4.7-35B-A3B
python scripts/check_weights.py weights --output results/runtime/weights-check.json

# 3) 单服务启动（校验通过后进入引擎初始化）
bash run.sh

# 4) 功能 / 性能
bash smoke.sh          # 8 项功能
bash bench.sh          # 6 格图片负载

# 5) 完整验证：自动两轮独立服务（同 8 功能 + 6 格负载），汇总 20% 偏差并留档
python scripts/run_validation.py

# 6) CPU复算已保存证据及2 pt数值门槛
python verify.py --selfcheck
python verify.py
# OCRBench数据已按固定来源放入data/OCRBench后，可独立重跑（自动启动/停止自有服务）
python scripts/run_ocrbench_review.py
```

验证脚本每次运行冻结当轮源码副本（`job/source/`），只清理自有进程组，完成后自动停止服务；首轮失败记录与全部历史 job 原样保留，不删改。

## 5. 功能测试

两个独立服务进程各 8 项，共 16 项全部通过：纯文本、单图（448²）、大图（896²）、双图、八图（8×448²，触及 `image:8` 上限）、SSE 流式（含 `[DONE]` 校验）、多轮指代（"red" 正确回指）、非法 `max_tokens` 返回 400。图像回答内容与图像实际内容一致（颜色 / 布局 / 序列描述正确）。原始记录：`results/validation-20261030T025528Z/round{1,2}/smoke.json`。

预处理链路在 CPU 上以真实 vLLM processor 完成单 / 双 / 八图探测，原生 HF processor 另完成单 / 双图 × 思考开 / 关 4 项探测（`results/vllm-processor-probe.json`、`results/processor-probe.json`）；离线回归 37 项通过、0 排除（`results/vllm-processor-probe-v2.log`）。

## 6. 精度测评

本轮仅执行 OCRBench 完整 1,000 题。使用相同官方固定权重，保持 instruct 模式、16x 视觉压缩；关闭思考、greedy temperature=0、seed=42、max_slice_nums=9、输出上限 1,024 token、4 并发、8K 上下文。数据题目与图片未手工改写或裁剪。11 条技术检查样本不拼入完整集。

| 记录 | 正确 / 总数 | OCRBench |
|---|---:|---:|
| 首次自评：64-token 上限 | 848 / 1,000 | 84.80% |
| 本次复测：1,024-token 上限 | 888 / 1,000 | **88.80%** |
| 发布者公开图：16x / instruct | — | 90.40% |

本次提高 4.00 个百分点，距发布者对应配置的公开值低 1.60 个百分点。发布者同一图中 4x 的分数为 91.20%，不能与本次 16x 混用。官方原始图已逐字核对并按模型清单校验 SHA256，保存在 `sources/accuracy-reference/official-instruct.png`。完整官方提示、切片和采样协议尚未公布，因此本表是同基准、同视觉压缩与模式下的自评对照，不代表严格同协议复现或官方验收。按赛事技术报告模板的百分制下降≤2 pt门槛，本次1.60 pt下降的**数值检查通过**；`verify_result.json` 将数值检查与官方验收状态分开记录。

独立核验 1,000 个唯一 ID、题目、gold、请求参数、完整 API 响应、usage、finish_reason 和逐题评分全部一致。继续使用冻结的首次评分语义，按发布者原始空白/大小写规范独立复算亦为 888/1,000。无 HTTP 错误、无空回答。3 题达到输出上限，但截断前已包含正确答案，按固定 OCRBench 包含匹配规则计分；这 3 题及全部失败记录均保留在 1,000 题分母中。

新旧逐题对比：43 题由错转对、3 题由对转错。恢复的 43 题中，40 题新回答超过 64 token，28 题完整延续旧回答的相同文本前缀，支持首次输出预算不足造成明显损失。文档问答从 142/200 提升至 179/200。现有证据尚不能将剩余差距归因为 DCU 算子错误。

证据：`results/accuracy/ocrbench-16x-review-20261030T060837Z/` 下的 `ocrbench/protocol.json`、`ocrbench/summary.json`、`independent-audit.json`、`budget-effect-analysis.json`。完整响应单独保存在共享数据目录，以 SHA256 `7ba8f32f4cf6b1f390ef128b8bd7c15f0993e0c1dacabb55469861eb1350e11d` 对应；不向代码仓库复制数据集图片或原题。原始 84.80% 记录保留。

本轮不执行 MMMU、ChartQA 或其他精度基准。

## 7. 首次性能

负载：448² 与 896² 图片各 × 并发 1/2/4，每格 1 次预热 + 3×并发条请求，单轮 48 条；两轮独立服务进程，共 96 条请求（12 条预热、84 条计时）。指标为单请求平均完成耗时（秒，含图像预处理、推理与返回；预热不计入）。负载为固定合成样张的公开自测协议，不冒充赛事最终负载。

| 图片 | 并发 | 第一轮均值 (s) | 第二轮均值 (s) | 偏差 |
|---|---:|---:|---:|---:|
| 448² | 1 | 4.246 | 4.086 | -3.77% |
| 448² | 2 | 4.405 | 4.213 | -4.36% |
| 448² | 4 | 4.964 | 4.452 | -10.31% |
| 896² | 1 | 1.875 | 1.876 | +0.05% |
| 896² | 2 | 1.927 | 1.886 | -2.13% |
| 896² | 4 | 1.959 | 1.883 | -3.90% |

六格偏差全部 ≤20%，`performance_reproduced=true`（`results/validation-20261030T025528Z/summary.json`）。首次完整验证（`results/validation-20261030T023129Z/`）中 448²×4 一格偏差 26.23% 超出门槛（`performance_reproduced=false`），该结果与失败原因（首轮进程整体偏慢，具体原因未定位）原样保留；本表为条件一致后的完整复跑，非挑选数据。

## 8. 复现与限制

同一 Notebook、同一镜像栈、双 BW TP=2 上，两个独立初始化服务进程按相同配置与负载运行；两轮 6 格最大绝对偏差 **10.31%**，≤20% 检查**通过**。复现范围为同一 Notebook 内两个独立进程，不含跨 Notebook / 跨镜像复验；本模型使用上述复用镜像与模型局部隔离依赖。

限制如实记录：OCRBench 已完成全量自评，低于发布者对应公开值 1.60 个百分点，完整官方协议尚未取得；首轮 26.23% 超偏差记录保留；发布者 chat template 默认开启思考模式，复现实验须显式传 `enable_thinking=False` 才与本报告口径一致。

报告记录已完成的验证，赛事验收结论以主办方审核为准。

OCRBench 数据来源为 [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) 的完整 test 分片，原 parquet SHA256 为 `d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5`。数据已取得时可用 `python scripts/prepare_ocrbench_data.py <parquet路径>` 转换；该脚本仅转换本地文件，不下载数据。公开逐题生成记录保存在精度任务目录的 `generated-responses.jsonl`，不含原题、标准答案或图像；完整请求与响应原件另行保存，其哈希与公开审计一致。
