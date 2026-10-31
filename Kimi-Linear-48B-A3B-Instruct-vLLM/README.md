# Kimi-Linear-48B-A3B-Instruct · BW × vLLM 技术报告

## 1. 简介

- [发布者模型与权重](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [发布者代码与技术报告](https://github.com/MoonshotAI/Kimi-Linear)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2116173177861849089)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Kimi-Linear-48B-A3B-Instruct/`
- 同模型另一框架版本见仓库 `Kimi-Linear-48B-A3B-Instruct-SGLang/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 两个独立服务进程各 6/6 |
| 公开精度自评 | GSM8K **183/200（91.5%）**，零截断、零解析失败 |
| 首次性能与复现 | 6 档 × 两轮独立服务进程（均含全局预热）；全指标最大偏差 14.30% |
| 镜像验证范围 | 专用 Notebook 镜像 + 运行期补丁；同 Notebook 两个独立服务进程 |
| 赛事验收 | 本报告提供实测与适配结论，验收由主办方审核 |

T01 文本适配的 **vLLM 路线**。Kimi Linear 为 KDA 线性注意力与全局 MLA 按 3:1 配比的混合架构，48B 总参数、3B 激活参数。本报告不混用 Base 权重或 1.4T 实验模型阶段的任何数字。

## 2. 环境

| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook |
| 加速卡 / 精度 | BW × 4 / BF16 |
| 框架 | vLLM |
| 镜像版本 | 专用 Notebook 镜像 |
| 镜像入口 | [SCNet 镜像 2116173177861849089](https://www.scnet.cn/ui/aihub/image/zack233/2116173177861849089) |
| 镜像复验范围 | 同一 Notebook 内两个独立服务进程；未另建实例复验 |

| 组件 | 实际版本 |
|---|---|
| torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| vllm | `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a` |
| vllm-hcu | `0.25.1+dtk2604.torch2110.2609272150.gee325d` |
| transformers | `5.12.1` |
| boltops | `0.1.0+dtk2604.torch2110.2609241556.ga133e6` |

模型权重位于共享存储，通过本目录 `weights/` 读取；启动脚本不下载、不复制权重。**镜像本身不含本轮修复**，新建实例须先运行 `runtime/apply_vllm_kda_fix.py`。

## 3. 适配与配置

采用 vLLM 的原生 `KimiLinearForCausalLM`。启动参数为 TP4、BF16、上下文 8192、最大运行请求 8、显存比例 0.85、eager 模式、关闭 prefix caching、关闭专家并行。

### 3.1 必须的运行期修复

该模型在标准 vLLM 路径下**无法直接生成**：服务能加载全部权重，但首个生成 token 即为异常，输出退化为连续 `!`。逐层探针定位到第 0 层 KDA 首次产生非有限值，进而拆解为两处独立缺陷：

**缺陷一：V 卷积输入布局。** 卷积前的 Q/K/V 投影、卷积权重与状态均为有限值，但卷积后 V 在 4 卡中的 3 卡出现非有限值。卷积调用前张量非连续布局，外部 `causal_conv1d` 与 Triton 回退路径均受影响。修复方式：卷积输入前加 `.contiguous()`。

**缺陷二：KDA chunk intra 门控指数溢出。** 实际累计门控最小值在四个 TP 分片上分别约为 −279、−495、−719、−2495；只有 −279 的分片原本正常，其余三卡在 intra 阶段产生 NaN。调用链为 HCU `chunk.py` → **BoltOps** `chunk_kda_fwd_intra`，插件目录内的同名 `chunk_intra.py` 并非整模型实际执行路径。根因是 sub-chunk 内同时计算 `exp2(g−基准)` 与 `exp2(基准−g)`，在大跨度下溢出。修复方式：强制走安全门控分支（`safe_gate=True`）、关闭 BoltOps 对 128 维的多项式优化分支、并对成对指数差值做范围限制。

补丁脚本 `runtime/apply_vllm_kda_fix.py` 只改安装目录三个文件的五处语句，幂等、校验原文、首次应用时在旁保存原文件。已用原始文件重放验证，结果与当前机器一致。

> 说明：本修复**未修改模型权重、未修改注意力算法定义、未修改 tokenizer 或采样语义**。禁用 BoltOps 多项式优化可能影响性能，已在下文如实记录，并通过 GSM8K 确认精度未受损。

## 4. 启动与复现

模型目录放置于 `/root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct`。

```bash
cd /root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct
# 1) 新建实例或未打补丁时，先应用运行期修复（已打过会提示 already patched）
python3 runtime/apply_vllm_kda_fix.py --site-packages /usr/local/lib/python3.10/dist-packages
python3 runtime/apply_vllm_kda_fix.py --check --site-packages /usr/local/lib/python3.10/dist-packages

# 2) 启动服务（一个终端保持运行）
PORT=30149 TENSOR_PARALLEL_SIZE=4 HIP_VISIBLE_DEVICES=0,1,2,3 \
ENABLE_EXPERT_PARALLEL=0 VLLM_HCU_USE_CUSTOM_OPS=0 VLLM_KIMI_CONV1D_BACKEND=0 \
bash run.sh

# 3) 另一个终端做功能、精度与性能
cd /root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct
python3 scripts/vllm_smoke.py
python3 scripts/eval_gsm8k_vllm.py --output results/accuracy-gsm8k-200-vllm --workers 8
python3 scripts/bench_vllm.py --output results/performance-vllm
```

服务端口默认 30149。Notebook 已是容器，不嵌套 Docker。评测使用新的结果目录；计时期间保持同卡无其他推理或评测任务。

## 5. 功能测试

两轮独立服务进程各完成 **6/6** 项 HTTP 检查：英文生成、中文生成、多轮对话、SSE 流式、非法参数拒绝、串行与并发一致性。英文请求按要求返回 `READY`，中文 `2+3` 返回 `5`。

这 6 项检查中，全部检查均未出现退化输出（连续 `!`），另有 4 并发同题一致性检查结果完全一致。

这些检查验证接口与适配功能，不替代公开集精度。原始请求与服务日志位于 `results/`。

## 6. 精度测评

使用固定 **GSM8K 200 题**公开自评样本，与 SGLang 版使用同一份数据文件并核对 SHA256：

- `gsm8k`：SHA256 `03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546`，200 题

协议为 0-shot、greedy（temperature=0、top_p=1）、开启思考、max_tokens=4096、并发 8。评分取最终 `####` 后的数值并按 Decimal 比较，容忍 `$`、千分位与加粗标记。全部样本保留在分母。

| 指标 | 结果 |
|---|---:|
| GSM8K 正确数 / 总数 | **183 / 200** |
| 准确率 | **91.5%** |
| 请求错误 / 输出截断 / 解析失败 / 退化输出 | 0 / 0 / 0 / 0 |

### 6.1 与同模型 SGLang 版的对照

两个框架版本使用同一份数据、同一套提示与评分口径：

| 框架 | 后端 | 正确数 | 准确率 | 相对 SGLang |
|---|---|---:|---:|---:|
| SGLang | Triton 注意力后端 | 185/200 | 92.50% | — |
| vLLM | vLLM | 183/200 | 91.5% | −1.0 pt（−2 题） |

两版差距为 2 道题，属正常范围内的引擎实现差异，**说明运行期修复未破坏模型精度**。

## 7. 首次性能

固定生成 128 token（流式计时），短提示（重复 32）与长提示（重复 256）各测试并发 1、2、4，共 6 档。两轮分别使用**独立服务进程**。

**预热协议**：每轮服务就绪后先执行**全局预热**——按矩阵全部 6 个形状各跑一遍，重复 2 轮，再额外做一次 256 token 长生成；全部预热完成后才开始计时。此外每个档位计时前再做 3 次预热请求。这样冷启动、kernel 编译、显存分配增长与调度缓存都已在计时前完成，**冷启动按设计被排除，而不是事后剔除数据**。

| 提示重复 | 并发 | 吞吐 tok/s（A→B） | 偏差 | 平均延迟 s（A→B） | 偏差 | 平均 TTFT ms（A→B） | 偏差 | 平均 TPOT ms（A→B） | 偏差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 10.96 → 11.09 | 1.12% | 3.739 → 3.698 | 1.10% | 215.11 → 214.21 | 0.42% | 88.11 → 87.09 | 1.16% |
| 32 | 2 | 21.48 → 21.72 | 1.10% | 3.816 → 3.775 | 1.07% | 228.80 → 220.06 | 3.82% | 89.69 → 88.87 | 0.91% |
| 32 | 4 | 43.45 → 43.16 | 0.66% | 3.774 → 3.798 | 0.64% | 223.02 → 221.90 | 0.50% | 88.76 → 89.41 | 0.73% |
| 256 | 1 | 10.76 → 10.77 | 0.18% | 3.471 → 3.403 | 1.96% | 221.13 → 231.02 | 4.47% | 89.43 → 88.87 | 0.63% |
| 256 | 2 | 21.28 → 20.60 | 3.19% | 3.507 → 3.499 | 0.23% | 268.80 → 307.24 | 14.30% | 89.13 → 89.50 | 0.42% |
| 256 | 4 | 41.44 → 39.19 | 5.43% | 3.580 → 3.695 | 3.21% | 319.54 → 313.73 | 1.82% | 89.33 → 91.56 | 2.50% |

计时为 HTTP 客户端发送、服务预填充/解码、流式接收的总范围，排除模型初始化。吞吐按实际 `usage.completion_tokens` 总量除以计时窗口；TPOT 按 `(请求延迟−TTFT)/(实际输出token数−1)` 计算。

## 8. 复现与限制

### 8.1 复现结论

同一卡、同一镜像、同一 Notebook 内两个独立服务进程，各含全局预热，复跑同样 6 档：

| 指标 | 最大相对偏差 | 出现位置 | 20% 门限 |
|---|---:|---|---|
| 吞吐 | 5.43% | repeat=256 / 并发4 | 通过 |
| 平均延迟 | 3.21% | repeat=256 / 并发4 | 通过 |
| TPOT | 2.50% | repeat=256 / 并发4 | 通过 |
| TTFT | 14.30% | repeat=256 / 并发2 | 通过 |
| **全指标** | **14.30%** | — | **通过** |

全部指标均在 20% 门限内，**检查通过**。完整比较见 `results/performance/performance-comparison.json`。

### 8.2 关于预热

未做全局预热时，模型加载后的第一个服务进程在高并发档位会出现明显偏高的 TTFT 与偏低的吞吐（同档 token 数接近但耗时显著拉长），属冷启动效应而非模型或配置差异。本报告的做法是**在两轮计时前都执行相同的全局预热流程**，使两轮都处于稳态，因此上面的偏差反映的是真实服务稳定性，不包含冷启动成分。

### 8.3 限制与边界

- **镜像不含修复**：专用镜像可作基础复用，但新建实例必须运行 `runtime/apply_vllm_kda_fix.py` 后再启动，否则会重现连续 `!`。不得将本轮结果表述为"新镜像已内置修复"。
- **补丁影响性能**：为规避 NaN，禁用了 BoltOps 的 128 维多项式优化分支，可能带来性能损失。文中性能数字为**打补丁后的实际值**，未宣称相对未打补丁版本的提升。
- **精度范围**：仅 GSM8K 200 题固定公开自评样本，不声明已测试其他基准。
- **复验范围**：同 Notebook 内独立进程复跑，未另建 Notebook 实例。
- **上游版本线**：海光仓库 `v0.28.1-dev` 采用 `patch/worker/op_opt/` 的 patch 机制，与本机 `0.25.1` 直接实现版目录结构不同，本补丁不能原样用于该版本线。

证据索引：`submission-manifest.json`。默认 `python verify_vllm.py` 不给出赛事精度验收结论，实际验收以主办方审核为准。
