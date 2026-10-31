# Kimi-Linear-48B-A3B-Instruct · BW × SGLang 技术报告

## 1. 简介

- [发布者模型与权重](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [发布者代码与技术报告](https://github.com/MoonshotAI/Kimi-Linear)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2116173177861849089)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Kimi-Linear-48B-A3B-Instruct/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 两个独立服务进程各 6/6 |
| 公开精度自评 | GSM8K 185/200（92.50%，零截断）；MMLU 未完成，不作精度结论 |
| 首次性能与复现 | 每轮 2 种提示长度 × 并发 1/2/4，共 6 档；两轮最大偏差 12.69% |
| 镜像验证范围 | 专用 Notebook 镜像内 SGLang TP4；两轮独立服务进程复跑 |
| 赛事验收 | 本报告提供实测与适配结论，验收由主办方审核 |

T01 文本适配，使用原版 **Kimi-Linear-48B-A3B-Instruct**。Kimi Linear 为 KDA 线性注意力与全局 MLA 按 3:1 配比的混合架构，48B 总参数、3B 激活参数。本轮不混用 Base 权重或 1.4T 实验模型阶段的任何数字。

## 2. 环境

| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook |
| 加速卡 / 精度 | BW × 4 / BF16 |
| 框架 | 镜像内 SGLang，模型类 `KimiLinearForCausalLM` |
| 镜像版本 | 专用 Notebook 镜像 |
| 镜像入口 | [SCNet 镜像 2116173177861849089](https://www.scnet.cn/ui/aihub/image/zack233/2116173177861849089) |
| 镜像复验范围 | 同一 Notebook 内两轮独立服务进程；未另建实例复验 |

模型权重位于共享存储，通过本目录 `weights/` 读取；启动脚本不下载、不复制权重。模型权重与评测数据集不随代码分发。

## 3. 适配与配置

采用 SGLang 原生 `KimiLinearForCausalLM`，保持发布者注意力实现与词表。启动参数为 TP4、BF16、上下文 8192、最大运行请求 4、静态显存比例 0.80、关闭 CUDA Graph，并显式指定 `--attention-backend triton`。

本镜像 SGLang 默认的 HCU MLA 后端在首次推理预热时，把旧式 Tensor 调度元数据传给要求 `FlashMLASchedMeta` 对象的 `flash_mla_with_kvcache`，触发类型断言。模型专用启动器改选 SGLang 的 Triton 注意力后端后，原权重正常加载并生成。**未修改全局 `site-packages`、未修改模型权重、未修改注意力算法定义。**

同镜像的 vLLM 0.25.1 路线能加载 TP4 权重，但聊天接口与原始补全接口均持续输出 token 0（`!`），关闭专家并行后现象不变，该路径未用于本报告任何结果。

## 4. 启动与复现

模型目录放置于 `/root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct`，平台辅助目录为 `/root/private_data/hks/platform/scnet-notebook`。

```bash
cd /root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct
# 核验权重并启动服务，在一个终端保持运行：
bash run-sglang.sh
```

在第二个终端执行功能、精度与性能：

```bash
cd /root/private_data/hks/models/Kimi-Linear-48B-A3B-Instruct
bash run-scnet.sh python scripts/http_checks.py --output results/functional-reproduce
bash run-scnet.sh python scripts/kl_gsm_mmlu.py --output results/accuracy-gsm8k-mmlu-200 \
  --url http://127.0.0.1:30148 --model kimi-linear-48b-a3b --workers 8 --max-tokens 4096
bash run-scnet.sh python scripts/bench_http.py --output results/performance-reproduce
```

服务端口默认 30148，可通过 `PORT`、`MODEL_PATH`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`TENSOR_PARALLEL_SIZE` 等环境变量调整。Notebook 已是容器，不嵌套 Docker。评测使用新的结果目录；计时期间保持同卡无其他推理或评测任务。

## 5. 功能测试

两轮独立服务进程各完成 **6/6** 项 HTTP 检查：英文生成、中文生成、多轮对话、SSE 流式、非法参数拒绝、串行与并发一致性。英文请求按要求返回 `READY`，中文 `2+3` 返回 `5`。

这些检查验证接口与适配功能，不替代公开集精度。原始请求与服务日志位于 `results/validation-20261030T153210Z/`。

## 6. 精度测评

发布者论文表 9 仅公布 **MMLU-Pro EM 72.7%**（5.7T Instruct 权重）。发布者**未公布**本 Instruct 权重的 GSM8K 与 MMLU 成绩：模型卡首页的 MMLU-Pro 51.0 属 1.4T 实验模型，Base 权重的 GSM8K 86.3 亦不适用于本权重，二者均已在 `sources/accuracy-reference.json` 的 `do_not_use` 字段中排除，本轮不引用。

评测使用固定 200+200 公开自评样本，与同批 MiMo-V2.6-Distill-Qwen-9B 使用同一份文件并核对 SHA256：

- `gsm8k`：SHA256 `03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546`，200 题
- `mmlu`：SHA256 `84f929be5a22b7f7895961ef19f3842dc3d9a276a42b82c42a6fe95e1395682d`，200 题

协议为 0-shot、greedy（temperature=0、top_p=1）、开启思考、max_tokens=4096、并发 32。GSM8K 解析最终 `####` 后的数值并按 Decimal 比较；MMLU 解析最终 `ANSWER: A/B/C/D`。全部样本保留在分母，错误、截断与解析失败均计为错误，无随机回退。

| 数据集 | 样本 | 正确 | 准确率 | 截断 | 解析失败 | 请求错误 |
|---|---:|---:|---:|---:|---:|---:|
| **GSM8K** | 200 | **185** | **92.50%** | 0 | 0 | 0 |
| MMLU（未完成） | 96 | 70 | 72.92% | 19 | 18 | 0 |

**GSM8K 为有效结果。** 200 题全部完成，零截断、零解析失败、零请求错误，是本模型唯一完整且干净的自评结果，作为本报告的精度结论。

**MMLU 未完成，不作为精度结论。** 该轮在跑完 96/200 时按用户决定停止，原因是单题耗时过长（中位 227 秒、最长 595 秒）；同时 4096 token 预算下有 19 条（19.8%）输出被截断，说明该模型在 MMLU 上的推理链显著长于同批 Qwen 系模型，4096 不足。72.92% 是截断且样本不完整情况下的中间值，**不对外引用，不填写为精度成绩。**

由于发布者仅公布 MMLU-Pro 全量 12,032 题的成绩，本报告**未取得可与发布者直接并列对照的完整基准成绩**：全量评测在单卡 DCU 上预估约 28 小时，本轮未执行。本节数字用于确认适配未破坏模型能力，不宣称为对发布者成绩的复现。

### 与同批模型的协议差异（如实记录）

temperature / top_p / max_tokens 与同批 MiMo-V2.6-9B 一致（0 / 1 / 4096），但本次并发为 **32**，同批 MiMo 为 **8**。greedy 解码下并发不影响逐题判定结果，仍在此注明，不宣称与之完全同协议。

原始逐条预测、请求参数清单与汇总位于 `results/accuracy-gsm8k-mmlu-200/`。

## 7. 首次性能

固定生成 128 token，短提示（重复 32）与长提示（重复 256）各测试并发 1、2、4，共 6 档。每组先执行 3 次预热，再执行 `3×并发数` 条计时请求；两轮分别重启服务。

| 提示重复数 | 并发 | 第一轮均值（秒/请求） | 第二轮均值（秒/请求） | 相对偏差 |
|---:|---:|---:|---:|---:|
| 32 | 1 | 11.138 | 11.072 | 0.59% |
| 32 | 2 | 12.909 | 11.271 | 12.69% |
| 32 | 4 | 11.416 | 11.270 | 1.28% |
| 256 | 1 | 11.246 | 11.021 | 1.99% |
| 256 | 2 | 11.334 | 11.311 | 0.20% |
| 256 | 4 | 11.189 | 11.287 | 0.88% |

计时为 HTTP 客户端发送、服务预填充/解码、接收的总范围，排除模型初始化。本矩阵为 TP4 四卡下的适配性能记录，不冒充赛事完整性能协议，亦不声明相对其他后端的优化提升。

## 8. 复现与限制

同一卡、同一镜像、同一 Notebook 内停止服务后重新初始化独立进程，重复同样 6 项功能与 6 档性能。两轮性能最大偏差 **12.69%**，低于复现阈值 20%，检查通过。较早的单次预热版本有两组偏差超过 20%，历史结果保留在 `results/validation-20261030T151416Z/`；增加预热后重新做了完整双轮测试。

限制与边界：

- MMLU 未完成且截断率偏高，不构成精度结论，不对外引用。
- 发布者唯一公布的成绩为 MMLU-Pro 全量 EM 72.7%，本轮未执行全量，因此无可与发布者直接对照的完整基准成绩。
- 精度评测并发为 32，与同批 MiMo 的 8 不一致，已在第 6 章注明。
- vLLM 路线在本镜像下输出异常（持续 token 0），未使用；本报告所有结果来自 SGLang TP4 + Triton 注意力后端。
- 复验范围为同 Notebook 内独立进程复跑，未另建 Notebook 实例。

证据索引：`submission-manifest.json`（20 个文件 SHA256）。默认 `python verify.py` 不给出赛事精度验收结论，实际验收以主办方审核为准。
