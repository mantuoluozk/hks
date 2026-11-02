# Kimi-VL-A3B-Instruct · BW × vLLM 技术报告

## 1. 简介

- [发布者模型与权重](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
- [发布者代码与技术报告](https://github.com/MoonshotAI/Kimi-VL)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114893402085822465/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Kimi-VL-A3B-Instruct/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 两个独立服务进程各 8/8 |
| 公开精度自评 | 完整 OCRBench 869/1000；发布者 867/1000 |
| 首次性能与复现 | 每进程4档、32个计数请求；吞吐最大偏差1.32%，四类指标最大偏差17.39% |
| 镜像验证范围 | 复用 Qwen-Image 镜像；同 Notebook 两个独立服务进程 |
| 赛事验收 | 本报告提供实测与公开基线对照，验收结论由主办方审核 |

T02 多模态适配，使用原版 **Kimi-VL-A3B-Instruct**，不混用 Thinking 或 Thinking-2506 的成绩。固定权重版本 `398eede0903cd983a2bfa0cc634e9ac1d843f375`；7 个 BF16 权重分片合计 32,816,098,528 字节，连同 tokenizer、processor、配置及说明等共20个文件，逐项通过发布者 SHA256 或 Git blob 哈希校验。来源清单及证据见 `sources/publisher-files.json`、`results/weights-download-20261028.json`。

## 2. 环境

| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；HIP 6.3.26113 |
| 框架 | vLLM 0.25.1 HCU 原生 KimiVL 实现 |
| 镜像版本 | `1.0.0` |
| 镜像地址 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0` |
| 镜像复验范围 | 同一 Notebook 内的两个独立服务进程；未另建实例复验 |

| 组件 | 实际版本 |
|---|---|
| torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| vllm | `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a` |
| transformers | `5.12.1` |
| tiktoken | `0.14.0` |
| pillow | `11.3.0` |
| openai | `2.30.0` |
| pydantic | `2.13.0` |
| pyarrow | `25.0.1` |
| requests | `2.34.2` |

复用镜像内的 `/opt/qwen3-asr-deps`，其中包含 OpenAI 2.30.0 等 API 依赖；不使用 Qwen-Image 的隔离 Transformers / Diffusers 运行栈。本模型未新增系统依赖，也未修改厂商 PyTorch、vLLM 或模型源码。实际包版本、运行参数和原生实现文件哈希见 `results/runtime-config.json`。

## 3. 适配与配置

采用 vLLM 原生 `KimiVLForConditionalGeneration`，保持发布者图像处理器、聊天模板与词表。单卡 BF16、上下文8192、最大运行序列2、最大批量token数8192、显存比例0.80、eager；图像上限2张，prefix cache和多模态处理缓存关闭。CPU线程数固定4；服务仅监听 `127.0.0.1:30004`。

厂商融合 MoE gate 不覆盖本模型 **64个路由专家、top-6** 配置，默认路径在初始化预热时明确报不支持。通过框架已有开关 `VLLM_HCU_USE_FUSE_MOE_GATE=0` 使用标准路由；不改专家权重、路由算法定义或缩放系数。该设置仅作用于本模型启动进程。实际 MoE 执行使用原生 Triton 后端，相关启动记录保存在 `results/scnet/native-vllm-v3-20261028/server.log`。

HTTP 功能、精度与性能均采用 greedy（temperature=0、top_p=1、seed=42）；发布者 Instruct 模型卡明确允许该评测设置。发布者日常采样推荐 temperature=0.2，与本次明确指定的 greedy 分开记录。

## 4. 启动与复现

模型目录放置于 `/root/private_data/hks/models/Kimi-VL-A3B-Instruct`，平台辅助目录为 `/root/private_data/hks/platform/scnet-notebook`。权重放在本模型 `weights/`，可用 `MODEL_PATH` 指定其他完整路径；来源与各文件哈希见上述清单。模型权重和数据集不随代码分发。

```bash
cd /root/private_data/hks/models/Kimi-VL-A3B-Instruct
# 获取并校验发布者的小型演示图片；此命令不下载权重：
bash run-scnet.sh python scripts/prepare_examples.py --download
# 核验权重、处理器并启动服务，在一个终端保持运行：
bash run.sh
```

在第二个终端执行功能与性能：

```bash
cd /root/private_data/hks/models/Kimi-VL-A3B-Instruct
bash run-scnet.sh python scripts/smoke_http.py --output results/functional-reproduce
bash run-scnet.sh python scripts/bench_http.py --smoke-summary results/functional-reproduce/summary.json --output results/performance-reproduce
```

评测使用固定 OCRBench 数据文件：`echo840/OCRBench`，版本 `92a54bd1384387c178d5a07140a2d85e0a3d12e1`，文件 `data/test-00000-of-00001.parquet`，67,576,988字节，SHA256 `d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5`。将该文件放至 `data/ocrbench/test.parquet`，或通过 `--data-path` 指定：

```bash
bash run-scnet.sh python scripts/eval_ocrbench.py --smoke-summary results/functional-reproduce/summary.json --output results/ocrbench-reproduce
python verify.py --selfcheck
```

Notebook 已是容器，不嵌套 Docker。使用新的结果目录；性能计时期间保持同卡无其他推理或评测任务。服务返回 OpenAI 兼容 `/v1/chat/completions`，支持单图、双图及SSE流式。

## 5. 功能测试

两个独立服务进程各完成 **8/8**：纯文本算术、英文红色图识别、中文蓝色图识别、合成文字HKS2026 OCR、发布者示例图、红蓝双图顺序识别、固定请求重复一致性、流式与非流式输出一致性。流式检查包含 `[DONE]` 及正常结束，保留原始分块和拼接内容。

这些检查验证接口及适配功能，不替代公开集精度。示例图通过发布者SHA256核验；合成测试图由固定代码生成。记录分别位于 `results/scnet/native-vllm-v3-20261028/functional/` 和 `results/scnet/fresh-vllm-20261028/functional/`。

## 6. 精度测评

使用完整 **OCRBench 1000题**，原始问题、原始图片；采用发布者图像处理配置，不另行覆盖分辨率。greedy、输出上限1024 tokens、并发2。请求中不提供参考答案，评分在推理后执行；所有题保留在分母。

| 指标 | 发布者公开值 | 本次实测 |
|---|---:|---:|
| OCRBench，满分1000 | 867 | **869** |
| 本次请求错误 / 输出截断 | — | 0 / 0 |

数值差为 **+2分**，即 +0.2 个准确率百分点。发布者成绩来自同一 Instruct checkpoint 的[模型卡](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/398eede0903cd983a2bfa0cc634e9ac1d843f375/README.md)。本报告明确所用数据版本、提示与解码设置，不将数值比较等同于完整评测框架逐项相同或赛事已验收。

评分使用 OCRBench 官方规则：普通文本忽略大小写，HME100k移除空格但保留大小写。另从保存的上游 `ocrbench_example.py` 提取实际评分循环，独立复算全部1000题，**逐题无差异**，结果仍为 869。见 `upstream-scoring-audit.json`；4项合成评分单测仅核对规则实现，不计入模型精度。

| 评测类别 | 题数 | 正确数 |
|---|---:|---:|
| Regular Text Recognition | 50 | 46 |
| Irregular Text Recognition | 50 | 41 |
| Artistic Text Recognition | 50 | 37 |
| Handwriting Recognition | 50 | 40 |
| Digit String Recognition | 50 | 33 |
| Non-Semantic Text Recognition | 50 | 48 |
| Scene Text-centric VQA | 200 | 186 |
| Doc-oriented VQA | 200 | 182 |
| Key Information Extraction | 200 | 166 |
| Handwritten Mathematical Expression Recognition | 100 | 90 |

原始预测、逐题评分、请求参数清单及汇总位于 `results/scnet/ocrbench-full-20261028/`。本次精度范围为 OCRBench，不声明已测试 MMMU、视频、OS Agent 等其他基准。

## 7. 首次性能

发布者同一张演示PNG、固定英文描述问题，服务实际计入1131个输入token。并发1/2 × 输出上限32/128，共4档；每档8个计数请求，共32。每轮先覆盖全部形状预热8请求，再每档预热2请求，总预热16，均不计入32个实测请求。

| 并发 | 输出上限 | 请求数 | 输出token/s | 平均延迟s | 平均TTFT ms | 平均TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 14.244 | 2.244 | 287.58 | 63.13 |
| 1 | 128 | 8 | 15.088 | 8.481 | 287.31 | 64.52 |
| 2 | 32 | 8 | 25.996 | 2.459 | 438.83 | 65.18 |
| 2 | 128 | 8 | 29.200 | 8.765 | 439.32 | 65.55 |

计时为HTTP客户端发送、服务预填充/解码、流式接收的总范围，排除模型初始化。吞吐按实际 `usage.completion_tokens` 总量除以计时窗口；TTFT取首个非空内容到达时间，TPOT按 `(请求延迟−TTFT)/(实际输出token数−1)` 计算，不把SSE分块数当token数。输出上限为负载参数，不把截断的性能样本作为精度结果。本矩阵不冒充赛事完整性能协议，亦不声明相对其他后端的优化提升。

## 8. 复现与限制

同一卡、同一镜像和Notebook内，停止首次服务后重新初始化独立服务进程，重复同样8项功能与4档性能；未另建Notebook实例。

| 并发 | 输出上限 | 首次输出token/s | 复跑输出token/s | 相对偏差 |
|---:|---:|---:|---:|---:|
| 1 | 32 | 14.244 | 14.115 | -0.91% |
| 1 | 128 | 15.088 | 15.127 | +0.26% |
| 2 | 32 | 25.996 | 26.340 | +1.32% |
| 2 | 128 | 29.200 | 29.283 | +0.28% |

输出吞吐最大绝对偏差 **1.32%**。进一步对输出吞吐、平均请求延迟、TTFT、TPOT四类指标逐项复算，16个比较值中最大绝对偏差 **17.39%**（单并发、输出上限32的TTFT），全部≤20%，检查 **通过**。完整比较见 `results/scnet/reproduction-metric-audit-20261028.json`。精度实测来自首次服务进程；第二进程验证范围为上述功能和性能，未重复完整OCRBench。不将同Notebook进程复跑表述为干净新实例复验。

`python verify.py --selfcheck` 核验已保存证据哈希、功能记录、精度计数及性能汇总。默认 `python verify.py` 校验已保存证据；通过返回0，不给出赛事精度验收结论。证据索引为 `results/evidence-manifest.json`，实际验收以主办方审核为准。

校验入口维护（2026-11-02）：默认校验已保存证据；检查通过返回0，检查失败返回非零。退出码不代表官方验收结论。原始预测、响应及计时记录保留。
