# Kimi-VL-A3B-Instruct · BW × SGLang 技术报告

## 1. 简介

- [发布者模型与固定版本](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/tree/398eede0903cd983a2bfa0cc634e9ac1d843f375)
- [发布者源码](https://github.com/MoonshotAI/Kimi-VL)
- [SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2117085616644177921)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)：`Kimi-VL-A3B-Instruct-SGLang/`
- [vLLM 原版](../Kimi-VL-A3B-Instruct/README.md)、[vLLM 优化版](../Kimi-VL-A3B-Instruct-Optimized-v1/README.md)

本版本以 SGLang 原生 MoonViT、DeepSeek-V2 MoE 实现部署 Kimi-VL-A3B-Instruct，提供文本、单图、双图与流式接口，并完成模型局部适配与基础图执行优化。

| 验证项 | 本轮结果 |
|---|---|
| 权重与配置 | 固定发布者版本运行文件全部校验通过；7 个 BF16 分片共 32,816,098,528 字节 |
| 功能 | eager、decode graph 及正式服务各 8/8；两候选功能输出逐项一致 |
| 精度 | 完整 OCRBench 872/1000；发布者参考 867/1000；上游评分循环重算一致 |
| 基础优化 | 四档带预热短测，同环境 SGLang eager 与 decode graph 对照 |
| 正式性能 | 六档各两次，608 条计数请求，第二次计分 |
| 独立服务完整性能复现 | null；本轮仅一个完整矩阵服务 |
| 官方验收 | null；自评结果与官方验收分开记录 |

## 2. 环境

SCNet Notebook，华中一区【A区】，单张 BW 64GB，TP1；CPU 配额 15 核，内存上限 63,350,767,616 字节。镜像为 `appstore.scnet.cn:5000/aihub/dcu/zack233/kimi-vl:v1.0.0`，版本 `v1.0.0`。Python 3.10.12，HIP 6.3.26113。

| 组件 | 实测版本 |
|---|---|
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| sglang | 0.5.18+dtk2604.torch2110.2609231937.gc3bbbd |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| pillow | 11.3.0 |
| requests | 2.34.2 |
| pyarrow | 25.0.1 |

沿用镜像运行栈与 API 依赖，没有安装或覆盖全局框架。模型目录携带进程局部投影补丁；镜像本身不包含本轮适配代码。当前 Notebook 完成 SGLang 加载、功能、精度和性能测试，不将已有 vLLM 镜像验证结论迁移为 SGLang 结论。

## 3. 适配与配置

当前 SGLang 的 `KimiVLMultiModalProjector` 先赋值 `GELUActivation()`，随后被 `QuickGELU()` 覆盖。固定发布者源码使用 GELU。本版本仅删除该覆盖语句，由 `scripts/launch.py` 对目标模型模块进行进程局部导入替换，原始与修正源码均保留；全局 site-packages 与已有 vLLM 目录保持原样。重建脚本 `scripts/prepare_overlay.py` 核对已审原文后才生成补丁。

视觉路径为 MoonViT → LayerNorm → 双线性层 GELU 投影；语言路径为 DeepSeek-V2，64 路由专家、每 token 选 6、2 个共享专家，保留 sigmoid/noaux_tc 路由、权重映射和发布者图像处理配置。MoE 使用 Triton runner，共享专家未融合。未量化、未修改权重、tokenizer 或聊天模板，也未缩减图像预算。

首次尝试 HCU MLA 后端在服务预热 decode 阶段触发 FlashMLA `tile_scheduler_metadata` 类型断言，完整失败记录保留在 `results/screen-20261103T030303Z/`；没有将该次加载算成功测试。后续使用 Triton 注意力后端与 SDPA 视觉注意力，功能通过后再筛选 decode graph。

最终参数：BF16、上下文 24576、显存比例 0.80、最大运行请求 8、chunked prefill 4096、CPU 数学库 4 线程、关闭 radix prefix cache，固定 seed 42。decode graph 捕获批次 1/2/4/8，约占 0.35 GiB；prefill 保持非图执行。正式服务启用 metrics。客户端并发 64 由服务端队列与最多 8 个运行请求处理。c64 表示客户端在途请求压力，不表示 64 条序列同时驻留 GPU。

## 4. 启动与复现

将固定版本完整模型放于 `weights/`，或建立指向现有完整权重的符号链接。OCRBench 使用 `sources/ocrbench-dataset.json` 固定的原始 parquet，置于 `data/ocrbench/test.parquet`；演示图置于 `data/examples/demo.png`。校验缺失或不一致时失败，不自动下载模型和数据。

```bash
bash env_check.sh
bash run.sh
# 另一个终端，对已启动服务执行：
bash smoke.sh
bash stop.sh
# 独立完整验证，自动管理自身服务：
bash bench.sh
# 只读校验已保存交付证据：
python verify.py
```

`bench.sh` 顺序执行启动、8 项功能、完整 OCRBench、上游独立评分、16K 图文上下文探测、六档性能矩阵和停止；不同时运行精度与性能流量。标准 `run.sh` 已用于本轮正式服务，功能脚本、完整测试及 `stop.sh` 的实际执行日志随结果保留。证据校验通过返回 0；缺失、不一致或真实检查失败返回非零，官方状态单独为 null。

## 5. 功能验证

8 项为纯文本算术、英文红色、中文蓝色、HKS2026 OCR、发布者演示图、红蓝双图顺序、相同请求重复、流式与非流式一致。三个服务均通过，输入、图像、响应、流式分块、结束标记与 usage 保留。该组是接口和功能检查，不作为完整精度成绩。

## 6. 精度测试

完整 OCRBench 1000 题，原图原题，2 并发；greedy、temperature 0、top_p 1、seed 42、最大输出 1024。所有题目保留在分母内，无按答题结果选择样本或补跑。实测正确 872，错误请求 0，输出截断 1，分数 872/1000（87.20%）。

| 类别 | 正确 / 总题数 |
|---|---|
| Regular Text Recognition | 45 / 50 |
| Irregular Text Recognition | 40 / 50 |
| Artistic Text Recognition | 36 / 50 |
| Handwriting Recognition | 40 / 50 |
| Digit String Recognition | 35 / 50 |
| Non-Semantic Text Recognition | 48 / 50 |
| Scene Text-centric VQA | 185 / 200 |
| Doc-oriented VQA | 182 / 200 |
| Key Information Extraction | 168 / 200 |
| Handwritten Mathematical Expression Recognition | 93 / 100 |

评分保留 OCRBench 的答案子串规则，HME100k 去空格，其余文本转小写。以固定上游评分循环对 1000 条预测重新计分，与本地逐题结果全部一致。发布者模型公开参考为 867/1000；完整发布者推理 harness 等价性未确认，不将数值对照冒充官方验收。历史 vLLM 的 869/1000 仍归属原框架。

## 7. 性能测试

### 7.1 基础优化短测

同 Notebook、同框架、同 BF16 权重、同演示图、相同提问与 greedy 参数。每候选各一遍，四档每档 8 个计时请求；额外共 16 个预热请求，初始化与图捕获不计入计时。输出上限 32/128，使用实际 usage 输出 token 与端到端墙钟计算吞吐。

| 并发 | 输出上限 | eager token/s | decode graph token/s | 相对提升 |
|---|---|---|---|---|
| 1 | 32 | 11.654 | 47.137 | 304.49% |
| 1 | 128 | 12.378 | 62.073 | 401.47% |
| 2 | 32 | 22.084 | 73.353 | 232.16% |
| 2 | 128 | 23.908 | 107.872 | 351.19% |

上述是 SGLang 同框架短负载对照，不能外推为正式长上下文矩阵的优化比例，也不以 vLLM 基线计算同框架优化收益。

### 7.2 正式矩阵

客户端长度参数 4096/16384，单张 1080p 随机图像，输出 1024，并发 1/8/64，seed 1，N=max(8,2C)。每档两次，共 608 计数请求，第二次计分；客户端内部预热与上下文探测另存。随机文本仅采样基础词表 0–163583，排除保留槽位与超出 embedding 的 ID，并重编码检查；权重和 tokenizer 文件不改。

| 文本长度参数 | 客户端并发 | 第二次请求数 | 输出 token/s | 平均 TTFT ms | 平均 TPOT ms |
|---|---|---|---|---|---|
| 4096 | 1 | 8 | 54.814 | 937.394 | 17.341 |
| 4096 | 8 | 16 | 129.601 | 10270.807 | 40.359 |
| 4096 | 64 | 128 | 132.040 | 333865.336 | 50.153 |
| 16384 | 1 | 8 | 32.950 | 7699.921 | 22.847 |
| 16384 | 8 | 16 | 56.690 | 43880.037 | 79.860 |
| 16384 | 64 | 128 | 58.030 | 782519.061 | 86.288 |

| 文本长度参数 | 客户端并发 | 墙钟秒 | 请求/s | 总 token/s | p99 TTFT ms | p99 TPOT ms | 平均 E2E ms |
|---|---|---|---|---|---|---|---|
| 4096 | 1 | 149.450 | 0.0535 | 335.471 | 977.071 | 18.925 | 18676.845 |
| 4096 | 8 | 126.419 | 0.1266 | 793.080 | 51082.012 | 45.708 | 51558.296 |
| 4096 | 64 | 992.672 | 0.1289 | 807.739 | 486996.299 | 59.470 | 385172.335 |
| 16384 | 1 | 248.622 | 0.0322 | 605.832 | 7718.492 | 22.878 | 31072.446 |
| 16384 | 8 | 289.008 | 0.0554 | 1042.455 | 127472.087 | 108.702 | 125576.522 |
| 16384 | 64 | 2258.710 | 0.0567 | 1067.142 | 1113642.997 | 111.583 | 870791.535 |

4K/16K 是客户端文本长度参数，实际输入包含图像与模板，以逐请求记录为准。客户端 input_lens 为处理器计算口径；16K 容量探测的 17796 token 与服务端五段预填充（4096×4+1412）一致，加 1024 输出预算低于 24576 上下文，交叉证据见 results/context-length-proof.json。保留客户端原始延迟、输出长度、错误与非空文本检查及服务端 metrics。TPOT/分位数按原客户端口径报告；没有逐 token 时间戳时，不声称可独立重建 p99 TPOT。c64 的 TTFT/E2E 包含请求发送、前端处理、服务端排队与调度、图像处理、预填充和流式接收等端到端开销；不能解释为 64 条请求同时执行时的纯预填充或模型计算延迟。该数据反映当前运行上限 8 配置的真实服务压力表现，保留原值，不扣除未经单独测量的排队时间。并发 1/8 同样是端到端计时，也可能包含处理和调度等待。吞吐是该配置的端到端服务吞吐，不代表框架或硬件的理论上限。调高 max-running-requests 不会自动增加 KV 容量，本轮未实测更高运行上限，不将显存不足或性能收益写成既定结论。两次档位计时属于同一服务，独立服务完整矩阵复现为 null。

## 8. 代码与原始数据

| 路径 | 内容 |
|---|---|
| `scripts/preflight.py`、`sources/` | 固定来源校验、真实框架源码、发布者投影与补丁 |
| `run.sh`、`serve-formal.sh`、`serve-scnet.sh` | 标准启动与最终参数 |
| `scripts/smoke_http.py` | 8 项 HTTP 功能及原始响应 |
| `scripts/eval_ocrbench.py`、`scripts/audit_ocrbench.py` | 完整精度与上游评分复核 |
| `scripts/run_formal_matrix.py`、`scripts/benchmark_valid_vocab.py` | 正式矩阵与合法随机词表 |
| `results/screen-20261103T030631Z/` | 两候选完整预热、逐请求输出与原始计时 |
| `results/formal-20261103T031700Z/` | 正式服务、1000 题预测、评分、矩阵、日志和运行源码 |
| `verify.py`、`submission-manifest.json` | 已保存证据与发布字节检查 |

原始运行数据位于共享存储 `/root/private_data/hks/models/Kimi-VL-A3B-Instruct-SGLang/`。重复请求内嵌图片在可发布视图中按内容哈希去重至 `assets/`，请求图像哈希、原始响应及计时保持可追溯；未改动共享原始记录。正式随机图像产物和大于 8 MiB 的原始输入清单按 `sources/external-benchmark-assets.json` 逐文件索引另存于对应共享路径。权重、数据集和缓存不随代码发布。历史失败记录保留，官方成果判定以评审为准。
