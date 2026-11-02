# Kimi-VL-A3B-Instruct · BW × vLLM 优化版v1技术报告

## 1. 简介

- [发布者模型与权重](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
- [发布者代码与技术报告](https://github.com/MoonshotAI/Kimi-VL)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2117085616644177921)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 独立目录 `Kimi-VL-A3B-Instruct-Optimized-v1/`
- [原版技术报告](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks/-/blob/main/Kimi-VL-A3B-Instruct/README.md)

| 验证项 | 当前结果 |
|---|---|
| 功能 | 优化版短负载服务与正式矩阵服务各8/8 |
| 公开精度自评（eager首版） | 完整 OCRBench 869/1000；发布者 867/1000 |
| 首次性能与复现 | eager首版：每进程4档、32个计数请求；吞吐最大偏差1.32%，四类指标最大偏差17.39% |
| 解码图优化 | 同卡、同框架、同简单负载，带预热各实测一遍；四档吞吐提升195.20%–384.94%，32条输出文本及长度逐条一致 |
| 官方六档性能 | 4K/16K文本参数 × 1024输出 × 并发1/8/64，每档两次，共608/608请求，全部输出1024 token |
| 提交镜像 | Kimi-VL专用镜像 `v1.0.0`，内置固定API依赖和模型启动代码 |
| 赛事验收 | 本报告提供实测与公开基线对照，验收结论由主办方审核 |

T02 多模态适配，使用原版 **Kimi-VL-A3B-Instruct**，不混用 Thinking 或 Thinking-2506 的成绩。固定权重版本 `398eede0903cd983a2bfa0cc634e9ac1d843f375`；7 个 BF16 权重分片合计 32,816,098,528 字节，连同 tokenizer、processor、配置及说明等共20个文件，逐项通过发布者 SHA256 或 Git blob 哈希校验。来源清单及证据见 `sources/publisher-files.json`、`results/weights-download-20261028.json`。

## 2. 环境

| 环境项 | 配置 |
|---|---|
| 平台 | SCNet Notebook |
| 历史首版区域 | 华中一区（A区） |
| 本轮资源组 | 149组，资源配置见随附截图 |
| 加速卡 / 精度 | BW × 1 / BF16；HIP 6.3.26113 |
| 框架 | vLLM 0.25.1 HCU 原生 KimiVL 实现 |
| 提交镜像版本 | `v1.0.0` |
| 提交镜像地址 | `appstore.scnet.cn:5000/aihub/dcu/zack233/kimi-vl:v1.0.0` |
| 本次性能实测镜像 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0` |
| 实测资源 | BW 64GB × 1，CPU配额15核，内存59GiB；资源配置见 `results/resource-20261102.png` |

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

使用 `/opt/qwen3-asr-deps` 中的OpenAI 2.30.0等固定API依赖；专用镜像以 `image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18` 为基础，离线内置依赖和启动代码。启动以当前仓库模型目录为准。权重和数据位于共享存储，镜像不包含权重。BF16框架与模型源码保持原版。历史环境证据为 `results/runtime-config.json`，本轮优化实测环境与源码哈希为 `results/runtime-optimized-20261102.json`，提交镜像信息为 `sources/image-optimized-20261102.json`。

## 3. 适配与配置

采用 vLLM 原生 `KimiVLForConditionalGeneration`，保持发布者图像处理器、聊天模板与词表。历史首版为单卡 BF16、上下文8192、最大运行序列2、最大批量token数8192、显存比例0.80、eager；图像上限2张，prefix cache和多模态处理缓存关闭。CPU线程数固定4；服务仅监听 `127.0.0.1:30004`。

解码图优化的简单负载对照保留上述参数，只将eager改为 `compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}`。mode=0关闭Inductor编译，解码阶段使用HIP图重放，预填充仍为eager。权重、BF16、标准MoE路由、采样和缓存开关均未改变。完整长输入矩阵使用 `serve-optimized.sh`：上下文32768、最大运行序列8、显存比例0.85，解码图捕获尺寸为1/2/4/8，客户端并发64由服务排队执行。

厂商融合 MoE gate 不覆盖本模型 **64个路由专家、top-6** 配置，默认路径在初始化预热时明确报不支持。通过框架已有开关 `VLLM_HCU_USE_FUSE_MOE_GATE=0` 使用标准路由；不改专家权重、路由算法定义或缩放系数。该设置仅作用于本模型启动进程。实际 MoE 执行使用原生 Triton 后端，相关启动记录保存在 `results/scnet/native-vllm-v3-20261028/server.log`。

HTTP 功能、精度与性能均采用 greedy（temperature=0、top_p=1、seed=42）；发布者 Instruct 模型卡明确允许该评测设置。发布者日常采样推荐 temperature=0.2，与本次明确指定的 greedy 分开记录。

## 4. 启动与复现

优化版目录放置于 `/root/private_data/hks/models/Kimi-VL-A3B-Instruct-Optimized-v1`，平台辅助目录为 `/root/private_data/hks/platform/scnet-notebook`。将完整固定版本权重放在本目录 `weights/`，或将 `weights/` 链接至已有同版本权重目录；来源与各文件哈希见上述清单。模型权重和数据集不随代码分发。原版与优化版分别使用各自代码目录，单卡运行时依次启动。

```bash
cd /root/private_data/hks/models/Kimi-VL-A3B-Instruct-Optimized-v1
# 获取并校验发布者的小型演示图片；此命令不下载权重：
bash run-scnet.sh python scripts/prepare_examples.py --download
# 复现历史eager首版，核验权重、处理器并启动服务：
HKS_PROFILE=baseline bash run.sh
```

在第二个终端执行功能与性能：

```bash
cd /root/private_data/hks/models/Kimi-VL-A3B-Instruct-Optimized-v1
bash run-scnet.sh python scripts/smoke_http.py --output results/functional-reproduce
bash run-scnet.sh python scripts/bench_http.py --smoke-summary results/functional-reproduce/summary.json --output results/performance-reproduce
```

评测使用固定 OCRBench 数据文件：`echo840/OCRBench`，版本 `92a54bd1384387c178d5a07140a2d85e0a3d12e1`，文件 `data/test-00000-of-00001.parquet`，67,576,988字节，SHA256 `d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5`。将该文件放至 `data/ocrbench/test.parquet`，或通过 `--data-path` 指定：

```bash
bash run-scnet.sh python scripts/eval_ocrbench.py --smoke-summary results/functional-reproduce/summary.json --output results/ocrbench-reproduce
python verify.py --selfcheck
```

Notebook 已是容器，不嵌套 Docker。使用新的结果目录；性能计时期间保持同卡无其他推理或评测任务。服务返回 OpenAI 兼容 `/v1/chat/completions`，支持单图、双图及SSE流式。

`run.sh`支持三个配置：`baseline`为历史eager首版；`short-optimized`为第7.1节的8192上下文/seq2解码图配置；默认`optimized`为32768上下文/seq8的完整负载配置。运行 `bash stop.sh` 仅停止本目录 `run.sh` 写入PID且命令行匹配的服务。`env_check.sh`执行权重、依赖与处理器检查。专用镜像已内置平台env时，可从镜像fallback读取；权重与数据仍从共享存储获取。

官方六档由控制器自行启动、预热及停止服务，执行前保持端口30004空闲：

```bash
bash bench.sh --processes 1 --launcher serve-optimized.sh
# 每档两次、第二次计入，共608条；不自动执行第二独立服务矩阵。
# 独立服务完整复跑需明确使用 --processes 2，共1216条。
```

## 5. 功能测试

两个独立服务进程各完成 **8/8**：纯文本算术、英文红色图识别、中文蓝色图识别、合成文字HKS2026 OCR、发布者示例图、红蓝双图顺序识别、固定请求重复一致性、流式与非流式输出一致性。流式检查包含 `[DONE]` 及正常结束，保留原始分块和拼接内容。

这些检查验证接口及适配功能，不替代公开集精度。示例图通过发布者SHA256核验；合成测试图由固定代码生成。记录分别位于 `results/scnet/native-vllm-v3-20261028/functional/` 和 `results/scnet/fresh-vllm-20261028/functional/`。

优化版v1的简单负载服务与完整矩阵服务也分别通过相同8项检查，记录位于 `results/optimization-20261102T020940Z/decode-graph/functional/` 和 `results/formal-20261102T021857Z/process1/functional/`。

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

历史eager首版的原始预测、逐题评分、请求参数清单及汇总位于 `results/scnet/ocrbench-full-20261028/`，精度范围为完整OCRBench。

## 7. 首次性能

发布者同一张演示PNG、固定英文描述问题，服务实际计入1131个输入token。并发1/2 × 输出上限32/128，共4档；每档8个计数请求，共32。每轮先覆盖全部形状预热8请求，再每档预热2请求，总预热16，均不计入32个实测请求。

| 并发 | 输出上限 | 请求数 | 输出token/s | 平均延迟s | 平均TTFT ms | 平均TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 14.244 | 2.244 | 287.58 | 63.13 |
| 1 | 128 | 8 | 15.088 | 8.481 | 287.31 | 64.52 |
| 2 | 32 | 8 | 25.996 | 2.459 | 438.83 | 65.18 |
| 2 | 128 | 8 | 29.200 | 8.765 | 439.32 | 65.55 |

计时为HTTP客户端发送、服务预填充/解码、流式接收的总范围，排除模型初始化。吞吐按实际 `usage.completion_tokens` 总量除以计时窗口；TTFT取首个非空内容到达时间，TPOT按 `(请求延迟−TTFT)/(实际输出token数−1)` 计算，不把SSE分块数当token数。输出上限为负载参数，不把截断的性能样本作为精度结果。本矩阵不冒充赛事完整性能协议，亦不声明相对其他后端的优化提升。

### 7.1 解码图优化：原简单负载对照

2026-11-02在同一张BW、同一vLLM发行版本上，复现历史简单负载作为当日eager对照，再测试解码图。两种配置均先全局预热8请求、每档预热2请求，各测一遍32条计数请求。`bench_http.py`、`smoke_http.py`、图像SHA、问题、采样、1131输入token和实际输出长度均一致；不存在并行精度流量。下表只比较相同简单负载，不外推完整六档的提升。

| 并发 | 输出上限 | 历史首次吞吐 token/s | 当日eager token/s | 解码图 token/s | 相对当日eager提升 |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 14.244 | 14.3268 | 53.8276 | +275.71% |
| 1 | 128 | 15.088 | 15.1446 | 73.4428 | +384.94% |
| 2 | 32 | 25.996 | 26.1346 | 77.1486 | +195.20% |
| 2 | 128 | 29.200 | 29.1672 | 117.6686 | +303.43% |

提升按 `(优化后输出吞吐/当日eager输出吞吐−1)×100%` 计算，吞吐由后端实际completion_tokens总数除以客户端计时窗口重算。两配置各8/8功能通过；32条计数响应逐条比较，文本、输入长度和输出长度全部相同。原始响应、启动参数、脚本快照与独立复算在 `results/optimization-20261102T020940Z/`，核验入口：

```bash
python scripts/verify_optimization.py --job optimization-20261102T020940Z --repeats 1
```

该对照每配置做一次带预热的计时。历史完整OCRBench 869/1000来自eager首版；本轮逐条一致的核验范围为上述32条简单性能样本。性能奖励是否认定，以主办方对首版基准及优化材料的审核为准。

### 7.2 官方六档：优化版v1

单BW、BF16、vLLM原生实现，使用第3章的32768上下文/seq8解码图配置。使用镜像内SGLang的 `vllm-chat` 压测客户端，每请求一张1080p随机JPEG，随机输入seed1，文本长度参数4096/16384，固定生成1024，客户端并发1/8/64。服务seed42，temperature=0、top_p=1，ignore_eos=true用于固定生成工作量。每档请求数为 `max(8,2C)`，连续测试两次，第二次计入下表。客户端每次另外发送1条32-token预热请求，计时窗口排除这些内部预热及模型初始化、数据准备。

随机文本采样使用基础词表0–163583，排除发布者的258个保留槽位（包含两个embedding越界ID），解码后重新编码检查范围，模型权重与tokenizer文件保持原版。该兼容处理避免随机文本误含图像占位符或越界ID，详见 `优化说明.md`。准备阶段的VMFault、占位符冲突与中断日志保留在相应历史任务目录，最终608条结果从统一修正协议重新执行。

| 文本长度参数 | 输出 | 并发 | 完成 | duration秒 | 请求/秒 | 输出token/秒 | 总token/秒 | mean TTFT毫秒 | p99 TTFT毫秒 | mean TPOT毫秒 | p99 TPOT毫秒 | mean E2E毫秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 1 | 8 | 106.0513 | 0.0754 | 77.2457 | 472.7525 | 465.3632 | 477.3246 | 12.5026 | 12.6920 | 13255.4867 |
| 4096 | 1024 | 8 | 16 | 53.1555 | 0.3010 | 308.2279 | 1886.1653 | 2441.3541 | 4717.6553 | 23.5609 | 25.8440 | 26544.1685 |
| 4096 | 1024 | 64 | 128 | 412.2075 | 0.3105 | 317.9758 | 1945.1853 | 137516.9715 | 187608.2867 | 24.2706 | 27.7142 | 162345.7750 |
| 16384 | 1024 | 1 | 8 | 118.0148 | 0.0678 | 69.4150 | 1276.3062 | 1346.8641 | 1349.6002 | 13.1027 | 13.1619 | 14750.9441 |
| 16384 | 1024 | 8 | 16 | 74.2922 | 0.2154 | 220.5345 | 4055.3104 | 6025.7130 | 13959.1547 | 30.3391 | 36.5867 | 37062.6615 |
| 16384 | 1024 | 64 | 128 | 657.9206 | 0.1946 | 199.2216 | 3663.6090 | 220067.8931 | 304759.9793 | 36.6021 | 43.8009 | 257511.8376 |

客户端processor统计的完整输入包含图像与聊天模板：4K三档分别为5228–5264、5226–5269、5219–5279 token；16K三档分别为17744–17836、17744–17838、17744–17879 token。服务最多驻留8条序列，客户端并发64包含排队时间，表中高并发TTFT/E2E反映该压力负载。

完整12格共608条计数请求，实际生成622592 token；每条后端输出长度均为1024，错误列表为空，文本非空。另有12条客户端内部预热和8条功能请求，独立保存的服务端Prometheus计数为628次成功请求、623008个生成token，等于622592个计数token + 384个内部预热token + 32个功能token。

原始结果、逐格日志、源码快照、服务端计数及独立核验位于 `results/formal-20261102T021857Z/`。独立核验重算吞吐与mean/p99 TTFT，并检查mean TPOT与mean E2E恒等式；p99 TPOT保留客户端统计值。核验入口：

```bash
python verify_formal.py --job formal-20261102T021857Z --processes 1
python scripts/audit_formal.py --job formal-20261102T021857Z
python scripts/export_formal_table.py --job formal-20261102T021857Z
```

任务包含加载、数据准备、功能与12格压测，总墙钟约63分钟，使用单张BW。第7.1节为同简单负载优化对照；本节给出优化版完整矩阵的实测值。

## 8. 复现与限制

历史eager首版在同一卡、同一镜像和Notebook内，停止首次服务后重新初始化独立服务进程，重复同样8项功能与4档性能。

| 并发 | 输出上限 | 首次输出token/s | 复跑输出token/s | 相对偏差 |
|---:|---:|---:|---:|---:|
| 1 | 32 | 14.244 | 14.115 | -0.91% |
| 1 | 128 | 15.088 | 15.127 | +0.26% |
| 2 | 32 | 25.996 | 26.340 | +1.32% |
| 2 | 128 | 29.200 | 29.283 | +0.28% |

历史eager首版的输出吞吐最大绝对偏差 **1.32%**。进一步对输出吞吐、平均请求延迟、TTFT、TPOT四类指标逐项复算，16个比较值中最大绝对偏差 **17.39%**（单并发、输出上限32的TTFT），全部≤20%，检查 **通过**。完整比较见 `results/scnet/reproduction-metric-audit-20261028.json`。精度实测来自历史首次服务进程；历史第二进程覆盖上述功能和性能。

`python verify.py --selfcheck` 核验已保存证据哈希、功能记录、精度计数及性能汇总。默认 `python verify.py` 不给出赛事精度验收通过结论，返回2。证据索引为 `results/evidence-manifest.json`，实际验收以主办方审核为准。

优化版v1的性能范围为两配置各一次带预热的简单负载对照，以及一个服务进程的完整官方六档（每档两次取第二次）。历史eager首版的两进程复跑结果按上表保留。本优化版使用独立代码目录和版本标记，原版目录与版本标记继续保留。

独立优化版目录的标准入口另经实际核对：20个权重运行文件哈希与索引通过，`run.sh`按默认optimized配置启动，算术请求返回4并正常结束，`stop.sh`停止本目录PID对应服务，显存释放。证据位于 `results/entry-20261102T032847Z/`。
