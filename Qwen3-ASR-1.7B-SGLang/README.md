# Qwen3-ASR-1.7B · BW × SGLang 技术报告

## 1. 简介

- [发布者模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)与[发布者代码](https://github.com/QwenLM/Qwen3-ASR)
- [SCNet模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen3-ASR-1.7B)
- [本轮SCNet镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)
- [SGLang原生ASR文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/supported-models/multimodal_language_models.mdx)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)：`Qwen3-ASR-1.7B-SGLang/`；原vLLM目录独立保留。

本版本在单张BW上使用SGLang原生 `Qwen3ASRForConditionalGeneration`、Qwen3音频处理器及转写路由，完成BF16离线语音识别、基础图执行优化、音频性能、完整test-clean与独立官方固定条件测试。模型归属T04，专项按T03/T04固定300条件独立执行。

| 验证项 | 结果 |
| --- | --- |
| 固定权重 | 11运行文件逐项SHA256通过，两片共4698521512字节 |
| 功能 | 模板修正后两候选及正式服务各12/12；重复与SSE文本一致 |
| 带预热短测 | 两候选各一遍，单并发图执行吞吐约提升283.16%；并发4冷编译记录保留 |
| 音频性能 | 并发1/4/8/16、两轮共128/128；另16预热，无错误/截断 |
| test-clean | 2620/2620，WER 1.7501%，CER 0.6396% |
| T03/T04固定条件 | 独立300/300请求，文本、反馈CSV、环境齐全 |
| 独立服务完整性能复现 / 官方验收 | null / null |

## 2. 环境

SCNet Notebook，华中一区【A区】。镜像 `appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1`，版本 `v1`；基础为 `image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`。单张BW64GB（物理显存68702699520字节）、TP1、BF16；CPU配额15核、内存上限63350767616字节、数学库固定4线程。共享文件存储400GiB，初检剩余约71GiB；根文件系统为overlay挂载，未单独核对系统盘配额。无外部服务映射，监听 `127.0.0.1:30129`。

Python3.10.12，HIP 6.3.26113，DTK `DTK-26.04-blas-nightly-0921`。组件原文见 `env.log`、`results/preflight-20261103T093739Z/environment.json`及补充环境记录。

| 组件 | 实际版本 |
| --- | --- |
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| triton | 3.6.0+dtk2604.torch2110.2609231415.g8cce4a |
| sglang | 0.5.18+dtk2604.torch2110.2609231937.gc3bbbd |
| sglang-kernel | 0.4.6.post1+dtk2604.torch2110.2609231956.gea1c29 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| aiter | 0.1.6+dtk2604.torch2110.2609242152.g36dc29 |
| vllm-hcu | 0.25.1+dtk2604.torch2110.2609272150.gee325d |
| librosa | 0.11.0 |
| soxr | 1.0.0 |
| soundfile | 0.13.1 |
| editdistance | 0.8.1 |
| numpy | 1.25.0 |
| pyarrow | 25.0.1 |
| regex | 2026.9.10 |
| openai | 2.30.0 |
| fastapi | 0.115.12 |
| python-multipart | 0.0.32 |

本轮使用镜像现有依赖，音频补充依赖从 `/opt/qwen3-asr-deps` 加载，没有新增安装或升级全局栈。镜像中的vllm-hcu是已有依赖；实际服务入口、模型实现、调度与解码路径均为SGLang，启动参数和源码快照随证据保存。

## 3. 适配与配置

模型采用24层音频编码器和28层Qwen3文本解码器。SGLang原生类复用 `Qwen3OmniMoeAudioEncoder` 音频编码器及dense `Qwen3ForCausalLM`；根据原处理器执行音频加载、重采样和特征处理，保留原始输入字节、特殊token及权重映射。实际模型、处理器、配置与转写源码在 `sources/native/`，哈希见预检记录。

首轮原生转写可用，chat/SSE返回400，原因是tokenizer没有自动加载固定权重中的 `chat_template.json`。`scripts/prepare_runtime.py` 校验该文件SHA256，逐字提取JSON内Jinja值到 `runtime/qwen3-asr.jinja`，验证空system与音频占位符渲染；`run.sh` 自动传入 `--chat-template`。每次启动重建相同内容，不修改权重、tokenizer或全局框架源码，也不需要人工patch。首轮仅2/12通过的失败记录保留在 `results/initial/`。

最终参数为BF16、TP1、上下文8192、最大运行请求32、显存比例0.60、CPU4线程；文本注意力Triton、音频注意力SDPA，关闭radix和多模态预处理缓存，关闭chunked prefill。decode graph捕获1/2/4/8/16，prefill图关闭；图捕获发生在请求计时前。

正式性能、精度和固定300条件使用 `/v1/chat/completions` 原生音频输入，temperature0、max_tokens1024、自然EOS、自动语种，不向请求输入参考答案或语种标签。原生 `/v1/audio/transcriptions` adapter会把请求温度0改成0.01、上限设为256；该接口按真实行为作功能验证，未用其结果冒充严格greedy/1024评测。

SGLang主动关闭时的子进程清理可能使API进程返回-9。停止脚本先验证cwd、命令、PID与进程组，再记录带时间和命令hash的停止意图，仅向API parent发送SIGTERM；owner只在本次启动之后存在匹配主动停止记录时接受该清理信号，意外-9仍失败。实际返回值保存在 `service-exit.json`。首个图候选在旧通用owner中的停止误判及failed状态完整保留；后续修正后的正式控制器实际完成并正常返回0。

## 4. 启动与复现

Notebook已是容器。将本目录部署至 `/root/private_data/hks/models/Qwen3-ASR-1.7B-SGLang`；权重默认路径为 `/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B`，可通过 `MODEL_PATH` 覆盖。固定运行文件及文件级来源修订见 `sources/weight-files.json`。脚本不下载权重、音频或数据集。

两条原始示范WAV放入本目录 `data/examples/`；test-clean源Parquet放入 `data/librispeech-source/test-clean.parquet`，原FLAC与trans文本放入 `data/LibriSpeech/test-clean/`；官方条件放入 `data/official/formal/conditions/qwen3_asr.jsonl` 并保持其相对音频目录结构。完整来源及校验见 `data/README.md`，支持 `LIBRI_PARQUET`、`LIBRI_TEST_CLEAN`、`OFFICIAL_CONDITIONS` 覆盖。已有共享存储可只读复用原ASR数据，优先使用本目录输入。

环境脚本优先读取工作根平台环境，缺失时使用镜像内环境或直接加载DTK/HYHAL；不依赖旧Notebook临时文件。`results/entry-preparation/`实际验证了缺失工作根平台脚本及旧vLLM数据目录条件下，独立目录音频与镜像环境可通过预检。

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh`、`env_check.sh` | 固定配置、11权重/2音频/原生类检查 | `env.log`、环境与hash证据 |
| `run.sh`、`stop.sh` | 自动提取模板、起服health/models、归属停止 | `server.log`、参数、PID、停止意图/实际退出 |
| `smoke.sh` | 在线12项音频功能 | `smoke.log`、完整响应/SSE分块、请求hash |
| `bench.sh` | 独立服务音频两轮128请求及CPU复算 | `bench/round1.jsonl`、`round2.jsonl`、summary与逐请求记录 |
| `accuracy.sh` | 独立服务完整test-clean2620评测和重算 | 原始预测、S/D/I/N、WER/CER、重核记录 |
| `official.sh` | 独立服务固定300条件和重核 | 300文本、反馈CSV、环境、原始响应 |
| `validate.sh` | 单个自有服务串行执行功能/性能/精度/官方条件后停止 | 时间戳完整任务目录及state |
| `verify.py` | 仅已保存证据检查与统计复算，非在线精度 | `verify_result.json`；通过0、真实失败非零 |
| `scripts/prepare_data.py`、`baselines/`、`data/README.md` | 固定输入验证、基线及协议说明 | 2620/300输入哈希与来源 |

```bash
cd /root/private_data/hks/models/Qwen3-ASR-1.7B-SGLang
bash env_check.sh
bash run.sh
# 另一个终端，对已启动服务执行：
bash smoke.sh
bash stop.sh
# 以下入口分别管理自己的新服务，串行运行：
bash bench.sh
bash accuracy.sh
bash official.sh
# 已保存交付证据核验，不再次推理：
python verify.py
```

完整合并验证可在端口空闲时运行 `bash validate.sh`。本轮实际采用该入口串行调用smoke/bench/accuracy/official；CPU音频审计另有原始日志，发布的bench入口已串联同一审计。`run.sh`前台运行，停止在另一终端执行。新运行写新的时间戳目录，当前任务指针及顶层日志可更新；已交付记录由 `results/release.json` 锁定，`bench/saved-release/`保留原发布轮次，避免重跑更新误触旧证据hash。缺失、损坏、请求失败或真实检查失败均返回非零。

## 5. 功能验证

模板修正后的eager、graph及正式服务各12/12。覆盖中英文chat音频、原生transcriptions、SSE输出、重复请求，以及4条并发请求；health/models实际通过。重复及SSE拼接文本与同服务普通chat一致，原始事件和 `[DONE]` 终止标记保留；transcriptions近greedy参数另行记录。

SSE测试上传的是完整音频文件，验证的是输出文本分块。未将其作为实时输入测试，也未迁移原vLLM的累计PCM重送适配器；连续PCM输入、WebSocket增量状态、时间戳及长录音分块不在本轮验证范围内。

## 6. 精度测试

### 6.1 LibriSpeech test-clean

使用完整2620条，源Parquet350452636字节，SHA256 `7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`，ModelScope镜像固定修订 `2a4a7918cd87e2f0e315e576b405ae7b7a65d83f`。全部FLAC字节及参考文本与源Parquet逐项一致。BF16、greedy、max1024、自然EOS、自动语种、并发4，参考文本只用于离线评分。

沿用原vLLM固定Whisper EnglishTextNormalizer，源码commit `86098128c0b4f24f0e2aa2994de830614b474227`。逐句保存原始响应、归一化预测、参考hash、词级S/D/I/N和字符错误数；WER为总词错误/总参考词，CER在同一归一化后排除空格。错误不剔除分母，发生请求错误或截断时测试技术状态失败。独立重新读取原响应、输入与实际请求体完成2620条重核。

| 项目 | 本轮SGLang |
| --- | --- |
| 完成 / 错误 / 截断 | 2620 / 0 / 0 |
| S / D / I / N | 660 / 199 / 69 / 53027 |
| 词错误 | 928 |
| WER | 1.7501% |
| 字符错误 / 参考字符 | 1473 / 230284 |
| CER | 0.6396% |

[发布者模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B#evaluation)公布的离线test-clean WER为1.63%，本轮增加0.1201个百分点；历史vLLM为1.6501%，本轮增加0.1000个百分点。历史结果归属原框架；发布者完整归一化等价性未确认，不猜测当前赛事容差或写成官方验收通过。中文演示音频属于功能检查，不是中文精度基准。

### 6.2 T03/T04官方条件

复用原条件的ID、顺序、音频和参数，AISHELL-1/test-clean/test-other各100条，另行发送300个新请求。BF16、greedy、max_new_tokens1024、seed42、自然EOS、自动语种；请求没有参考文本或语种标签。300/300成功处理、无请求错误或截断，300文本、原始响应、反馈CSV及环境均保留，并独立核对输入/输出及条件顺序。

300条处理成功表示请求和产物符合执行条件，不等于300条识别文本逐字正确或官方精度已验收。未复用test-clean既有预测或原vLLM的300结果作为新框架成绩。

## 7. 性能测试

### 7.1 带预热短测

同Notebook、同SGLang、同音频/采样/缓存/CPU配置，仅切换eager和decode graph；并发1/4，每格4条计数、两条示范音频预热，各候选一遍。两候选输出文本集合一致。保存首轮模板缺失与停止器误判记录，没有覆盖失败状态或重跑短测以替换不利数据。

| 并发 | 每候选计数请求 | eager音频秒/墙钟秒 | graph音频秒/墙钟秒 | 图相对eager |
| --- | --- | --- | --- | --- |
| 1 | 4 | 8.0796 | 30.9579 | +283.16% |
| 4 | 4 | 18.3488 | 7.0572 | -61.54% |

单并发改善约283.16%。graph并发4计时包含两次首次Triton编译2.31秒和2.51秒，原始耗时完整保留，不扣除编译、不声明该档优化收益。选择图配置继续正式两轮评估，不把单一短负载收益外推至全部负载。

### 7.2 音频性能

英文原WAV为15.051229秒/48000Hz，中文为4.203938秒/16000Hz，按原字节发送。并发1/4/8/16，每格N=max(8,2C)，各两轮，共128计数请求；每格每轮另2条预热，共16。自然EOS，输出预算1024，实际用量来自API usage，不强制填满token。

| 客户端并发 | 每轮请求 | 首轮音频秒/墙钟秒 | 第二轮音频秒/墙钟秒 | 第二轮平均逐请求RTF | 第二轮平均延迟秒 | 第二轮p99延迟秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 31.0400 | 30.9123 | 0.033235 | 0.310728 | 0.480481 |
| 4 | 8 | 60.3029 | 74.1626 | 0.048200 | 0.436549 | 0.726879 |
| 8 | 16 | 83.8655 | 85.9744 | 0.078796 | 0.729598 | 1.277362 |
| 16 | 32 | 166.0065 | 168.8988 | 0.079655 | 0.701755 | 1.226326 |

音频吞吐＝累计输入音频秒/计时墙钟秒；平均RTF先逐请求计算耗时/音频秒再取均值，未用两个均值相除代替。墙钟覆盖线程池生命周期、HTTP编码/传输/服务排队/推理及结果解析；输入读盘/base64提前完成，原始文件在计时批次结束后写入。逐请求记录可独立重算吞吐、RTF、平均及尾部延迟。

报告第二轮，首轮完整保留。同服务两轮吞吐最大偏差22.98%（并发4）；本轮未做另一独立服务的完整性能复跑，不据此宣称通过独立环境20%复现门槛。客户端并发与运行上限32分别记录，端到端延迟包含排队及处理开销；没有估算扣除排队，也未执行608条生成矩阵。

历史vLLM同两条示范音频第二轮吞吐为32.20/53.63/61.56/68.86，仅作跨框架、跨日期历史参考；其差异不计作本次同框架优化奖励。两条示范音频协议不外推为独立大样本或真实连续输入业务性能。

## 8. 复现与限制

### 8.1 复现范围与限制

本轮在所列镜像Notebook内完成原生加载、修正后两候选和正式服务功能、一遍候选短测、一个最终服务的128音频性能、完整test-clean及独立300条件。没有另建Notebook重复完整矩阵；单服务两轮不等于独立服务复现。精度、接口及编译开销限制分别见第5—7章。

`run.sh`自动准备模板和最终参数，模型/框架全局源码不需要手动改动；固定输入准备后可按第4章命令复跑。`verify.py`只核验保存证据及统计一致性，不调用模型或替代在线精度。官方验收和未做的独立完整复现单独记为null；检查通过返回0，真实缺失、损坏或检查失败返回非零。`scripts/check_delivery_cpu.py`在隔离副本实际验证：完整证据返回0、缺少release返回1、压缩资产损坏返回1、可变日志更新返回0；检查不调用模型，结果见 `results/delivery-cpu-checks.json`。

### 8.2 代码与原始数据

- `results/preflight-20261103T093739Z/`：11文件权重校验、音频校验、原生源码hash和实际环境。
- `results/initial/`：模板未加载时的2/12功能及400失败、请求/响应与停止记录。
- `results/eager-template/`、`results/screen-20261103T095044Z/graph/`：两候选功能、短测、预热、源码；原screen停止器failed状态保留，选择依据另存 `selection.json`。
- `results/formal-20261103T100245Z/`：完整启动参数/源码、12功能、128逐请求及16预热、2620预测与重核、300原响应/文本/CSV/环境，以及主动停止和实际退出。
- `bench/`：两轮JSONL、summary、逐请求/预热记录及独立CPU复算；`saved-release/`固定保存本版本轮次。
- `results/test-clean-inputs.json`、`official-inputs.json`：原始输入hash与条件映射；数据和完整HTTP请求体按hash保存在共享 `data/` 指定位置，通过固定输入和代码可重建同一请求。
- `verify_result.json`、`submission-manifest.json`：本版本保存证据检查结果及发布字节清单。

原始记录位于 `/root/private_data/hks/models/Qwen3-ASR-1.7B-SGLang/`。较大的JSON/JSONL可用 `file-zlib-v1` 引用和内容hash去重，单资产片≤2MiB；`evidence_io.py`透明恢复原字节，映射及字节对照在 `sources/evidence-view-map.json` 和 `evidence-view-validation.json`。权重、音频数据集、缓存和凭据不入库，原始失败记录保留。
