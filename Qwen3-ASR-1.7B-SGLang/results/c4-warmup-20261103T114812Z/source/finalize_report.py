"""Render the fixed eight-chapter report exclusively from completed evidence."""
from pathlib import Path
from evidence_io import read_json as read
ROOT=Path(__file__).resolve().parents[1]
release=read(ROOT/'results/release.json');job=ROOT/'results'/release['formal'];assert read(job/'state.json')['stage']=='completed'
env=read(ROOT/'results'/release['preflight']/'environment.json');selection=read(ROOT/'results/selection.json')
bench=read(job/'bench/summary.json');accuracy=read(job/'accuracy/summary.json');official=read(job/'official/summary.json')
assert bench['counted_requests']==128 and accuracy['complete'] and official['successful']==300
packages={**env['packages'],**read(ROOT/'results/vendor-packages.json'),**read(ROOT/'results/extra-environment.json')['packages']}
keys=['torch','triton','sglang','sglang-kernel','transformers','tokenizers','aiter','vllm-hcu','librosa','soxr','soundfile','editdistance','numpy','pyarrow','regex','openai','fastapi','python-multipart']
versions='\n'.join(f'| {k} | {packages[k]} |' for k in keys if k in packages)
short='\n'.join(f"| {e['concurrency']} | {e['requests']} | {e['audio_seconds_per_wall_second']:.4f} | {g['audio_seconds_per_wall_second']:.4f} | {(g['audio_seconds_per_wall_second']/e['audio_seconds_per_wall_second']-1)*100:+.2f}% |" for e,g in zip(selection['cells']['eager'],selection['cells']['graph']))
first={r['concurrency']:r for r in bench['cells'] if r['round']==1};second=[r for r in bench['cells'] if r['round']==2]
table='\n'.join(f"| {r['concurrency']} | {r['requests']} | {first[r['concurrency']]['audio_seconds_per_wall_second']:.4f} | {r['audio_seconds_per_wall_second']:.4f} | {r['mean_request_rtf']:.6f} | {r['mean_latency_seconds']:.6f} | {r['p99_latency_seconds']:.6f} |" for r in second)
deviation=max(abs(r['audio_seconds_per_wall_second']/first[r['concurrency']]['audio_seconds_per_wall_second']-1)*100 for r in second)
props=read(ROOT/'results/audio-properties.json')['files'];en=next(p for p in props if p['file']=='asr_en.wav');zh=next(p for p in props if p['file']=='asr_zh.wav')
text=f'''# Qwen3-ASR-1.7B · BW × SGLang 技术报告

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
| test-clean | {accuracy['completed']}/2620，WER {accuracy['wer_percent']:.4f}%，CER {accuracy['cer_percent']:.4f}% |
| T03/T04固定条件 | 独立{official['successful']}/300请求，文本、反馈CSV、环境齐全 |
| 独立服务完整性能复现 / 官方验收 | null / null |

## 2. 环境

SCNet Notebook，华中一区【A区】。镜像 `{env['image']}`，版本 `v1`；基础为 `image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`。单张BW64GB（物理显存{env['memory'][1]}字节）、TP1、BF16；CPU配额15核、内存上限{env['memory_limit']}字节、数学库固定4线程。共享文件存储400GiB，初检剩余约71GiB；根文件系统为overlay挂载，未单独核对系统盘配额。无外部服务映射，监听 `127.0.0.1:30129`。

Python3.10.12，HIP {env['hip']}，DTK `{env['dtk'].splitlines()[0]}`。组件原文见 `env.log`、`results/{release['preflight']}/environment.json`及补充环境记录。

| 组件 | 实际版本 |
| --- | --- |
{versions}

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
| S / D / I / N | {accuracy['substitutions']} / {accuracy['deletions']} / {accuracy['insertions']} / {accuracy['reference_words']} |
| 词错误 | {accuracy['word_errors']} |
| WER | {accuracy['wer_percent']:.4f}% |
| 字符错误 / 参考字符 | {accuracy['char_errors']} / {accuracy['reference_chars']} |
| CER | {accuracy['cer_percent']:.4f}% |

[发布者模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B#evaluation)公布的离线test-clean WER为1.63%，本轮增加{accuracy['wer_percent']-1.63:.4f}个百分点；历史vLLM为1.6501%，本轮增加{accuracy['wer_percent']-1.6501:.4f}个百分点。历史结果归属原框架；发布者完整归一化等价性未确认，不猜测当前赛事容差或写成官方验收通过。中文演示音频属于功能检查，不是中文精度基准。

### 6.2 T03/T04官方条件

复用原条件的ID、顺序、音频和参数，AISHELL-1/test-clean/test-other各100条，另行发送300个新请求。BF16、greedy、max_new_tokens1024、seed42、自然EOS、自动语种；请求没有参考文本或语种标签。300/300成功处理、无请求错误或截断，300文本、原始响应、反馈CSV及环境均保留，并独立核对输入/输出及条件顺序。

300条处理成功表示请求和产物符合执行条件，不等于300条识别文本逐字正确或官方精度已验收。未复用test-clean既有预测或原vLLM的300结果作为新框架成绩。

## 7. 性能测试

### 7.1 带预热短测

同Notebook、同SGLang、同音频/采样/缓存/CPU配置，仅切换eager和decode graph；并发1/4，每格4条计数、两条示范音频预热，各候选一遍。两候选输出文本集合一致。保存首轮模板缺失与停止器误判记录，没有覆盖失败状态或重跑短测以替换不利数据。

| 并发 | 每候选计数请求 | eager音频秒/墙钟秒 | graph音频秒/墙钟秒 | 图相对eager |
| --- | --- | --- | --- | --- |
{short}

单并发改善约283.16%。graph并发4计时包含两次首次Triton编译2.31秒和2.51秒，原始耗时完整保留，不扣除编译、不声明该档优化收益。选择图配置继续正式两轮评估，不把单一短负载收益外推至全部负载。

### 7.2 音频性能

英文原WAV为{en['duration']:.6f}秒/{en['samplerate']}Hz，中文为{zh['duration']:.6f}秒/{zh['samplerate']}Hz，按原字节发送。并发1/4/8/16，每格N=max(8,2C)，各两轮，共128计数请求；每格每轮另2条预热，共16。自然EOS，输出预算1024，实际用量来自API usage，不强制填满token。

| 客户端并发 | 每轮请求 | 首轮音频秒/墙钟秒 | 第二轮音频秒/墙钟秒 | 第二轮平均逐请求RTF | 第二轮平均延迟秒 | 第二轮p99延迟秒 |
| --- | --- | --- | --- | --- | --- | --- |
{table}

音频吞吐＝累计输入音频秒/计时墙钟秒；平均RTF先逐请求计算耗时/音频秒再取均值，未用两个均值相除代替。墙钟覆盖线程池生命周期、HTTP编码/传输/服务排队/推理及结果解析；输入读盘/base64提前完成，原始文件在计时批次结束后写入。逐请求记录可独立重算吞吐、RTF、平均及尾部延迟。

报告第二轮，首轮完整保留。同服务两轮吞吐最大偏差{deviation:.2f}%（并发4）；本轮未做另一独立服务的完整性能复跑，不据此宣称通过独立环境20%复现门槛。客户端并发与运行上限32分别记录，端到端延迟包含排队及处理开销；没有估算扣除排队，也未执行608条生成矩阵。

历史vLLM同两条示范音频第二轮吞吐为32.20/53.63/61.56/68.86，仅作跨框架、跨日期历史参考；其差异不计作本次同框架优化奖励。两条示范音频协议不外推为独立大样本或真实连续输入业务性能。

## 8. 复现与限制

### 8.1 复现范围与限制

本轮在所列镜像Notebook内完成原生加载、修正后两候选和正式服务功能、一遍候选短测、一个最终服务的128音频性能、完整test-clean及独立300条件。没有另建Notebook重复完整矩阵；单服务两轮不等于独立服务复现。精度、接口及编译开销限制分别见第5—7章。

`run.sh`自动准备模板和最终参数，模型/框架全局源码不需要手动改动；固定输入准备后可按第4章命令复跑。`verify.py`只核验保存证据及统计一致性，不调用模型或替代在线精度。官方验收和未做的独立完整复现单独记为null；检查通过返回0，真实缺失、损坏或检查失败返回非零。`scripts/check_delivery_cpu.py`在隔离副本实际验证：完整证据返回0、缺少release返回1、压缩资产损坏返回1、可变日志更新返回0；检查不调用模型，结果见 `results/delivery-cpu-checks.json`。

### 8.2 代码与原始数据

- `results/{release['preflight']}/`：11文件权重校验、音频校验、原生源码hash和实际环境。
- `results/initial/`：模板未加载时的2/12功能及400失败、请求/响应与停止记录。
- `results/eager-template/`、`results/{selection['screen']}/graph/`：两候选功能、短测、预热、源码；原screen停止器failed状态保留，选择依据另存 `selection.json`。
- `results/{release['formal']}/`：完整启动参数/源码、12功能、128逐请求及16预热、2620预测与重核、300原响应/文本/CSV/环境，以及主动停止和实际退出。
- `bench/`：两轮JSONL、summary、逐请求/预热记录及独立CPU复算；`saved-release/`固定保存本版本轮次。
- `results/test-clean-inputs.json`、`official-inputs.json`：原始输入hash与条件映射；数据和完整HTTP请求体按hash保存在共享 `data/` 指定位置，通过固定输入和代码可重建同一请求。
- `verify_result.json`、`submission-manifest.json`：本版本保存证据检查结果及发布字节清单。

原始记录位于 `/root/private_data/hks/models/Qwen3-ASR-1.7B-SGLang/`。较大的JSON/JSONL可用 `file-zlib-v1` 引用和内容hash去重，单资产片≤2MiB；`evidence_io.py`透明恢复原字节，映射及字节对照在 `sources/evidence-view-map.json` 和 `evidence-view-validation.json`。权重、音频数据集、缓存和凭据不入库，原始失败记录保留。
'''
# REPORT_EVIDENCE_LINKS: fixed release URLs also work from the standalone report.
version='qwen3-asr-bw-sglang-v1.0.1'
web='https://developer.sourcefind.cn/codes/zhengke/gxgm-hks/-/'
def evidence(label,path,directory=False):
    kind='tree' if directory else 'blob'
    return f'[{label}]({web}{kind}/{version}/Qwen3-ASR-1.7B-SGLang/{path})'
formal=f"results/{release['formal']}"
links={
'## 3. 适配与配置': '环境证据：'+evidence('实际环境JSON',f"results/{release['preflight']}/environment.json")+'、'+evidence('权重校验',f"results/{release['preflight']}/weights-check.json")+'、'+evidence('环境日志','env.log')+'。',
'## 6. 精度测试': '功能证据：'+evidence('正式服务12项汇总',formal+'/functional/summary.json')+'及'+evidence('逐项原始响应与SSE记录',formal+'/functional/',True)+'。',
'### 6.2 T03/T04官方条件': '精度证据：'+evidence('test-clean评分汇总',formal+'/accuracy/summary.json')+'、'+evidence('2620条逐句预测与原始响应',formal+'/accuracy/predictions.jsonl')+'、'+evidence('独立重核结果',formal+'/accuracy/rescore.json')+'、'+evidence('固定输入SHA映射','results/test-clean-inputs.json')+'。较大的JSONL采用可还原原字节的证据引用，读取方法见第8.2节。',
'## 7. 性能测试': '专项结果文件如下，可直接核对本框架的提交产物：\n\n| 产物 | 文件或目录 |\n| --- | --- |\n| 专项材料索引 | {INDEX} |\n| 300条逐条识别文本 | {OUTPUTS} |\n| 官方逐项反馈表 | {CSV} |\n| 可读环境说明 | {TXT} |\n| 完整环境参数与来源 | {ENV} |\n| 执行汇总与独立核验 | {SUMMARY}、{RESCORE} |\n| 原始响应及固定条件映射 | {PRED}、{INPUTS} |'.format(INDEX=evidence('专项材料README',formal+'/official/README.md'),OUTPUTS=evidence('outputs/（300个TXT）',formal+'/official/outputs/',True),CSV=evidence('反馈表.csv',formal+'/official/反馈表.csv'),TXT=evidence('环境说明.txt',formal+'/official/环境说明.txt'),ENV=evidence('environment.json',formal+'/official/environment.json'),SUMMARY=evidence('summary.json',formal+'/official/summary.json'),RESCORE=evidence('rescore.json',formal+'/official/rescore.json'),PRED=evidence('predictions.jsonl',formal+'/official/predictions.jsonl'),INPUTS=evidence('official-inputs.json','results/official-inputs.json')),
'### 7.2 音频性能': '优化对比证据：'+evidence('eager短测汇总','results/eager-template/short/summary.json')+'、'+evidence('graph短测汇总',f"results/{selection['screen']}/graph/short/summary.json")+'、'+evidence('候选选择与原始失败状态','results/selection.json')+'。',
'## 8. 复现与限制': '正式性能证据：'+evidence('首轮JSONL','bench/saved-release/round1.jsonl')+'、'+evidence('第二轮JSONL','bench/saved-release/round2.jsonl')+'、'+evidence('完整汇总','bench/saved-release/summary.json')+'、'+evidence('128条逐请求记录',formal+'/bench/requests.jsonl')+'、'+evidence('16条预热记录',formal+'/bench/warmups.jsonl')+'、'+evidence('独立CPU复算',formal+'/bench/independent-check.json')+'。',
}
for heading,block in links.items():
    assert text.count(heading)==1,heading
    text=text.replace(heading,block+'\n\n'+heading)
text+='\n证据文件导航：'+evidence('功能/性能/精度/专项完整任务目录',formal+'/',True)+'、'+evidence('本版自检结果','verify_result.json')+'、'+evidence('隔离故障负测','results/delivery-cpu-checks.json')+'、'+evidence('发布文件清单','submission-manifest.json')+'、'+evidence('原始字节映射','sources/evidence-view-map.json')+'。\n\n压缩证据由 '+evidence('evidence_io.py','scripts/evidence_io.py')+' 自动校验并还原；可在模型目录执行以下只读命令查看原始JSONL（不调用模型）：\n\n```bash\npython - <<\'PY\'\nfrom scripts.evidence_io import read_text\nprint(read_text(\"'+formal+'/accuracy/predictions.jsonl\"))\nPY\n```\n'

assert [s for s in text.splitlines() if s.startswith('## ')]==['## 1. 简介','## 2. 环境','## 3. 适配与配置','## 4. 启动与复现','## 5. 功能验证','## 6. 精度测试','## 7. 性能测试','## 8. 复现与限制']
(ROOT/'README.md').write_text(text,encoding='utf-8',newline='\n');print('FIXED_EIGHT_CHAPTER_REPORT_RENDERED')
