"""Render the fixed eight-chapter report from completed experiment evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
release=read(ROOT/'results/release.json');screen=ROOT/'results'/release['screen'];job=ROOT/'results'/release['formal']
summary=read(job/'summary.json');assert summary['complete']
env=read(ROOT/'results'/release['preflight']/'environment.json');platform=read(ROOT/'results/platform-environment.json')
weights=read(ROOT/'results'/release['preflight']/'weights-check.json')
a=read(screen/'eager/short/summary.json')['cells'];b=read(screen/'graph/short/summary.json')['cells']
short='\n'.join(f"| {x['max_concurrency']} | 512 | 128 | 8 | {x['output_throughput']:.3f} | {y['output_throughput']:.3f} | {(y['output_throughput']/x['output_throughput']-1)*100:.2f}% |" for x,y in zip(a,b))
packages={**env['packages'],**platform['extra_packages']}
table='\n'.join(f'| {k} | {v} |' for k,v in packages.items())
perf='\n'.join(f"| {r['random_input_len']} | {r['max_concurrency']} | {r['completed']} | {r['duration']:.3f} | {r['output_throughput']:.3f} | {r['mean_ttft_ms']:.3f} | {r['mean_tpot_ms']:.3f} |" for r in summary['round2'])
tail='\n'.join(f"| {r['random_input_len']} | {r['max_concurrency']} | {r['request_throughput']:.4f} | {r['total_throughput']:.3f} | {r['p99_ttft_ms']:.3f} | {r['p99_tpot_ms']:.3f} | {r['mean_e2e_latency_ms']:.3f} |" for r in summary['round2'])
weight_bytes=sum(f['bytes'] for f in weights['files'] if f['path'].endswith('.safetensors'))
report=f'''# MiMo-V2.6-Distill-Qwen-9B · BW × vLLM 技术报告

## 1. 简介

- [发布者模型与固定版本](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/tree/2367e865d009c13ac81713a2878291d33ab28177)
- [发布者聊天模板](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/chat_template.jinja)
- [本轮 SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2117085616644177921)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)：`MiMo-V2.6-Distill-Qwen-9B-vLLM/`；原 SGLang 目录独立保留。

MiMo-V2.6-Distill-Qwen-9B 是基于 Qwen3.5-9B 的 Agent SFT 模型。本版本使用镜像原生 vLLM Qwen3.5 混合全注意力/GDN 与视觉实现，完成服务接口适配、基础图执行优化和 T01 纯文本性能测试。

| 验证项 | 本轮结果 |
| --- | --- |
| 固定运行文件 | 15 文件逐项校验；4 个 BF16 分片共 {weight_bytes} 字节 |
| 功能 | 初始服务、两筛选候选、正式服务各 17/17 |
| 基础优化 | eager 与 decode graph 各一遍带预热短测，同框架同负载对照 |
| 正式性能 | 六档各两次，共 608/608，实际输出 622592 token，第二次计分 |
| 精度 | 本版本未执行新的在线精度评估；历史 SGLang 分数不迁移 |
| 独立服务完整性能复现 | null；完整矩阵仅一个最终服务 |
| 官方精度 / 验收 | null / null |

## 2. 环境

SCNet Notebook，{platform['region']}；单张 BW 64GB（物理显存 {env['memory'][1]} 字节），TP1、BF16。CPU 配额 15 核，内存上限 {env['memory_limit']} 字节，数学库固定 4 线程。共享文件存储配额 400G，预检时剩余约 72G。容器根文件系统为 overlay；不将底层 df 挂载视图误写成 Notebook 独立系统盘配额。无额外外部服务映射，API 监听 `127.0.0.1:30128`。

镜像 `{platform['image']}`，版本 `v1.0.0`；Python 3.10.12，HIP {env['hip']}，DTK `{platform['dtk'].splitlines()[0]}`。环境原文见 `env.log`、`results/platform-environment.json`。

| 组件 | 实测版本 |
| --- | --- |
{table}

使用镜像现有组件与 `/opt/qwen3-asr-deps` API 依赖，没有新增安装包或覆盖全局框架。镜像提供运行栈，本模型代码目录提供启动与验证入口；不将镜像名视为 MiMo 已适配的证明。本轮在该镜像 Notebook 内实际完成下述验证，未另建实例重复完整矩阵。

## 3. 适配与配置

模型声明 `Qwen3_5ForConditionalGeneration`：32 层中 24 层线性 GDN、8 层全注意力，dense MLP；保留原始视觉塔、词表、模板与权重映射。原生类与 `mimo` reasoning/tool parser 先经导入检查，再完成全部权重加载及真实 health/models 检查。当前 `mimo` 注册映射至 Qwen3 引擎解析器，与模板的 `<think>`、`<tool_call><function=...>` 格式对应，thinking 开关与整数工具参数已在线验证。

权重声明 `mamba_ssm_dtype=float32`，当前原生缓存 `auto` 会跟随模型 dtype，因此 `run.sh` 显式设置 `--mamba-ssm-cache-dtype float32`；权重及卷积缓存仍按原生 BF16 路径。未照搬 SGLang 的 KV 布局或 Mamba 槽位开关，也未修改全局 vLLM 源码。镜像原生视觉路径日志记录 tanh GELU 回退为非近似 GELU，本轮保留该行为；图像功能通过不等于完整视觉精度等价。

最终为上下文 32768、最大运行序列 64、批 token 上限 8192、显存比例 0.80；关闭 prefix cache 和多模态预处理缓存，图像上限 2、视频上限 0。采用 `mode=0`、`FULL_DECODE_ONLY`，捕获批次 1/2/4/8/16/32/64，prefill 不使用该解码图。最终服务日志的 KV 容量为 {summary['kv_capacity_tokens']} token。

保留发布者 `generation_config.json`：do_sample=true、temperature 0.6、top_p 0.95、top_k 20。功能及性能请求显式采用 greedy、top_p=1、seed42；性能客户端另采用其默认 `ignore_eos=true` 以测量固定输出长度，功能请求正常按EOS结束。这些是测试覆盖设置，不称为发布者默认。

压测使用现有 SGLang 客户端的 vLLM `/v1/completions` 后端。首轮预热遇到客户端直接索引空 `choices[0]`，无法接收合法 usage-only SSE 分块，原始失败保留在 `results/initial/`。`scripts/bench_native.py` 严格核对原函数后，仅在当前客户端进程内修正空 choices 和独立 usage 处理，移除“以输出预算代替实际输出数”的回退；缺 usage 及 HTTP 失败仍返回失败。CPU 回归验证合法 usage-only、实际33与预算128不同、缺 usage 失败及 HTTP500 保留，未发送模型请求。源码原文、实际修正逻辑与检查结果随目录提供，不需要手动 patch 全局框架。

## 4. 启动与复现

部署本目录至共享工作根 `/root/private_data/hks` 的模型独立目录。环境脚本优先读取工作根的 `platform/scnet-notebook/`，缺失时读取指定镜像内 `/opt/kimi-vl-vllm/platform/env-scnet.sh`。固定版本权重置于 `/root/private_data/hks/weights/MiMo-V2.6-Distill-Qwen-9B`，或在 `env.sh` 支持的 `MODEL_PATH` 中指定完整快照；预检核对固定版本15个运行文件的长度和哈希，不自动下载。已有 upstream-source.json 时核对其版本；直接提供完整发布者快照时，预检依据全部固定哈希生成本目录来源证据，不要求快照含有自定义来源文件。纯文本性能输入和功能图片由脚本生成，无外部评测数据下载。

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh`、`env_check.sh` | 统一配置；核对15个固定文件与原生实现 | `env.log`、预检JSON |
| `run.sh`、`scripts/service.py` | 传入FP32状态与最终图配置，起服并检查health/models | `server.log`、`command.json`、归属PID |
| `smoke.sh`、`scripts/smoke.py` | 在线17项功能 | `smoke.log`、原始请求/响应/SSE分块 |
| `stop.sh` | 验证目录、命令及进程组归属后停止 | 停止日志 |
| `bench.sh`、`scripts/formal.py` | 新建最终配置服务，功能、容量探测、完整矩阵、停止及CPU汇总 | 独立任务目录、12格JSONL及逐请求计时 |
| `scripts/audit_formal.py`、`bench/` | 从原始请求独立核对已完成矩阵，导出第二轮及相对路径索引 | `round1.jsonl`、`round2.jsonl`、`summary.json` |
| `verify.py` | 只读保存证据检查，非在线精度评估 | `verify_result.json`；成功0、真实失败非零 |
| `baselines/`、`data/README.md` | 真实历史参考与固定输入协议说明 | 来源与协议说明 |
| `scripts/bench_native.py`、`sources/benchmark_openai_completions.original.py` | 自动重放进程局部客户端兼容修正 | 原始输入、逐请求输出和计时 |

```bash
cd /root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B-vLLM
bash env_check.sh
bash run.sh
# 另一个终端，对已启动服务执行：
bash smoke.sh
bash stop.sh
# 独立完整验证，自动管理自己的新服务：
bash bench.sh
# 只读核验本版本已保存的交付证据，不重测精度：
python verify.py
```

`run.sh` 前台保持运行；smoke/stop 在另一终端执行，端口空闲后再启动独立矩阵。`results/formal-current.json` 指向新任务，`state.json` 与退出状态用于判断完整完成。`bench.sh` 在服务停止后调用 `scripts/audit_formal.py`，从逐请求证据重算并写入 `bench/`；本轮矩阵与这一步CPU后处理分开执行并分别保留日志。`verify.py` 使用 `results/release.json` 锁定已交付记录，同时核对当前矩阵及 `bench/`；顶层日志、当前任务指针和当前bench输出允许重跑更新，原发布两轮另存于 `bench/saved-release/`。当前矩阵未完成或检查失败仍返回非零。不能把保存证据校验解释为一次新运行或精度评估。标准启动、功能、完整矩阵与停止均有实际日志。

## 5. 功能验证

17 项覆盖算术、中文、多轮、thinking、重复、普通流式、thinking流式、红色单图、中文蓝色单图、HKS2026 OCR、红蓝双图、整数工具参数、工具结果回填及4条并发算术请求。另核对 `/health` 与 `/v1/models`。思考请求检查 reasoning 和最终回答，非思考请求检查无多余 reasoning；流式保留原始分块及 `[DONE]`。工具参数实测为整数2和3，结果5可正常回填。

初始服务、eager筛选、graph筛选、正式服务各17/17。工具回填为合成功能探测，不是 SWE-bench、OfficeQA 或任何 Agent 基准成绩。原始输入、响应、usage、结束原因与功能图片随对应任务目录保留。

## 6. 精度测试

本 vLLM 版本没有执行新的在线精度评估，`official_accuracy_pass=null`。赛事最低参考仍为 Qwen3.5-9B，未完成同协议对照，不能据功能或 CPU 证据核验填报精度通过。

历史 SGLang 的 GSM8K179/200、MMLU160/200仅为原框架小样本公开自评，不归属本 vLLM 版本，也不直接对照 Qwen3.5 的 MMLU-Pro 等不同口径。发布者公布的 Agent 基准同样需要匹配任务、采样、工具与评分协议。本轮没有执行 Agent 基准或其他在线精度测试。

## 7. 性能测试

### 7.1 基础优化短测

同 Notebook、同 vLLM、BF16权重、FP32状态、32768上下文、64运行上限及相同缓存配置，唯一候选差异为 eager 与 decode graph。原生输入512 token、输出128、并发1/4，每档8个计时请求；每候选各一遍，额外分别1/4个32-token客户端预热。初始化与图捕获不计入计时，实际输出数来自API usage。

| 并发 | 输入 | 输出 | 计时请求 | eager token/s | graph token/s | 同框架提升 |
| --- | --- | --- | --- | --- | --- | --- |
{short}

选择 graph。历史 SGLang 512/128短测与正式性能按独立来源保留，不能将跨框架、跨日期差异算作本框架优化奖励；上述短负载收益不外推为长上下文矩阵收益。

### 7.2 正式矩阵

T01纯文本 `random-ids`，seed1，输入4096/16384 token-ID，输出1024，客户端并发1/8/64。每档N=max(8,2C)，各两次，共608计数请求，第二次304请求计分；另有12个32-token内部预热。原随机ID生成器保持不变，仅在数据构造时选择ID列表，直接发送vLLM原生completions，不加入图片或聊天模板。16K容量探测的服务端usage为16384输入、32输出，16384+1024小于32768；不将原生文本口径外推到图文或对话输入。

| 原生输入token | 客户端并发 | 第二次请求数 | 墙钟秒 | 输出token/s | 平均TTFT ms | 平均TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
{perf}

| 原生输入token | 客户端并发 | 请求/s | 总token/s | p99 TTFT ms | p99 TPOT ms | 平均E2E ms |
| --- | --- | --- | --- | --- | --- | --- |
{tail}

保存实际请求IDs、逐请求可见输出、usage计数、首内容延迟、E2E及客户端ITL记录，计时窗口内不写原始文件，结束后落盘；客户端处理本身仍是端到端开销的一部分。逐格独立重算计数、输出吞吐、平均TTFT/TPOT/E2E。TPOT及分位数按原客户端定义报告，不把SSE分块数当token数。

客户端并发与 `max_num_seqs=64` 是不同的量；实际驻留还受 {summary['kv_capacity_tokens']} token 的 KV 预算和调度约束。16K/c64第二次测试的一次只读metrics采样记录7条运行、57条因capacity等待、KV使用率98.95%，原文为 `during-16k-c64.prom`；这是单次状态，不作为固定驻留上限或峰值。TTFT从请求发送计至首个非空内容，包含排队、输入处理、预填充及首token生成和接收；E2E另包括后续全部解码和传输。c64的这两类延迟均为端到端压力结果，不解释为64条长请求同时驻留时的纯计算延迟，也不扣除未单独测量的排队时间。两轮属于同一服务，独立服务完整性能复现为null。服务端计数原文在任务 `.prom` 文件中，独立交叉核对见 `bench/summary.json`。

## 8. 复现与限制

### 8.1 复现范围与限制

本轮完成固定镜像 Notebook 上的原生加载、四服务功能、同框架一次候选短测与一个最终服务的608请求矩阵。候选之间用于执行策略筛选，不是同一最终配置的独立服务完整复现；单服务各档两次也不等于另建Notebook复验。

模型推理由 `run.sh` 自动传入所需状态缓存和图配置，模型执行源码未作全局补丁；客户端兼容修正在 `bench.sh` 调用的进程内自动执行并核对原文，不需要额外手动修改环境。`python verify.py` 只核验本版本保存证据，成功返回0，缺文件、损坏、请求失败或真实检查失败返回非零；官方精度与验收状态单独记录。第6章精度限制和第7.2节排队口径保持有效，不能用功能或CPU校验代替精度评估。

### 8.2 代码与原始数据

- `results/{release['preflight']}/`：权重检查、原生导入与环境原文；`env.log`为对应原始环境记录及平台补充。
- `results/initial/`：首个服务的17项功能、客户端预热失败、原始日志和运行源码，未计为成功短测。
- `results/client-compat-v2/`：零模型请求的客户端协议回归；客户端协议诊断和失败记录亦保留。
- `results/{release['screen']}/`：两候选的功能、预热、16条计时请求、输出与原始计时。
- `results/{release['formal']}/`：最终配置、17功能、容量探测、两轮矩阵、原始请求、逐请求计时、metrics、停止日志及运行源码。
- `bench/`：当前两轮JSONL的副本、由原始证据重算的summary及相对路径索引；`saved-release/`固定保存本版本原两轮，重跑不覆盖；不是另造一次运行。
- `verify_result.json`、`submission-manifest.json`：已保存证据核验结果和当前发布字节清单，历史失败数据不改写。

完整原始文件保留于 `/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B-vLLM/`。如发布视图对重复图片或连续token-ID数组作可逆去重编码，转换规则、原文路径与转换前后hash由 `sources/evidence-view-map.json` 记录；不把转换后的文件称为原始字节。权重、数据集、缓存、凭据与连接配置不入库。
'''
(ROOT/'README.md').write_text(report,encoding='utf-8',newline='\n')
assert [line for line in report.splitlines() if line.startswith('## ')]==['## 1. 简介','## 2. 环境','## 3. 适配与配置','## 4. 启动与复现','## 5. 功能验证','## 6. 精度测试','## 7. 性能测试','## 8. 复现与限制']
print('Rendered fixed eight-chapter report from completed evidence')
