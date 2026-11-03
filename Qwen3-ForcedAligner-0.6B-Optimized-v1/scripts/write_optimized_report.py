import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
O=R/'results'/Path(read(R/'results/current.json')['path']).name
assert read(O/'state.json')['status']=='completed'
best=read(O/'selected.json');summary=read(O/'final/smoke/bench/summary.json');baseline=read(R/'sources/baseline/performance.json')
old={x['batch_size']:x for x in baseline['workloads'] if x['round']==2}
rows=[x for x in summary['workloads'] if x['round']==2]
table='|批次|首版历史吞吐|本轮吞吐|变化|与本轮单条一致|最大边界差ms|\n|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(f"|{x['batch_size']}|{old[x['batch_size']]['audio_seconds_per_wall_second']:.3f}|{x['audio_seconds_per_wall_second']:.3f}|{100*(x['audio_seconds_per_wall_second']/old[x['batch_size']]['audio_seconds_per_wall_second']-1):.2f}%|{x['same_as_serial']}/64|{1000*x['maximum_timestamp_delta_seconds']:.1f}|" for x in rows)
J=R/'results'/Path(read(R/'results/review-current.json')['path']).name
commands=read(J/'commands.json');assert len(commands)==6 and all(c['exit']==0 for c in commands)
official=read(J/'t03t04/verified-summary.json');assert official['successful']==30
repeat={w['batch_size']:w for w in read(J/'bench/smoke/bench/summary.json')['workloads'] if w['round']==2}
max_repeat=max(abs(repeat[x['batch_size']]['audio_seconds_per_wall_second']/x['audio_seconds_per_wall_second']-1)*100 for x in rows)
settings=json.dumps(best['settings'],ensure_ascii=False,indent=2)
environment=read(O/'final/preflight/summary.json')
packages='\n'.join(f'| {k} | {v} |' for k,v in environment['packages'].items())

text=f"""# Qwen3-ForcedAligner-0.6B · BW × vLLM 技术报告（优化版v1）

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen3-ForcedAligner-0.6B)
- [发布者模型说明](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Qwen3-ForcedAligner-0.6B-Optimized-v1/`
- [保留首版技术报告](../Qwen3-ForcedAligner-0.6B/README.md)
<!-- report-resources:end -->

本版提供音频与已知文本的强制对齐，输出词/字级时间戳。原生vLLM pooling/token_classify，原权重、分类头和发布者后处理不改。功能、时间戳差异和性能分别记录。

| 验证项 | 当前结果 |
|---|---|
| 运行文件 | 9项固定来源文件SHA256通过 |
| 功能 | 最终、默认入口和评审命令实跑中英文均2/2 |
| T03/T04专项 | 优化配置30/30生成，原始输出和反馈完整 |
| 性能 | 批次1/2/4/8各两轮，共512条计数输入 |
| 时间戳对照 | 本轮批次与单条一致；相对首版历史最大80ms差异 |
| 独立进程完整性能复跑 | 评审命令另512条完成；吞吐最大偏差{max_repeat:.2f}% |
| 官方验收 | 未判定，null |

## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；TP1 |
| 框架 | vLLM；实际组件版本见下文 |
| 镜像用途 | 复用 ASR 专用镜像 |
| 镜像版本 | `v1` |
| 镜像地址 | `appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1` |
| 镜像复验 | 本 Notebook 两个独立进程各512条；吞吐最大偏差{max_repeat:.2f}% |
<!-- report-environment:end -->

SCNet Notebook、单BW、BF16。实际复用 qwen-asr:v1 镜像，地址`appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1`；本轮区域为华中一区【A区】，由Notebook页面信息确认。实际设备、包版本、CPU线程及原生实现指纹见`results/{O.name}/final/preflight/summary.json`。固定9个运行文件通过来源清单SHA256校验。

| 组件 | 实测版本 |
|---|---|
{packages}

HIP版本`{environment['hip']}`；设备`{environment['devices'][0]['name']}`，显存{environment['devices'][0]['memory']/2**30:.2f}GiB，TP1。未新增依赖，未修改全局框架源码。交付时环境原始记录另保存在`env.log`，取自最终进程preflight；`env_check.sh`用于重新采集当前环境。

## 3. 适配与配置

固定长度专门化编译尝试因启动成本偏高而中止，原始日志和中止标记保留在`results/optimization-20261103T013804Z/`。随后改用动态范围编译。三个已完成候选各一遍带预热：同进程eager4线程、同进程PIECEWISE4线程、同进程PIECEWISE2线程。筛选输入为相同两条中英文演示音频，各批次8个计数输入；失败候选及日志保留。最终配置为`{best['name']}`：

```json
{settings}
```

上下文8192、max_num_seqs8、显存比例0.35、关闭prefix及多模态预处理缓存。仅在模型独立目录设置环境，不修改镜像全局源码。

## 4. 启动与复现

权重路径可通过MODEL_PATH设置，默认`/root/private_data/SothisAI/model/Aihub/Qwen3-ForcedAligner-0.6B/main/Qwen3-ForcedAligner-0.6B`。代码部署到`/root/private_data/hks/models/Qwen3-ForcedAligner-0.6B-Optimized-v1`，配套平台目录位于`/root/private_data/hks/platform/scnet-notebook`。音频按`sources/audio-sources.json`来源及校验准备，放`data/examples/`；不自动下载。

```bash
bash prepare_data.sh
bash env_check.sh
bash run.sh
bash smoke.sh
RUN_RESULTS_DIR="$PWD/results/reproduce" bash bench.sh
bash official.sh
python verify.py
```

| 文件 | 作用 / 产出 |
|---|---|
| `prepare_data.sh` / `data/README.md` | 指定现有输入目录并校验固定条件，无数据下载 |
| `official.sh` | T03/T04专项30条生成及输出检查 |
| `baselines/Qwen3-ForcedAligner-0.6B.json` | 发布者AAS参考值与适用范围 |
| `env.sh` / `run-scnet.sh` | 统一环境参数 / 命令包装器 |
| `env_check.sh` | 当前环境检查，输出`env.log` |
| `run.sh` / `smoke.sh` | 权重校验、preflight及中英文对齐；离线CLI完成即退出 |
| `bench.sh` | 四批次各两轮，保存逐条分类档位、时间戳及逐批耗时 |
| `verify.py` | 核验保存的原始证据，输出`verify_result.json` |
| `server.log` / `smoke.log` | 评审命令run/smoke的原始对齐日志 |
| `bench/round1.jsonl` / `round2.jsonl` | 从原始汇总逐行导出对应轮四档，完整逐请求原始数据路径见第5章 |
| `submission-files.json` | 当前发布文件与字节数清单 |

默认配置入口另完成中英文2/2，输出与最终性能进程的单条参考逐项一致，记录见`results/entry-default/`。smoke与bench为离线CLI，进程完成自行退出。verify默认核验已保存证据，检查成功返回0，真实失败非零；官方验收状态单独保留null。新测输出目录通过RUN_RESULTS_DIR指定，默认按UTC时间新建；不覆盖已发布结果。`scripts/optimize.py`可从头重做候选筛选，原始历史执行脚本保留在各任务`source/`中。

## 5. 功能测试

最终配置中英文2/2完成，保存每个单元的起止秒数、原始分类档位、分类形状、批次耗时与逐边界差异。两条重复示例不是独立512条语料。完整原始记录见`results/{O.name}/final/smoke/bench/requests.jsonl`；源配置、源码及运行日志在同一任务目录。

## 6. T03/T04 固定条件测试

优化配置实际完成英文LibriSpeech test-clean 30条、单条不超过15秒，BF16、seed42、给定文本。30/30成功，反馈表无失败；条件和音频在加载前逐项SHA256校验，输出词序、秒单位、单调性及边界范围通过。条件标记为BOUNDARIES_PENDING，未提供人工时间边界，未计算AAS。

原始输出与反馈位于`results/{J.name}/t03t04/`，包括30份JSON、反馈CSV、environment.json、summary.json、verified-summary.json。优化配置使用PIECEWISE和同进程引擎，准确参数记入环境；不是复制首版输出。首版逐边界对照29/30条完全一致，另1条最大差80ms，见`results/t03t04-historical-comparison.json`。


逐单元文本、时间戳与首版历史输出及本轮单条输出对照。原版批次8已有80ms差异；80ms为一档分类间隔及历史观察范围，不是官方精度容差。逐边界检查保留真实差异，不将格式与单调性通过当作AAS精度通过。最终512条各批次与本轮单条输出完全一致；相对首版历史单条输出最大边界差80ms，不能称与首版逐边界一致。本轮完成优化版专项30条，不等于复现发布者AAS；首版专项结果独立保留。

## 7. 首次性能

最终单进程批次1/2/4/8，各档两轮、每轮64条，共512个计数输入；每轮另先两条预热，报告第二轮。计时为llm.encode，包含内部特征处理、执行及返回，排除音频读取、重采样、初始化和输出解析。吞吐单位为音频秒/墙钟秒，批次完整调用耗时不能当作独立请求延迟。

{table}

对照采用已发布首版历史第二轮，跨日期新Notebook；没有重新测首版，不声称严格同一进程A/B。完整逐批计时及汇总保留，首套两轮属于同一进程；本次另通过评审命令启动独立进程复跑512条，第二轮吞吐最大偏差{max_repeat:.2f}%，完整复跑数据位于`results/{J.name}/bench/`。

## 8. 复现与限制

首套正式512条与评审入口独立复跑512条分别保留。六条评审命令均实际返回0，见`results/{J.name}/commands.json`；官方验收未判定。强制对齐不生成1024输出token，不适用4K/16K×并发1/8/64的608条生成矩阵。示例任务覆盖English/Chinese，不宣称其他语言或完整语料表现。历史失败候选和原始记录保留，未改写为成功。
"""
text='\n'.join('| '+' | '.join(cell.strip() for cell in line.strip().strip('|').split('|'))+' |' if line.startswith('|') else line for line in text.splitlines())+'\n'
(R/'README.md').write_bytes(text.encode())
