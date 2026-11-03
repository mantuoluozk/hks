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
settings=json.dumps(best['settings'],ensure_ascii=False,indent=2)
text=f"""# Qwen3-ForcedAligner-0.6B × vLLM 优化版v1

## 1. 简介

[发布者模型](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) · [SCNet镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490) · [光合代码](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · [保留首版](../Qwen3-ForcedAligner-0.6B/README.md)

本版提供音频与已知文本的强制对齐，输出词/字级时间戳。原生vLLM pooling/token_classify，原权重、分类头和发布者后处理不改。功能、时间戳差异和性能分别记录。

## 2. 环境

SCNet Notebook、单BW、BF16。实际复用 qwen-asr:v1 镜像，地址`appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1`；本轮区域未独立记录。实际设备、包版本、CPU线程及原生实现指纹见`results/{O.name}/final/preflight/summary.json`。固定9个运行文件通过来源清单SHA256校验。

## 3. 优化配置

固定长度专门化编译尝试因启动成本偏高而中止，原始日志和中止标记保留在`results/optimization-20261103T013804Z/`。随后改用动态范围编译。三个已完成候选各一遍带预热：同进程eager4线程、同进程PIECEWISE4线程、同进程PIECEWISE2线程。筛选输入为相同两条中英文演示音频，各批次8个计数输入；失败候选及日志保留。最终配置为`{best['name']}`：

```json
{settings}
```

上下文8192、max_num_seqs8、显存比例0.35、关闭prefix及多模态预处理缓存。仅在模型独立目录设置环境，不修改镜像全局源码。

## 4. 运行与复现

权重路径可通过MODEL_PATH设置；默认共享路径与首版一致。音频按`sources/audio-sources.json`来源及校验准备，放`data/examples/`；不自动下载。

```bash
bash run-scnet.sh bash smoke.sh
RUN_RESULTS_DIR="$PWD/results/reproduce" bash run-scnet.sh bash bench.sh
python verify.py
```

默认配置入口另完成中英文2/2，输出与最终性能进程的单条参考逐项一致，记录见`results/entry-default/`。smoke与bench为离线CLI，进程完成自行退出。verify默认核验已保存证据，检查成功返回0，真实失败非零；官方验收状态单独保留null。

## 5. 功能与原始数据

最终配置中英文2/2完成，保存每个单元的起止秒数、原始分类档位、分类形状、批次耗时与逐边界差异。两条重复示例不是独立512条语料。完整原始记录见`results/{O.name}/final/smoke/bench/requests.jsonl`；源配置、源码及运行日志在同一任务目录。

## 6. 时间戳与精度范围

逐单元文本、时间戳与首版历史输出及本轮单条输出对照。原版批次8已有80ms差异；80ms为一档分类间隔及历史观察范围，不是官方精度容差。逐边界检查保留真实差异，不将格式与单调性通过当作AAS精度通过。最终512条各批次与本轮单条输出完全一致；相对首版历史单条输出最大边界差80ms，不能称与首版逐边界一致。本轮不新增完整精度评测，首版T03/T04的30条结果仍归属首版。

## 7. 性能结果

最终单进程批次1/2/4/8，各档两轮、每轮64条，共512个计数输入；每轮另先两条预热，报告第二轮。计时为llm.encode，包含内部特征处理、执行及返回，排除音频读取、重采样、初始化和输出解析。吞吐单位为音频秒/墙钟秒，批次完整调用耗时不能当作独立请求延迟。

{table}

对照采用已发布首版历史第二轮，跨日期新Notebook；没有重新测首版，不声称严格同一进程A/B。完整逐批计时及汇总保留，两轮属于同一进程，不冒充独立进程复现。

## 8. 复现与限制

本轮仅一套最终512输入协议，独立进程完整性能复现与官方验收未判定。强制对齐不生成1024输出token，不适用4K/16K×并发1/8/64的608条生成矩阵。示例任务覆盖English/Chinese，不宣称其他语言或完整语料表现。历史失败候选和原始记录保留，未改写为成功。
"""
(R/'README.md').write_bytes(text.encode())
