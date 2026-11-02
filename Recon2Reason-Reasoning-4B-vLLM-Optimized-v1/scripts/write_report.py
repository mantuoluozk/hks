"""Render the technical report from verified saved evidence only."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
 job=ROOT/'results'/Path(a.job).name;verified=json.loads((job/'verified-summary.json').read_text(encoding='utf-8'));assert verified['complete']
 short=ROOT/'results/optimization-20261102T042055Z'
 screen=json.loads((short/'independent-screen-audit.json').read_text(encoding='utf-8'));assert screen['all_requests_usage_text_equal']
 gains=[x['gain_percent'] for x in screen['cells']]
 summary=(ROOT/'bench/summary.md').read_text(encoding='utf-8')
 shorttable='| 提示重复次数 | 并发 | eager token/s | 解码图 token/s | 提升 |\n|---:|---:|---:|---:|---:|\n'
 for r in json.loads((short/'comparison.json').read_text(encoding='utf-8'))['cells']:
  shorttable+=f"| {r['prompt_repeats']} | {r['concurrency']} | {r['baseline_tps']:.4f} | {r['candidate_tps']:.4f} | {r['gain_pct']:.2f}% |\n"
 fgains=[x['change_percent']['output_throughput'] for x in verified['comparison']]
 report=f'''# Recon2Reason-Reasoning-4B · BW × vLLM 优化版v1技术报告

## 1. 简介

- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Recon2Reason-Reasoning-4B)
- [发布者模型说明](https://huggingface.co/BAAI/Recon2Reason-Reasoning-4B)
- [SCNet Recon 专用镜像 v1.0.0](https://www.scnet.cn/ui/aihub/image/zack233/2116559999150473218)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)，独立目录 `Recon2Reason-Reasoning-4B-vLLM-Optimized-v1/`
- [原版技术报告](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks/-/blob/main/Recon2Reason-Reasoning-4B-vLLM/README.md)

本版在单张 BW 上使用 vLLM 原生 Qwen3-VL 实现运行原版 BF16 Recon2Reason-Reasoning-4B（约4.44B参数），将 eager 解码切换为 `FULL_DECODE_ONLY` 图执行。模型权重、tokenizer、图像处理、精度、框架算子源码均未修改。原版目录与 tag 保留，结果分别记录。

| 验证项 | 实测结果 |
|---|---|
| 功能 | 同日 eager 短测、解码图短测、解码图完整矩阵与标准入口检查各5/5 |
| 简单负载对照 | 每配置带预热一遍，48计数请求；吞吐提升{min(gains):.2f}%–{max(gains):.2f}% |
| 输出对照 | 两配置50条请求（含2预热）的请求、usage和可见文本逐条相同 |
| 完整性能 | 官方六档、每档两次取第二次，共608/608矩阵请求，每条输出1024 token |
| 完整矩阵吞吐变化 | 对照原版匹配档位，输出吞吐变化{min(fgains):.2f}%–{max(fgains):.2f}%；九类指标全部列出 |
| 精度 | 赛事对本模型不要求精度，不将功能或随机压测作为精度成绩 |
| 复现范围 | 本轮优化版一个完整矩阵服务；独立进程复现字段为null |
| 官方验收 | 由主办方审核，本报告不声明已获官方验收 |

## 2. 环境

本轮于2026-11-02在 SCNet Notebook 的单张 BW 64GB（gfx936）运行，TP1、BF16。CPU配额15核，内存限制63,350,767,616字节（59GiB），数学库线程固定4。运行时 CPU affinity 128不代表分配128核；采用 cgroup 配额。证据见 `results/optimization-environment.json` 和服务日志。

镜像为 `appstore.scnet.cn:5000/aihub/dcu/zack233/recon2reason-vllm:v1.0.0`。复用原 Recon 专用镜像，通过本版仓库入口指定解码图配置。

| 组件 | 实测版本 |
|---|---|
| Python / DTK | 3.10.12 / 26.04 |
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| vLLM | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| vllm-hcu | 0.25.1+dtk2604.torch2110.2609272150.gee325d |
| SGLang客户端 | 0.5.18+dtk2604.torch2110.2609231937.gc3bbbd |
| Transformers | 5.12.1 |
| OpenAI API依赖 | /opt/qwen3-asr-deps 中固定2.30.0 |
| Triton | 3.6.0+dtk2604.torch2110.2609231415.g8cce4a |
| tokenizers / Pillow | 0.22.2 / 11.3.0 |

## 3. 适配与优化配置

保持32768上下文、显存比例0.80、max_num_seqs64、TP1、BF16、多图上限2、默认 prefix cache 与 chunked prefill 开启。唯一执行策略变化为移除 `--enforce-eager`，设置：

```json
{{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}}
```

该配置针对解码阶段重放图，prefill仍走原路径，不开启Inductor编译、不改变原生算子。短测服务实际捕获7档，日志记录捕获约2秒、图内存0.28GiB。不能把短负载收益直接外推到长输入高并发。完整协议见 `OPTIMIZATION-PROTOCOL.md`；原始 launcher 和源码快照随各任务保存。

## 4. 启动与复现

将本目录部署到 `/root/private_data/hks/models/Recon2Reason-Reasoning-4B-vLLM-Optimized-v1`；复用镜像内环境脚本，或部署仓库公共 `platform/scnet-notebook`。权重默认使用 `env.sh` 中社区完整路径，也可用 `MODEL_PATH` 指定。Notebook内直接执行：

```bash
bash run.sh
bash env_check.sh
bash smoke.sh
bash stop.sh
bash bench.sh
python verify.py
```

服务监听127.0.0.1:30123，模型名 `recon2reason-4b`。`bench.sh` 从空闲端口运行本版单服务608请求矩阵、保存带时间戳的结果，完成后停止归属服务。`verify.py`仅核验已有证据，不启动模型。原版完整1216请求基准保留，原版入口不变。本版标准入口已实跑通过，证据为 `results/entry-20261102T062207Z/`，包含环境、5项功能和停服日志。

## 5. 功能测试

短测两服务、完整矩阵服务与标准入口服务各通过文本算术5、单图左侧red、双图red/blue、多轮右侧blue、单图SSE流式red与DONE，共5项。合成图及请求、响应保存在 `results/smoke-*`，各服务日志给出对应目录。

短测两配置分别按历史文本负载执行：提示重复32/256、固定128输出、temperature0、ignore_eos，每提示长度一次预热，并发1/2/4各8计数请求。50条请求及usage、可见文本逐条相同，独立核验见 `{short.relative_to(ROOT).as_posix()}/independent-screen-audit.json`。这属于合成功能与执行一致性检查，不代表真实任务精度。

## 6. 精度测评

依据赛事说明，本模型不要求精度测试，规则证据为 `sources/competition-policy-20261028.json`。本vLLM版本未新增OCRBench、MMMU等精度测试，没有引用SGLang成绩充当本版精度。

## 7. 优化性能

### 简单负载同日对照

同一单卡Notebook、同框架和服务配置，每配置仅一遍带预热，不将第二次筛选作为取优机会。每档输出token/s由实际usage与计时独立重算。

{shorttable}

### 完整六档矩阵

使用未修改的镜像SGLang `vllm-chat`客户端，seed1、每请求1张1080p随机JPEG，4K/16K文本长度参数、输出1024、并发1/8/64、N=max(8,2C)。每档两次取第二次，计数请求共608，实际计数输出622592 token；客户端12次各32-token内部预热及功能请求另计。流式usage提供实际输出计数，temperature0、ignore_eos保持原协议。4096/16384为文本参数，实际输入包含chat template及图像。

默认缓存设置保留，第二次不代表冷前缀。优化版与原版每档的输入长度分布和输出长度逐条核对一致。原版对照取 `results/formal-20261031T163337Z/process1/round2.jsonl`；本轮取 `{job.relative_to(ROOT).as_posix()}/process1/round2.jsonl`。服务端计数快照在矩阵结束、特殊token诊断之前采集，记录622988生成token与625成功请求，与622592矩阵token、12次32-token内部预热、5项功能请求的账目相符，原文及核验见本任务 `backend-audit.json`。

原版是历史Notebook基准，本轮是同卡型、同框架、同负载和相同非图参数的优化版测量。

{summary[summary.index('## 原版'):].replace('## 原版','### 原版',1).replace('## 优化版','### 优化版',1).replace('## 变化百分比','### 变化百分比',1)}

原始JSONL、逐档日志及 `verified-summary.json` 均保留。全部九类指标共54项变化同时呈现，吞吐提升与延迟变化分开解释，未删除不利档位。

## 8. 复现与限制

本轮完整矩阵仅一个独立服务，不将原版两进程最大偏差9.6517%继承为图执行版复现成绩。短测输出一致与完整矩阵协议一致分别验证，不将合成文本视为精度成绩。

完整矩阵可见文本为空的请求共{len(verified['hidden_special_requests'])}条，保留原记录和计数。若出现隐藏特殊token，停服前在本轮图配置下对匹配请求进行非流式单请求重放，保存真实1024 token IDs与显示开关响应，证据在正式任务 `graph-empty-output/`；只有当前配置证据通过才接受解释，未直接沿用原版豁免。该重放支持隐藏特殊token的解释，原流式矩阵未直接记录逐请求token IDs。没有可见内容的客户端TTFT=0不表示真实首token零延迟。

CPU核验独立重算请求数、输入/输出token、吞吐及mean/p99 TTFT。原客户端output-details没有逐条E2E原始延迟，TPOT/E2E保留客户端聚合值，不能称p99 TPOT已独立重建。完整核验与文件哈希清单仅验证已保存证据的完整性；官方验收由主办方执行。
'''
 (ROOT/'README.md').write_text(report,encoding='utf-8',newline='\n')
 print('Report written from verified results')
if __name__=='__main__':main()
