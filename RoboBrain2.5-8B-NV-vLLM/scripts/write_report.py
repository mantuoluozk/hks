"""Write the first native-vLLM report only after the complete matrix audit."""
import json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
O=R/'results'/Path(read(R/'results/formal-current.json')['path']).name
v=read(O/'verified-summary.json');e=read(R/'results/environment.json');assert v['complete']
bench=R/'bench';bench.mkdir(exist_ok=True)
for n in [1,2]:shutil.copyfile(O/f'round{n}.jsonl',bench/f'round{n}.jsonl')
header='| 输入长度参数 | 并发 | duration s | 请求/s | 输出token/s | 总token/s | mean TTFT ms | p99 TTFT ms | mean TPOT ms | p99 TPOT ms | mean E2E ms |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
table=header+'\n'+'\n'.join('| '+' | '.join([str(c['random_input_len']),str(c['max_concurrency'])]+[f"{c[k]:.3f}" for k in ['duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms']])+' |' for c in v['metrics'])
(bench/'summary.md').write_bytes(('# 第二轮六档性能\n\n'+table+'\n').encode())
packages='\n'.join(f'| {k} | {value} |' for k,value in e['packages'].items())
short=read(R/'results/initial/short/summary.json')
small='\n'.join(f"| {c['concurrency']} | {c['output_budget']} | {c['requests']} | {c['actual_output_tokens']} | {c['output_tokens_per_second']:.2f} |" for c in short)
a=read(R/'results/initial/functional-v2/summary.json')['cases'];b=read(O/'functional/summary.json')['cases']
repeat=[{'case':x['case'],'answer_equal':x['answer']==y['answer']} for x,y in zip(a,b)]
assert [x['case'] for x in a]==[x['case'] for x in b]
(R/'results/functional-repeat.json').write_bytes((json.dumps(repeat,indent=2)+'\n').encode())
equal=sum(c['answer_equal'] for c in repeat)
text=f'''# RoboBrain2.5-8B-NV · BW × vLLM 技术报告

## 1. 简介

- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/BAAI/RoboBrain2.5-8B-NV)
- [发布者模型说明](https://modelscope.cn/models/BAAI/RoboBrain2.5-8B-NV)
- [本轮SCNet镜像]({e['image_page']})
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)：`RoboBrain2.5-8B-NV-vLLM/`
- [已有SGLang版本](../RoboBrain2.5-8B-NV/README.md)

RoboBrain2.5-8B-NV为多模态具身理解模型，采用Qwen3-VL结构。本版本通过vLLM原生`Qwen3VLForConditionalGeneration`提供文本、图像、多轮、流式及坐标/轨迹格式输出服务，作为独立vLLM首版记录功能与性能。

| 验证项 | 本轮结果 |
|---|---|
| 运行文件 | 固定来源13项关键文件通过发布者清单校验 |
| 功能 | 初始格式适配后与正式服务各8项通过；两次功能文本{equal}/8一致 |
| 性能 | 官方六档参数，每档两次，共608/608计数请求；实际输出622592 token |
| 服务端交叉核对 | 矩阵阶段增量620成功请求、622976生成token，包含12次32-token客户端内部预热 |
| 独立进程完整性能复现 | 本轮仅一个完整矩阵服务，字段为null |
| 精度 | 本轮功能/格式诊断；完整公开基准结果仍归属原SGLang版本 |

## 2. 环境

SCNet Notebook，单张BW64GB、TP1，CPU配额15核，内存约59GiB。本轮实例区域未独立记录。镜像为`{e['image']}`，版本`{e['image_version']}`。Python {e['python']}。

| 组件 | 实测版本 |
|---|---|
{packages}

使用镜像现有HCU运行栈和API依赖。未额外安装依赖或修改全局框架源码。模型权重共两片，运行态约16.98GiB。启动日志记录KV cache容量241344 token、解码图约0.25GiB。

## 3. 适配与配置

checkpoint的`architectures`已为`Qwen3VLForConditionalGeneration`，直接使用vLLM现有原生实现。权重、tokenizer、chat template、图像处理配置均保留，未新增自定义模型类。

BF16、上下文32768、`max_num_seqs=8`、`max_num_batched_tokens=8192`、显存比例0.80。启用`mode=0`、`FULL_DECODE_ONLY`，捕获批次1/2/4/8；CPU数学库4线程。prefix cache与多模态预处理缓存关闭，服务只监听`127.0.0.1:30126`，最多两张输入图片。

SGLang版本中的确定性MM/BMM补丁属于原框架实现；本vLLM版本使用镜像自带算子路径，无须重放该补丁。checkpoint的图像处理器配置保留，本轮图像验证范围包括功能演示图和正式压测1080p随机图片。

Grounding首轮使用发布者原始提示，模型返回两个角点的文字描述，HTTP与非空输出正常，但不满足四元素列表格式。保留该记录后，格式接口提示补充“只返回归一化0–1000整数坐标JSON列表[x1,y1,x2,y2]”，通过原有坐标顺序和范围检查；没有放宽检查。原始与修订请求分别见`results/initial/functional/`、`functional-v2/`。

## 4. 启动与复现

部署模型目录及根目录`platform/scnet-notebook`；模型文件通过`MODEL_PATH`指定，默认使用SCNet共享模型路径。代码仓库不包含权重或数据集。

```bash
cd /root/private_data/hks/models/RoboBrain2.5-8B-NV-vLLM
bash run.sh           # 校验固定运行文件、启动并等待health
bash smoke.sh         # 8项功能及原始请求/响应
bash stop.sh          # 只停止属于本模型标准入口的PID
bash bench.sh         # 自行管理一个服务，六档各两轮；端口占用即退出
python verify.py --selfcheck
```

API兼容`/v1/chat/completions`，模型名`robobrain25-8b-vllm`。支持文本、单图、双图、多轮和流式。功能诊断显式temperature=0；短负载另固定top_p=1、seed42、ignore_eos=true，参数与响应保存。

标准入口在完整矩阵结束后通过独立启动、health、模型名、算术请求及停止验证；`results/entry-20261102T112332Z/summary.json`记录`passed=true`、`stop_exit=0`。该检查仅验证交付入口，不属于第二套完整性能矩阵。

## 5. 功能测试

修订格式提示后的初始服务与正式矩阵服务分别通过8项：算术、单图颜色、双图颜色、多轮、流式、grounding、pointing、trajectory。流式响应保存真实usage，全部用量大于0。

Grounding示例输出`[0, 301, 620, 1000]`；pointing为`[(901, 600)]`；trajectory为3个(x,y,depth)元组。坐标检查验证格式、顺序、有限数值和范围，不作为定位或机器人任务精度结论。

最初grounding列表格式失败及文字角点描述完整保留。修订后与正式服务的同名功能输出{equal}/8文本一致，见`results/functional-repeat.json`。该对照属于功能范围，不替代完整性能复现或公开基准评测。

## 6. 精度说明

依据本项目保存的2026-10-28赛事口径记录，本模型不要求额外精度测评，见`sources/competition-policy-20261028.json`。本轮未启动完整CV-Bench或其他精度任务。

原SGLang版本已有2638题CV-Bench补充自测88.10；该结果属于原框架及其补丁配置，不作为本vLLM版本新测成绩。发布者基线及原始协议见[原版本报告](../RoboBrain2.5-8B-NV/README.md)。正式性能使用随机token，不解释为语义或精度数据；赛事验收结论由主办方审核。

## 7. 首次性能实测

### 7.1 带预热短负载

固定448×336演示图，客户端并发1/8/16、固定输出32/128 token；每档先1次预热，计数N=max(4,2C)。六档共104个计数请求，实际输出8320 token。固定示例重复、不同客户端并发的成绩仅用于功能负载与调度探测，不作为同框架优化奖励对照。

| 客户端并发 | 输出预算 | 计数请求 | 实际输出token | 输出token/s |
|---:|---:|---:|---:|---:|
{small}

### 7.2 官方六档参数

依据保存的官方赛事bench.sh与技术报告模板第7章，输入长度参数4096/16384、输出1024、客户端并发1/8/64；每档两次取第二次，N=max(8,2C)。总计(8+16+128)×2种长度×2轮=608，第二轮计分304。依据说明见`sources/performance-protocol.md`。

使用SGLang自带客户端的`vllm-chat`后端，image数据、每请求一张1080p随机图，seed1，random_range_ratio=1，ignore_eos，stream include_usage；未修改压测客户端。长度参数不等于含视觉与模板的完整输入长度。服务最多同时运行8序列，客户端64并发时其余请求排队，延迟如实包含等待。

完整任务`{v['job']}`，608/608成功、每条实际输出1024，共622592个计数输出token；12次内部预热另生成384 token。正式功能结束后记录服务端基准计数，矩阵结束后按增量核对：620成功请求与622976生成token，和客户端完全对齐。

下表为第二次结果，九类模板指标完整保留：

{table}

原始结果为`bench/round1.jsonl`、`round2.jsonl`，完整日志、服务端计数及源码在`results/{v['job']}/`。吞吐与平均TTFT可从逐请求记录复算；p99 TPOT采用客户端统计，独立复算范围为吞吐和平均TTFT。

## 8. 复现范围与限制

本版为BW×vLLM独立首版，正式矩阵使用一个服务进程，各档第一轮预热、第二轮报告。未额外运行第二套独立服务完整矩阵，独立进程复现字段为null。

SGLang与vLLM结果按各自框架独立保留。本版未将跨框架、跨时点数值变化写作同框架优化收益。8项功能中的坐标/轨迹检查仅覆盖演示输入和格式范围；未据此声明完整具身任务精度。

`verify.py --selfcheck`检查13项运行文件证据、8项功能、六档两轮608请求、实际1024输出用量、非空文本、计时指标与服务端计数增量。默认`verify.py`保留赛事验收未判定状态。镜像与代码目录共同复现，权重与数据按发布者来源准备。
'''
(R/'README.md').write_bytes(text.encode('utf8'));print(R/'README.md')
