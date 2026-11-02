# Trident-Qwen3VL-2B · vLLM · 优化版 v1 技术报告

## 1. 简介

- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Trident-Qwen3VL-2B)
- [发布者模型](https://huggingface.co/OpenBMB/Trident-Qwen3VL-2B)
- [本轮 SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)：`Trident-Qwen3VL-2B-Optimized-v1/`
- [原版报告](../Trident-Qwen3VL-2B/README.md)

Trident 是文本、图片、图文融合检索嵌入模型，返回2048维L2归一化向量。本版本优化单BW上的同步批处理执行，保持发布者权重、指令、图像处理参数及LAST池化。

| 验证项 | 本轮结果 |
|---|---|
| 功能 | 6条多模态输入、HTTP embedding及非法role 400检查通过 |
| 性能 | 选定配置单进程24格、1536条计数输入；报告第二轮12档 |
| 历史公开精度 | 原版ChartQA Origin NDCG@10 ×100=69.4483；归属原eager首版 |
| 优化版精度范围 | 演示输入向量及token一致性检查；本轮未重跑完整检索集 |
| 镜像范围 | 本轮Notebook使用ASR镜像v1，已执行优化候选及最终性能 |
| 独立进程复现 | 最终配置完整矩阵仅一个进程；独立进程复现字段为null |

## 2. 环境

SCNet Notebook，BW64GB单卡，CPU配额15核、内存约59GiB。本轮实例区域未独立记录。镜像`appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1`，版本`v1`。实际软件版本见`results/environment-20261102.json`：

| 组件 | 版本 |
|---|---|
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| vllm | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| openai | 2.30.0 |
| pillow | 11.3.0 |
| torchvision | 0.26.0+dtk2604.torch2110.2609101802.gee55be |

随模型提供qwen-vl-utils源码和发布者固定实现，沿用原依赖及许可。

## 3. 适配与优化配置

vLLM原生Qwen3-VL backbone，`runner=pooling`、`convert=embed`、`seq_pooling_type=LAST`、`use_activation=true`。BF16、8192上下文、最大序列数8、显存比例0.45。图像min/max pixels=262144/1310720；CPU数学库线程固定为2。

选定候选`piecewise-best-cpu`：`VLLM_ENABLE_V1_MULTIPROCESSING=0`，执行模式`piecewise`。同进程调用可将离线batch直接交给引擎，减少跨进程通信，并改变请求进入调度器的批次组织。实际吞吐收益包含该执行与调度变化，不能全部归为单个kernel收益。

分段编译候选使用`mode=3`、`cudagraph_mode=PIECEWISE`、token捕获档64/128/256/512/1024/2048/4096。短测文本/图片/融合输入每条分别35/447/453 token。编译候选首次torch.compile约193.41秒，启动与图捕获计时独立于稳态性能；其他Notebook的冷启动耗时可能不同。

关闭prefix cache和多模态预处理缓存；每个计数图片输入使用独立UUID。权重和图像像素预算保持原配置。

## 4. 启动与复现

将模型目录与根目录`platform/scnet-notebook`部署到同一个项目根目录，准备完整发布者模型并通过`MODEL_PATH`指定。模型目录不包含权重和数据集。

```bash
export HKS_ROOT=/root/private_data/hks
cd "$HKS_ROOT/models/Trident-Qwen3VL-2B-Optimized-v1"
export MODEL_PATH=/path/to/Trident-Qwen3VL-2B
bash run-scnet.sh bash run.sh
bash run-scnet.sh bash bench.sh
bash run-scnet.sh python scripts/serve_embeddings.py --host 127.0.0.1 --port 8000
python verify.py --selfcheck
```

`run.sh`检查运行文件、环境和功能；`bench.sh`先准备演示输入参考，再执行两轮embedding矩阵。HTTP桥接的`POST /v1/embeddings`支持query/doc角色、每次1–8条输入、2048维float向量。HTTP桥接串行处理请求，下文性能单位为离线同步batch吞吐，batch大小不等同HTTP并发。

## 5. 功能与数值检查

文本query、文本doc、单图、图文、无关文本、双图6条输入均通过。最终配置与发布者BF16参考的最低余弦为0.99962628，所有展开token序列一致；输出均为有限2048维归一化向量，HTTP与直接调用对照通过。

候选逐batch与本轮附带eager记录对照，保留余弦和最大绝对差。0.9999为额外严格诊断，部分批次未达到；原有功能检查标准为余弦>0.99。不同batch下eager自身也有约0.9998量级余弦差异；不能声明优化版逐元素相同。数值诊断见`results/optimized-summary.json`和筛选目录`batch-vector-diagnostic.json`，原严格诊断失败日志保留。

## 6. 精度测评

原版完整ChartQA Origin：1250查询、1509候选，NDCG@10 ×100=69.4483；同BW发布者实现69.5031，论文参考70.19。[原报告](../Trident-Qwen3VL-2B/README.md)记录数据、评分协议和完整证据。

以上完整精度属于原eager首版。优化版本轮检查演示输入、向量及token序列，未重跑ChartQA；不将历史分数作为优化后重新实测成绩。赛事验收结论由主办方审核。

## 7. 性能筛选与最终实测

### 7.1 配置筛选

各候选一遍带预热：3类固定输入×batch 1/2/4/8，每格64条计数输入、2次预热调用，共768条。以全部12格累计`LLM.embed`耗时最小选择配置，并检查功能、输入token及向量。主性能对照采用初版已发布基线；本轮附带eager记录保留在原始证据，未再次补跑基线。

| 候选 | 12格累计计时/秒 | 合计请求/秒 |
|---|---:|---:|
| inprocess | 32.221 | 23.84 |
| piecewise-inprocess | 25.930 | 29.62 |
| inprocess-cpu1 | 33.643 | 22.83 |
| inprocess-cpu2 | 31.694 | 24.23 |
| piecewise-best-cpu | 25.700 | 29.88 |

### 7.2 最终完整embedding矩阵

任务`final-20261102T082038Z`，选定配置单个初始化进程，两轮共24格、1536条计数输入，实际返回输入token共478720。每格64条计数输入、2次预热调用；图片使用独立UUID。下表取第二轮，与首版2026-10-26已发布第二轮成绩比较。初版与本轮跨日期、跨Notebook，表中为相同负载下相对历史成绩的数值变化，不提供同窗口因果归因或统计置信区间。

| 输入 | batch | 初版请求/秒 | 优化版请求/秒 | 相对变化 | 优化版输入token/秒 |
|---|---:|---:|---:|---:|---:|
| text | 1 | 25.27 | 58.32 | +130.84% | 2041.32 |
| text | 2 | 25.34 | 113.20 | +346.72% | 3962.08 |
| text | 4 | 49.66 | 213.77 | +330.44% | 7482.06 |
| text | 8 | 96.18 | 363.99 | +278.46% | 12739.80 |
| image | 1 | 11.27 | 16.29 | +44.51% | 7281.29 |
| image | 2 | 11.56 | 22.40 | +93.71% | 10010.79 |
| image | 4 | 17.50 | 25.35 | +44.81% | 11329.76 |
| image | 8 | 20.64 | 27.18 | +31.69% | 12149.55 |
| fused | 1 | 11.45 | 16.19 | +41.41% | 7334.18 |
| fused | 2 | 11.52 | 20.59 | +78.71% | 9328.27 |
| fused | 4 | 18.15 | 23.14 | +27.47% | 10480.33 |
| fused | 8 | 21.68 | 26.27 | +21.14% | 11899.29 |

计时累计同步`LLM.embed`调用，包含模型侧预处理、执行和结果返回；图片文件读取、payload/UUID构造、输出审计和模型启动在计时外。保存逐调用耗时，可重算均值及p99批调用延迟；批调用延迟不能当作HTTP端到端延迟。

Trident为embedding接口，生成文本模板中的固定1024输出token、TTFT及TPOT不适用。本轮采用上述完整embedding协议，不把1536条嵌入输入写成608条生成请求或官方生成矩阵通过。每格为固定演示输入重复，不是1536条独立样本。

## 8. 复现范围与证据

最终进程两轮12个配对档位最大吞吐偏差为7.76%，属于同进程轮次变化。最终完整矩阵只运行一个进程，独立进程复现字段为null；原版历史3.88%独立进程偏差仍仅归属原版。

`results/optimized-summary.json`给出独立复算汇总；`results/final-20261102T082038Z/`包含功能向量、HTTP结果、实际token数、每次调用耗时、配置选择、日志和运行源码。`results/optimization-20261102T080346Z/`保留各候选与严格诊断记录。`python verify.py --selfcheck`检查24格、1536输入、478720实际输入token、归一化、功能与计时一致性。默认`verify.py`保留赛事验收未判定状态。
