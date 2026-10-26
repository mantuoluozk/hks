# Qwen3-ForcedAligner-0.6B · BW × vLLM 技术报告

## 1. 简介

T04 强制对齐，模型为 `Qwen/Qwen3-ForcedAligner-0.6B`，输入音频及对应文本，输出词 / 字级起止时间。权重来自 [SCNet AI 社区](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen3-ForcedAligner-0.6B)，9 个运行文件逐项 SHA256 校验通过；单文件 BF16 权重 1,835,544,544 字节，SHA256 为 `47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7`。社区说明文件不计入运行文件匹配。完整清单与实际校验见 `sources/modelscope-files.json` 和 `results/scnet/validation-20261026T094057Z/runtime/weights-check.json`。

## 2. 环境

SCNet Notebook，华中一区（A区），单张 BW，逻辑设备 0；HIP 6.3.26113。实际包版本：

| 组件 | 版本 |
|---|---|
| PyTorch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| vLLM | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| Transformers | 5.12.1 |
| OpenAI Python | 2.30.0 |
| librosa | 0.11.0 |

[实际使用的 SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)：`appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1`，版本 `v1`。镜像保持私有，提交时开放；固定摘要待提交前补充。沿用镜像内 `/opt/qwen3-asr-deps`，本模型未新增系统依赖或修改框架源码。原生强制对齐实现的导入检查及代码哈希见 `results/scnet/validation-20261026T094355Z/preflight/summary.json`。

## 3. 适配与配置

使用原生 `Qwen3ASRForcedAlignerForTokenClassification`，`runner=pooling`、`pooling_task=token_classify`。通过 `hf_overrides.architectures` 选择已存在的实现，不修改权重、分类头或 tokenizer。BF16、eager、上下文 8192、最大运行序列 8、显存比例 0.35；关闭 prefix cache 和多模态处理缓存；OMP / OpenBLAS / MKL / NumExpr 线程数固定为 4。

文本处理与时间戳后处理采用发布者固定版本 `7c6daf77a2421100f5fb066495372c00129d39ff` 的原始处理器类。通过 AST 提取该类，避免引入另一个 HF 模型后端；类及构造函数、韩语词典资产不改写，文件哈希逐项验证。本适配入口目前只验证 English / Chinese，不宣称所有上游语言均已验证。

分类头含 5000 档，每档 80ms。只在 `<timestamp>` 位置取分类 argmax，按模型配置换算毫秒，再执行发布者原始 `fix_timestamp` / `parse_timestamp`，最终输出秒。原始分类档位和后处理结果同时保留。

## 4. 启动脚本

本版为离线强制对齐 CLI，不使用生成式聊天接口。将模型目录部署到 `/root/private_data/hks/models/Qwen3-ForcedAligner-0.6B`，配套平台目录放到 `/root/private_data/hks/platform/scnet-notebook`。Notebook 已是容器，不嵌套 Docker。完整社区权重放在 `run-scnet.sh` 默认路径，或显式设置 `MODEL_PATH`。

示例音频按 `sources/audio-sources.json` 的地址及哈希放到 `data/examples/`；给定文本见 `sources/functional-cases.json`。数据与权重不随源码分发。

```bash
cd /root/private_data/hks/models/Qwen3-ForcedAligner-0.6B
bash run-scnet.sh python scripts/prepare_examples.py
RUN_RESULTS_DIR="$PWD/results/scnet/reproduce" bash run-scnet.sh bash smoke.sh
# 性能测试同时包含功能检查；使用新的输出目录：
RUN_RESULTS_DIR="$PWD/results/scnet/benchmark" bash run-scnet.sh bash bench.sh
```

`run.sh` 顺序执行权重校验、环境检查及真实对齐。自定义输入可通过 `ALIGNMENT_CASES=/路径/cases.json` 与 `AUDIO_DIR=/音频目录` 指定，JSON 列表格式沿用 `sources/functional-cases.json` 的 id、language、audio_file、text 字段。当前仅接受 English / Chinese。默认不下载模型或音频；需要获取两个小型演示 WAV 时，显式给 `prepare_examples.py` 加 `--download`。运行前应停止同卡其他推理作业。

## 5. 功能测试

两个真实中英文示例均通过，共 **2/2**；输出 English 37 个单元、Chinese 13 个单元，包含每个单元的起止秒数。检查分类维度、有限数值、时间戳 token 数、单元数量、后处理单调性及音频范围。原始档位、分类形状与时间戳见 `results/scnet/validation-20261026T094057Z/smoke/`。独立进程的单条结果比较：en-demo 完全一致、zh-demo 完全一致。

批量测试均保留与本轮单条输出的逐边界比较。批次 8 中存在一档 80ms 边界差异，不能标记全部批量结果逐字 / 逐时间戳一致；首版建议使用单条或批次不超过 4。具体数量及最大差异见第 7 节。

## 6. 精度测评

发布者公开 AAS 参考值：MFA-Labeled Raw 中文 33.1ms、英文 37.5ms、平均 42.9ms；Concat-300s 平均 52.9ms；Human-Labeled 平均 32.4ms。不同评测组不能混用，来源及口径见 `官方精度基线查证.md`。

当前两个演示样本用于功能及吞吐检查，没有匹配的真实边界标注，未计算或声称通过 AAS。英文给定文本取自演示音频转写，不作为人工时间边界真值。正式精度等待赛事匹配的标注集、评分代码及 GPU 基线，`official_accuracy_pass=null`。格式合法、单调、批量一致或独立进程一致均不能替代 AAS 验收。

## 7. 首次性能

批次大小 1/2/4/8，每档每轮请求数 max(64,4B)，共四档 × 两轮 × 每轮 64 个请求，即 **8 项 / 512 个计数请求**。中英文演示 WAV 交替输入；每轮先用两条样本预热，报告第二轮。计时范围为 `llm.encode`，包含其内部特征处理、模型执行和结果返回，排除初始化、文件读取、重采样、文本准备及输出解析；不是完整端到端服务延迟。逐条记录中的 `elapsed_seconds` 表示该条所在批次的完整调用耗时，与 `batch_elapsed_seconds` 相同，不用作独立逐请求延迟或 TPOT；512 个计数请求来自两条演示音频的重复，不是 512 条独立音频。

| 批次大小 | 请求数 | 音频秒/墙钟秒 | 与单条结果一致 | 最大边界差 ms |
|---:|---:|---:|---:|---:|
| 1 | 64 | 134.12 | 64/64 | 0 |
| 2 | 64 | 137.87 | 64/64 | 0 |
| 4 | 64 | 249.07 | 64/64 | 0 |
| 8 | 64 | 398.08 | 40/64 | 80 |

吞吐越高越好。两条演示音频的结果不代表完整语音场景或官方 T04 性能协议，也不作为已达到优化奖励门槛的证明。逐请求文本单元、原始分类档位、时间戳及批次耗时见 `results/scnet/validation-20261026T094057Z/smoke/bench/`。

## 8. 复现与限制

在同一 SCNet Notebook、同一镜像及同一张卡上，两次独立初始化 vLLM 进程，按相同代码、音频、文本和参数重复全部功能及性能。无额外系统依赖安装。第二轮性能对比如下：

| 批次大小 | 首次吞吐 | 复跑吞吐 | 偏差 |
|---:|---:|---:|---:|
| 1 | 134.12 | 130.71 | -2.54% |
| 2 | 137.87 | 134.54 | -2.42% |
| 4 | 249.07 | 231.18 | -7.18% |
| 8 | 398.08 | 394.63 | -0.87% |

最大绝对偏差 **7.18%**，≤20% 检查 **通过**。此项为同镜像独立进程复跑，不冒充另建 Notebook 的独立实例复验；后者尚未执行。汇总与 SHA256 索引见 `results/scnet/validation-summary.json`，原始结果分别位于 `results/scnet/validation-20261026T094057Z/`、`results/scnet/validation-20261026T094355Z/`。

`python verify.py --selfcheck` 校验保存的证据；默认 `python verify.py` 保留正式 AAS 待复验状态并返回 2。首版保留上述限制，后续优化需同卡、同框架、同负载另行配对测量。
