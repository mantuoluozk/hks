# Trident-Qwen3VL-2B · BW × vLLM 技术报告

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Trident-Qwen3VL-2B)
- [发布者模型说明](https://huggingface.co/OpenBMB/Trident-Qwen3VL-2B)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Trident-Qwen3VL-2B/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 6 条多模态输入及 HTTP embedding |
| 公开精度自评 | ChartQA Origin NDCG@10 ×100 = 69.4483 |
| 首次性能与复现 | 每进程 1536 请求；复跑最大偏差 3.88% |
| 镜像验证范围 | 复用 ASR 镜像与随仓库 Python 依赖；同 Notebook 两个独立进程验证 |
| 发布者精度对照 | ChartQA Origin：发布者 70.19，实测 69.4483 |
<!-- report-resources:end -->

T02 混合模态检索嵌入模型，输入文本、图片或图文融合内容，输出 2048 维 L2 归一化向量。使用 [SCNet AI 社区模型](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Trident-Qwen3VL-2B)，13 个运行文件逐项 SHA256 校验通过。BF16 权重 4,255,140,312 字节，SHA256 `984c6dad40e50c68497764e363b4980634a3c82ea9396d395e81e0668e62ea2f`；未替换模型参数，LoRA 已由发布者合并。许可与固定来源见 `sources/`。

## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；设备与运行参数见下文 |
| 框架 | vLLM；实际组件版本见下文 |
| 镜像用途 | ASR 专用镜像；其他模型按报告复用 |
| 镜像版本 | `v1` |
| 镜像地址 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1` |
| 访问状态 | 私有，提交时开放 |
| 镜像复验 | 复用 ASR 镜像与随仓库 Python 依赖；同 Notebook 两个独立进程验证 |
<!-- report-environment:end -->

SCNet Notebook，华中一区（A区），单张 BW、64GB / 80 CU、逻辑设备 0，gfx936。Python 3.10.12；实际版本：

| 组件 | 版本 |
|---|---|
| torch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| vllm | 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a |
| transformers | 5.12.1 |
| pillow | 11.3.0 |
| torchvision | 0.26.0+dtk2604.torch2110.2609101802.gee55be |

实际使用 [SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions)：`appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1`，版本 `v1`。镜像保持私有。沿用镜像推理栈，无额外系统安装或框架源码修改；`qwen-vl-utils 0.0.14` 的小型 Python 源码随本模型目录提供，通过 PYTHONPATH 加载，并保留 Apache-2.0 许可与哈希。设备、版本与框架文件指纹见 `results/scnet/validation-20261026T114901Z/preflight/summary.json`。

## 3. 适配与配置

自定义 `Qwen3VLForEmbedding` 权重的 backbone 与 Qwen3-VL 相同。使用 vLLM 现有 `Qwen3VLForConditionalGeneration` 加通用 embedding adapter，`runner=pooling`、`convert=embed`、`seq_pooling_type=LAST`、`use_activation=true`，不生成文本，不加载额外生成头。BF16、eager、上下文 8192、最大序列数 8、显存比例 0.45，prefix cache 与多模态预处理缓存关闭；CPU 数学库线程均固定 4。

query 指令为 `Represent the query for retrieving relevant content.`，doc 指令为 `Represent the candidate content for retrieval.`；以 system 角色注入，图像在用户文本之前，保留 assistant generation prefix，对最后有效 token 池化。图像 min/max pixels 为 262144 / 1310720。原始发布者实现与解析、格式化及评分代码固定在 `OpenBMB/Trident@7fbf4b2033e1b3a43a99185fc125d765a549c52d`。保留原始文件，未将上游 CUDA 依赖清单直接安装到 DTK 环境。

## 4. 启动与复现

代码部署在 `/root/private_data/hks/models/Trident-Qwen3VL-2B`，配套 `platform/scnet-notebook` 环境脚本放在项目根目录；Notebook 已是容器。权重采用 `run-scnet.sh` 默认社区目录，或通过 `MODEL_PATH` 指定完整权重路径。

```bash
cd /root/private_data/hks/models/Trident-Qwen3VL-2B
RUN_RESULTS_DIR="$PWD/results/reproduce" bash run-scnet.sh bash run.sh
RUN_RESULTS_DIR="$PWD/results/benchmark" bash run-scnet.sh bash bench.sh
# 本机 HTTP embedding 服务：
bash run-scnet.sh python scripts/serve_embeddings.py --host 127.0.0.1 --port 8000
```

`POST /v1/embeddings` 接受 `{"input":"检索问题","role":"query"}`；doc 角色可使用扩展格式 `{"input":[{"text":"图表说明","images":["/绝对路径/chart.png"]}],"role":"doc"}`。当前仅支持 float 输出、2048 维、每次 1–8 条输入、每条最多 4 个本地图片路径；返回 data/embedding、index 与实际输入 token 用量。测试覆盖两图输入，不宣称四图均已验证。该桥接服务串行处理 HTTP 请求；批量性能不是 HTTP 并发性能。

公开 ChartQA 数据准备：`python scripts/prepare_chartqa.py`，按固定版本下载、核对 Git blob / LFS 哈希并安全解压。再指定新的 RUN_RESULTS_DIR 和已生成的 REFERENCE_RESULTS，执行 `bash run-scnet.sh python scripts/evaluate_chartqa.py --backend native`；参考后端改为 `--backend publisher`。源代码包不包含权重或数据集。

## 5. 功能测试

文本 query、文本 doc、单图、图文融合、无关文本和双图文档共 **6 条输入**通过，均为有限数值的 2048 维归一化向量。双图用同一演示图表的两个副本检查多图位置与编码链路，不作为多图语义精度样本。与同 BW 上的发布者 BF16 / SDPA 实现逐向量对照，最低余弦相似度 **0.99962413**，所有输入的展开 token 序列一致。原始向量、相似度及排序见 `results/scnet/validation-20261026T114901Z/publisher_reference/` 和 `native_embedding/`。

HTTP 接口两次成功请求覆盖同样 6 条输入，与直接 native 调用对照通过；非法 role 返回 400。接口响应维度、归一化、用量与对照结果见 `results/scnet/validation-20261026T114901Z/native_embedding/api-summary.json`。这些检查属于功能与实现一致性，不能替代检索精度。

## 6. 精度测评

发布者论文 [Table 1](https://arxiv.org/abs/2610.11816v1) 第 8 页给出 ChartQA Origin NDCG@10 ×100 **70.19**，Mix **65.82**；后者使用六个数据集合并候选库，不能与单 ChartQA 比较。查证及来源指纹见 `官方精度基线查证.md`、`sources/public-baseline.json`。

完整公开 ChartQA Origin：1,250 查询、1,509 图表候选，固定数据版本 `15c41335843f8880e24778843b462234b6427e4f`，逐文件哈希校验通过。使用发布者解析与指标代码，BF16 推理，向量按上游默认 float16 保存后评分；未删除失败查询、未丢弃缺失 gold。两后端均 0 查询失败、0 缺失候选标注。

| 对照 | NDCG@10 ×100 |
|---|---:|
| 发布者论文 GPU 参考 | 70.19 |
| 发布者实现 / 同一 BW | 69.5031 |
| native vLLM / BW | 69.4483 |

native 相对同 BW 参考差 **-0.0548 pt**，相对论文参考差 **-0.7417 pt**。完整 Recall / nDCG / MRR / MAP、每查询 top-10 排序及 token 数分别见 `results/scnet/release-20261026T113612Z/publisher_chartqa/`、`native_primary/chartqa/`。论文环境与本次 DTK 环境不同，分数差异原样记录。本报告精度范围为完整 ChartQA Origin，不包含六数据集 Mix。

## 7. 首次性能

文本、图片、图文融合三条固定演示输入 × batch 1/2/4/8 × 两轮，每格 64 条计数请求，共 **24 格 / 1,536 请求 / 进程**，不是 1,536 条独立样本。每格先预热两次，以下报告第二轮。累计同步 `llm.embed` 调用耗时，包含输入预处理、模型执行及向量返回，排除模型初始化、图片文件读取、payload 构造及结果审计。每个图片位置均使用独立 UUID 防止视觉编码缓存复用，并关闭 prefix cache 与预处理缓存。吞吐为请求/秒，越高越好。

| 输入 | batch | 首次请求/秒 | 独立进程复跑 | 偏差 |
|---|---:|---:|---:|---:|
| text | 1 | 25.27 | 24.29 | -3.88% |
| text | 2 | 25.34 | 24.84 | -1.98% |
| text | 4 | 49.66 | 48.66 | -2.03% |
| text | 8 | 96.18 | 94.52 | -1.73% |
| image | 1 | 11.27 | 11.06 | -1.87% |
| image | 2 | 11.56 | 11.74 | +1.56% |
| image | 4 | 17.50 | 17.17 | -1.90% |
| image | 8 | 20.64 | 21.01 | +1.81% |
| fused | 1 | 11.45 | 11.41 | -0.37% |
| fused | 2 | 11.52 | 11.90 | +3.30% |
| fused | 4 | 18.15 | 17.48 | -3.71% |
| fused | 8 | 21.68 | 21.86 | +0.82% |

首版完整原始记录见 `results/scnet/release-20261026T113612Z/native_reproduction/performance/summary.json`；复跑见 `results/scnet/validation-20261026T114901Z/native_embedding/performance/summary.json`。这是同步批处理的公开自测协议，不以生成文本的输出 token/s 或 TPOT 解释 embedding 性能，也不冒充赛事最终负载。固定执行顺序与两个进程的成绩不提供统计置信区间，不据此声明模型优化奖励。

## 8. 复现与限制

同一个 SCNet Notebook、同一镜像、同一 BW 上，两个独立初始化进程按相同缓存策略与负载运行。第二轮 12 个配对档位最大绝对吞吐偏差 **3.88%**，≤20% 检查 **通过**。本项复现范围为同一 Notebook 内的两个独立进程。提交前开放镜像访问。

`python verify.py --selfcheck` 校验来源文件、公开测评与功能/性能证据哈希。逐项证据清单见 `results/scnet/validation-summary.json`。报告记录已完成的验证，赛事验收结论以主办方审核为准。
