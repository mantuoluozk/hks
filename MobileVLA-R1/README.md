# MobileVLA-R1 · BW × vLLM 技术报告

## 1. 简介

- [发布者模型与权重](https://huggingface.co/AIGeeksGroup/MobileVLA-R1)
- [发布者代码](https://github.com/AIGeeksGroup/MobileVLA-R1) · [原版论文](https://arxiv.org/html/2511.17889v1)
- [实际使用的 SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114893402085822465/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `MobileVLA-R1/`

| 验证项 | 本版结果 |
|---|---|
| 功能 | 两个独立 HTTP 服务进程各 12 项通过；包括单图、双图、八图 |
| 参考对照 | 两种合成单图的贪心输出与模块化 Transformers 参考逐 token 一致 |
| 首次性能与复现 | 每进程 12 请求（9 计数、3 预热）；各档平均延迟复跑最大偏差 4.64% |
| 镜像验证范围 | 复用 Qwen-Image 镜像；同 Notebook 两个独立服务进程，未另建实例复验 |
| 官方精度 | 尚未执行闭环导航或 QUARD 任务成功率评测 |
| 赛事验收 | 当前为 RGB 功能与性能版本，不声明精度或官方验收通过 |

适配对象为原版 **MobileVLA-R1 的 RL 检查点**，不混用 MobileVLA-R1 2.0。权重固定版本 `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a`；26 个 ZIP 分卷共 26,871,863,199 字节，逐项通过发布者 LFS SHA256 校验。仅解压其中 RL 检查点，共 16,998,243,602 字节，逐文件 ZIP CRC 和 SHA256 已记录；不需要先拼接额外一份完整 ZIP。来源与证据见 `sources/archive-manifest.json`、`results/archive-check.json`、`results/extracted-rl.json`。

## 2. 环境

| 环境项 | 实际配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1，64 GB / BF16 |
| 镜像版本 | `1.0.0` |
| 镜像地址 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0` |
| Python | `3.10.12` |
| torch | `2.11.0+dtk2604.2609201441.gf74cd7` |
| vLLM | `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a` |
| transformers / openai | `5.12.1` / `2.30.0` |
| pillow / safetensors | `11.3.0` / `0.8.0` |
| fastapi / uvicorn | `0.115.12` / `0.53.0` |

实际版本与运行配置保存在 `results/runtime-config.json`。使用镜像全局 Transformers 及既有 `/opt/qwen3-asr-deps` API 依赖，不使用 Qwen-Image 隔离的 Transformers / Diffusers 栈；未新增系统依赖，未修改厂商 PyTorch 或 vLLM。

## 3. 适配与配置

采用发布权重中的 SigLIP 视觉编码器、`mlp_downsample` 投影器及 Llama3-8B 语言模型。视觉编码选倒数第二层 `cls_patch` 特征，27×27 patch 补齐后进行 2×2 下采样，得到每图 196×4096 视觉向量；与原始文本嵌入按图像占位顺序拼接，再通过 vLLM `prompt_embeds` 接口生成。视觉与语言参考权重加载均无缺失或多余键，投影权重严格加载。

这是**发布者 RGB 前端与 vLLM 语言后端的组合适配**，不声明 vLLM 原生支持完整 MobileVLA 架构，也不声明已实现论文深度/点云分支。每个 HTTP 请求实时处理 RGB 图片，不读取预先缓存的视觉嵌入。

发布者源码固定 `9d98cce3b623dfddf479740d78869ec30364b75a`，对应文件 Git blob 与 SHA256 见 `sources/reference-files.json`。运行脚本在核验文件哈希后，仅提取原始 `DownSampleBlock`、聊天模板、图像占位分词、`observation_question` 和 `process_image` 等必要定义，不执行上游环境安装或训练脚本。

Transformers 5 的 SigLIP 处理器暴露 `crop_size=None`，原始图像函数的 `hasattr` 判断会误选该字段。本模型局部将其设为同一处理器的 `size`，保持原始 resize 路径的 384×384 目标尺寸。服务通过 FastAPI lifespan 显式关闭 vLLM 引擎，避免退出后子进程残留占用显存。

运行配置为单卡、上下文 4096、最大运行序列 1、显存比例 0.50、eager、禁用 prefix cache、CPU 线程 4。视觉前端与语言调用串行加锁；接口最多 8 张 RGB 图片。实测使用 temperature=0、seed=42、最大输出 64 token；服务允许输出预算 1–512，且输入加输出不能超过上下文。

## 4. 启动与复现

模型代码放置在 `$HKS_ROOT/models/MobileVLA-R1`，其中 `HKS_ROOT` 默认为 `/root/private_data/hks`，公共平台脚本放在 `$HKS_ROOT/platform/scnet-notebook`。光合仓库克隆后，将模型目录放入该结构，公共平台目录保持同一根目录。Notebook 已是容器，直接运行以下命令。

从上述固定权重版本取得全部 26 个 `weight.zip.part-*` 分卷，设其实际目录为 `ARCHIVE_DIR`。首次解压前执行：

```bash
cd /root/private_data/hks/models/MobileVLA-R1
mkdir -p results
bash run-scnet.sh python scripts/inspect_archives.py --path "$ARCHIVE_DIR" --verify
bash run-scnet.sh python scripts/extract_rl.py --path "$ARCHIVE_DIR"
```

提取结果为本模型 `weights/mobilevla-r1-rl/{llm,vision_tower,mm_projector,...}`。解压程序拒绝覆盖已存在文件；已有完整检查点不应重复解压。权重及分卷不随代码分发。

启动服务：

```bash
bash run-scnet.sh python scripts/serve.py --port 30006
```

服务仅监听 `127.0.0.1:30006`，提供 `GET /health`、`POST /generate`。生成请求采用 `images`（1–8 个 Base64 PNG/JPEG 字符串）、`prompt`、`max_tokens`、`temperature`、`seed`；输出文本、token IDs、结束原因和计数。该接口是本适配提供的 RGB 接口，不声称与 OpenAI 或发布者远程 API 完全兼容。

在另一终端执行：

```bash
bash run-scnet.sh python scripts/check_http.py --out results/reproduce-functional
bash run-scnet.sh python scripts/bench_http.py --out results/reproduce-performance
# 对随代码保留的原始证据进行 CPU 独立复算，不启动模型：
python scripts/verify.py
```

`check_http.py`对照随仓库提供的 `results/reference-probe-v5/summary.json`。需要重新生成模块化参考时，可先单独运行 `bash run-scnet.sh python scripts/probe_reference.py`，完成后再启动 HTTP 服务；参考与服务应顺序运行，避免同时加载两套模型。性能输出使用新的空目录，保留每次实测证据。

## 5. 功能测试

有效 HTTP 记录为 `results/http-v5/`、`results/http-v6/`，两个独立服务进程均执行相同 12 项检查：健康检查；红色单图、相同输入重复、蓝色单图；双图和八图；无图片、空提示、无效 Base64、未支持的 depth 输入、超过图片上限及无效输出预算的拒绝检查。

两条单图输出分别与模块化 Transformers 参考逐 token 一致，同输入贪心复跑一致，2/8 图请求返回正常。参考对照在忽略最后一个结束 token 的表示差异后比较全部生成 token，原始 IDs 均保留。此结果验证指定样本上的链路与数值输出一致，不等同一般视觉语义正确率；合成色块输入的回答存在额外形状描述，未将其判为语义正确。

## 6. 精度测评与复测难点

本版**没有官方基准实测精度**。原版论文公布 R2R-CE / RxR-CE 的 val-unseen 闭环导航指标，以及 QUARD 六项任务的成功率；这些指标依赖实际环境交互与轨迹，不能用图片聊天、动作格式通过率或训练集上的静态对答代替。[原版论文第5节](https://arxiv.org/html/2511.17889v1#S5)

| 发布者基准 | 论文数值 | 本版测量状态 |
|---|---|---|
| R2R-CE val-unseen | SR 68.3%，SPL 65.2% | 未测 |
| RxR-CE val-unseen | SR 71.5%，SPL 66.8% | 未测 |
| QUARD | 六任务平均成功率约73%；每任务25回合 | 未测 |

复测存在以下具体难点：

1. **检查点与论文配置未对应。** 已解压检查点包含 RGB 视觉、投影和语言权重，未发现论文完整深度/点云分支对应权重，不能随机补齐后对照完整模型成绩。论文表6另列 RGB-only 的 R2R SR62.5% / SPL59.0%、RxR SR66.0% / SPL61.5%，但尚无充分依据确认该消融检查点就是本版发布权重。[论文表6](https://arxiv.org/html/2511.17889v1#S7)
2. **需要闭环导航环境。** 发布者提供 R2R/RxR 的 `mobilevla.yaml` 评测配置，但仍须准备 Habitat / VLN-CE 环境、对应场景资产与 val-unseen 任务，并核对观察输入、动作解析、终止和评分协议。静态图片集合不能复现 SR/SPL。[发布者评测说明](https://github.com/AIGeeksGroup/MobileVLA-R1#evaluation-and-remote-api)
3. **深度来源影响可比性。** 发布者明确区分外部深度估计器与 Habitat 仿真深度；直接替换观察来源会改变协议。QUARD 的实际机器人/环境执行接口也未由该代码补丁完整提供，不能仅根据输出文字计算论文任务成功率。[发布者说明](https://github.com/AIGeeksGroup/MobileVLA-R1#evaluation-and-remote-api)

因此，本版只发布已完成的 RGB 功能、参考输出对照和性能结果，精度字段保留“未测”；不推断免测，也不宣称达到论文基线。

## 7. 首次性能

端到端计时从客户端发出 HTTP 请求到收齐 JSON 响应，包含图片解析、实时视觉编码与投影、语言生成及传输，不包含模型加载。合成图片为 384×384 红/蓝色块；每档并发1、先1次预热、再3次计数。每进程合计12请求，其中9次计数；两轮功能前缀相同。

| 图片数 | 首轮均值(s) | 复跑均值(s) | 首轮输出token/s | 复跑输出token/s | 均值延迟偏差 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.1407 | 1.1692 | 31.5590 | 30.7903 | 2.50% |
| 2 | 1.5721 | 1.6451 | 32.4400 | 31.0016 | 4.64% |
| 8 | 1.8512 | 1.9122 | 29.7105 | 28.7625 | 3.30% |

输出 token/s 为该档三次实际生成 token 总数除以端到端耗时总和，含视觉前端耗时，不是纯 decode 速度。各档每请求实际生成36/51/55 token（含结束 token），最大预算64；不同输出长度与图片数量共同影响延迟，不据此直接比较图片数的快慢。三档平均延迟相对首轮的最大复跑偏差4.63997%，均在20%内。未测大并发，也无同协议原生框架基线，不声明优化加速比。

## 8. 复现与限制

镜像与代码共同部署；当前验证为同一 Notebook 内两个独立服务进程，不等同重新创建 Notebook 后的完整复验。该镜像未内置本模型适配代码，需同时部署本模型目录及公共平台脚本。仅验证本报告指定的 RGB 合成输入、贪心生成与串行负载；不扩展为机器人安全执行、真实导航或全部视觉任务通过。

首次探测及重启失败证据保留在 `results/history/`：包括探测脚本缺少 forward/主入口保护、依赖导入、处理器字段兼容与引擎退出清理问题。最终有效结果以 `reference-probe-v5`、`http-v5`、`http-v6`为准，不将失败记录覆盖为成功。

`scripts/verify.py`可在CPU从原始请求独立重算24项功能、24条性能请求、参考token对照及复跑偏差；发布文件哈希见`results/evidence-manifest.json`，发布者源码哈希另见`sources/reference-files.json`。本版完成范围为RGB功能与性能适配，精度与赛事正式验收分别记录。
