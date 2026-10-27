# Qwen-Image-2.1-Turbo · BW × SGLang 技术报告

## 1. 简介

<!-- report-resources:start -->
- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen-Image-2.1-Turbo)
- [发布者模型说明](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
- [SCNet 镜像与版本](https://www.scnet.cn/ui/aihub/image/zack233/2114893402085822465/versions)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `Qwen-Image-2.1-Turbo/`

| 验证项 | 当前结果 |
|---|---|
| 功能 | 两个独立进程各 8/8 |
| 公开精度自评 | CLIP score / FID 待官方评测 |
| 首次性能与复现 | 每进程 9 实测 + 3 预热；复跑最大偏差 17.09% |
| 镜像验证范围 | 专用镜像已完成新 Notebook 复验 |
| 官方精度与验收 | 待官方评测代码、参考与容差对齐；未判定通过 |
<!-- report-resources:end -->

当前阶段：专用镜像 1.0.0 已在新 Notebook 完成环境预检——镜像内 1093 个构建文件、10 个 wheel 的 4,881 个安装文件 SHA256 全部通过，依赖导入路径与服务 launcher 同构建记录一致；两个独立服务进程各自执行相同的 8 项功能检查，两轮全部通过；自测性能两轮复测，三档最大偏差 17.09%，未超过 20% 复现门槛。正式 T03 精度尚未下发，`official_accuracy_pass=null`，本报告不宣称官方验收通过。

正式 T03 图像指标为公示 prompt/条件集上的 CLIP score/FID 子集，对照 GPU 参考劣化 ≤2%；固定 seed 或肉眼观察不能替代该精度验收。

权重来自 [SCNet AI 社区](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Qwen-Image-2.1-Turbo)，4 个 safetensors 共 32,440,133,648 字节，全部 BF16。许可为 Qwen Research License。官方实现见 [Qwen 模型页](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 与 [源码仓库](https://github.com/QwenLM/Qwen-Image-2.1)。

## 2. 环境

<!-- report-environment:start -->
| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 / 精度 | BW × 1 / BF16；设备与运行参数见下文 |
| 框架 | SGLang；实际组件版本见下文 |
| 镜像用途 | Qwen-Image 专用镜像 |
| 镜像版本 | `1.0.0` |
| 镜像地址 | `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0` |
| 访问状态 | 私有，提交时开放 |
| 镜像复验 | 专用镜像已完成新 Notebook 复验 |
<!-- report-environment:end -->

SCNet Notebook，华中一区（A区），单张 BW、gfx936、80 CU、总显存 68,702,699,520 字节；Python 3.10.12。专用镜像为 `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0`，版本 1.0.0，保持私有，[镜像版本页](https://www.scnet.cn/ui/aihub/image/zack233/2114893402085822465/versions)。

新 Notebook 预检通过：镜像内 1093 个构建文件、10 个 wheel 对应的 4,881 个安装文件 SHA256 校验全部通过；实际依赖导入路径与镜像构建记录一致，服务 launcher 与构建时相同。预检记录及环境证明见 `results/scnet/image-1.0.0-preflight/summary.json`。

厂商 Torch distribution 版本为 `2.11.0+dtk2604.2609201441.gf74cd7`；SGLang 0.5.18。隔离使用 Transformers 5.17.0、tokenizers 0.23.1，以及兼容 Python 3.10 的 Diffusers 提交 `ea981b8b7d80f10deaff369bcda8cbf2c90428dd`（0.41.0.dev0），不覆盖厂商框架。版本及实际导入路径见 `results/scnet/image-1.0.0-preflight/summary.json`。

<!-- report-components:start -->
| 组件 | 版本 |
|---|---|
| PyTorch（distribution） | 2.11.0+dtk2604.2609201441.gf74cd7 |
| SGLang | 0.5.18 |
| Transformers（隔离） | 5.17.0 |
| tokenizers（隔离） | 0.23.1 |
| Diffusers（隔离） | 0.41.0.dev0；提交 ea981b8b7d80f10deaff369bcda8cbf2c90428dd |
<!-- report-components:end -->

## 3. 适配与配置

SGLang 使用 generic Diffusers 后端，未声明原生优化算子或性能提升。进程内保留新版 Transformers 内置 ASR 注册，恢复视觉配置原生 RoPE 标准化方法，保持 axial 配置；通用后端保留完整条件图片列表，交由官方 pipeline 预处理。未修改 checkpoint 或框架安装文件。导入前后配置一致性见 `results/config-probe-fixed/config-comparison.json`，具体边界见 `优化说明.md`。

单卡 BF16、CPU 数学库线程 4、CPU Generator、seed=42、true_cfg_scale=1、use_kv_cache=true，保留 checkpoint 内置 8 个 sample_sigmas。SGLang 当前自动策略关闭 DiT CPU 卸载，保留文本编码器 CPU 卸载；两轮性能使用同一策略。完整配置见 `runtime-config.json`。

## 4. 启动与复现

代码放在 `/root/private_data/hks/models/Qwen-Image-2.1-Turbo`，配套项目根目录 `platform/scnet-notebook` 环境脚本；Notebook 已是容器。权重路径使用包装器默认社区目录，或设置 MODEL_PATH。

```bash
cd /root/private_data/hks/models/Qwen-Image-2.1-Turbo
bash env_check.sh
bash run.sh  # 前台服务，另一个终端执行 smoke.sh
bash smoke.sh
# 停止已有服务、确保30003端口空闲，再跑两轮独立服务自测：
bash bench.sh
# 交付自测，预期退出码 0
python verify.py --selfcheck
# 精度验收：官方 CLIP/FID 与 GPU 参考下发前保持待验，返回退出码 2
python verify.py
```

上述 verify 命令校验随本报告冻结的证据；新 bench 复跑产生独立结果目录及逐档偏差，不自动覆盖本报告成绩。代码也可在光合仓库的模型独立目录运行，HKS_ROOT 指向包含 platform 的项目根目录。

`bench.sh` 在每个服务进程内先执行相同 8 项功能检查，再按每档 1 次预热 + 3 次实测采样。generations 接口使用 JSON，edits 接口使用 multipart `image` 或 `image[]`，返回 base64 PNG。包装器优先使用镜像内 `/opt/qwen-image21`，其次使用模型目录 runtime；`HKS_REQUIRE_BAKED_RUNTIME=1 bash env_check.sh` 可拒绝误用共享依赖。

## 5. 功能测试

512×512 单次 pipeline 兼容性探测耗时 7.604 秒，初始化 35.015 秒，峰值分配显存 37,361,179,136 字节。该结果包含逐步有限值检查，属于基础功能探测，不是 HTTP 服务性能，也不纳入第 7 章性能表。原始记录位于 `results/scnet/runtime-20261026T124357Z/smoke512/summary.json`。

新镜像下两个独立服务进程各执行相同 8 项功能检查，两轮全部通过：512 文生图、相同 seed 重复、第二个 prompt、1024 中文海报、2048 文生图、单图编辑、双图编辑、seed=43 负对照。各次均 HTTP 200、输出尺寸正确、像素有效；相同 seed 重复输出逐像素一致，seed=43 与 seed=42 输出不同。人工抽查中文海报、茶壶变绿与黄鸟组合图，符合演示提示。证据见 `results/scnet/service-validation-20261027T015635Z/`。

同 BW Diffusers 参考图与 SGLang 512 输出的最大差为 1 个 8 位像素单位、平均绝对差约 0.0994，原样记录，不将编码或量化差异解释为正式精度。后端请求 trace 已核对 CPU Generator、CFG=1、尺寸及单/双条件图的实际传入。

以下为新镜像复验的原始样例，使用 BF16、CPU Generator、seed=42 和原生 8 步采样。样例用于展示功能，不替代正式精度评测。

**中文海报生成（1024×1024）**：白色背景、深蓝色“国芯国模”文字和红色圆环。

![Qwen-Image 中文海报生成样例](results/scnet/service-validation-20261027T015635Z/round-1/smoke/text1024.png)

**单图编辑（512×512）**：将原图的红色茶壶改为绿色，保留蓝色杯子与木桌。

| 输入图片 | 编辑结果 |
| --- | --- |
| ![红色茶壶输入图](results/scnet/service-validation-20261027T015635Z/round-1/smoke/text512.png) | ![茶壶变绿的单图编辑结果](results/scnet/service-validation-20261027T015635Z/round-1/smoke/edit-one.png) |

**双图编辑（512×512）**：将第一张参考图的茶壶与第二张参考图的黄鸟组合在木桌场景中。

| 参考图一 | 参考图二 | 合成结果 |
| --- | --- | --- |
| ![茶壶参考图](results/scnet/service-validation-20261027T015635Z/round-1/smoke/text512.png) | ![黄鸟参考图](results/scnet/service-validation-20261027T015635Z/round-1/smoke/bird512.png) | ![茶壶与黄鸟的双图合成结果](results/scnet/service-validation-20261027T015635Z/round-1/smoke/edit-two.png) |

## 6. 精度测评

T03 要求公示 prompt/条件集上的 CLIP score / FID 子集，与 GPU 参考按官方单位及容差判定。评测代码、固定输入及 GPU 参考文件尚未下发，`official_accuracy_pass=null`。未声明正式精度或赛事验收通过。

## 7. 首次性能

测量协议：n=1、并发 1 的串行 HTTP 请求；提示词固定，seed=42、CPU Generator、8 步原生 pipeline。每个服务进程性能段共 12 次请求——512/1024/2048 每档 1 次预热（不计入）+ 3 次实测，合计 3 次预热、9 次实测。计时覆盖 HTTP 请求发出至完整 base64 PNG 响应接收，解码与校验在计时外。表中均值、中位数、范围与吞吐直接取自 `round-1/benchmark/summary.json` 与 `round-2/benchmark/summary.json`；吞吐为均值倒数，不是并发吞吐。每档仅 3 次实测，不报 P95/P99 等小样本高分位；官方 T03 最终负载尚未下发，本轮为自测协议（`official_performance_protocol=false`），不冒充官方性能结论。

表 7-1 两轮独立服务进程串行 HTTP 实测

| 尺寸 | 轮次 | 均值（秒/张） | 中位数（秒） | 范围（秒） | 吞吐（张/秒） |
| --- | --- | ---: | ---: | ---: | ---: |
| 512×512 | 第 1 轮 | 23.4626 | 21.5351 | 18.3517–30.5011 | 0.04262 |
| 512×512 | 第 2 轮 | 21.0037 | 20.3585 | 15.6441–27.0085 | 0.04761 |
| 1024×1024 | 第 1 轮 | 31.5648 | 31.2079 | 27.7426–35.7439 | 0.03168 |
| 1024×1024 | 第 2 轮 | 26.1695 | 29.2538 | 19.1984–30.0562 | 0.03821 |
| 2048×2048 | 第 1 轮 | 41.1050 | 40.9601 | 40.9259–41.4290 | 0.02433 |
| 2048×2048 | 第 2 轮 | 34.9789 | 35.8162 | 31.8665–37.2540 | 0.02859 |

表 7-2 复现偏差（相对第 1 轮均值）

| 尺寸 | 第 1 轮均值（秒/张） | 第 2 轮均值（秒/张） | 偏差（绝对值） |
| --- | ---: | ---: | ---: |
| 512×512 | 23.4626 | 21.0037 | 10.48% |
| 1024×1024 | 31.5648 | 26.1695 | 17.09% |
| 2048×2048 | 41.1050 | 34.9789 | 14.90% |

三档偏差均未超过 20%，最大偏差为 1024 档的 17.09%。原始记录：`results/scnet/service-validation-20261027T015635Z/`（`status.json`、`round-1/benchmark/summary.json`、`round-2/benchmark/summary.json`）。

## 8. 复现与限制

- 新镜像 1.0.0 已在全新 Notebook 复验：预检哈希与依赖路径一致，两个独立服务进程各 8 项功能全部通过，性能三档偏差 10.48% / 17.09% / 14.90%，最大 17.09% ≤ 20%，通用复现门槛通过。
- 正式 CLIP score / FID 子集与 GPU 参考文件尚未下发，评测代码与固定输入待统一；`official_accuracy_pass=null`，不声明正式精度通过或赛事验收通过。
- 镜像 1.0.0 仍为私有；正式精度及官方验收另行记录。
- 技术限制（历史记录）：`results/scnet/service-validation-20261027T005955Z/` 中 512 档两轮偏差为 70.24%，且两个进程在性能采样前的前置步骤不同——第 1 进程先执行了 8 项功能检查，第 2 进程没有相同步骤，当时未定位 512 差异成因。该结果与 70.24% 偏差原样保留、不删除，作为未达到复现门槛的失败记录；前置差异与波动的因果关系尚未确定，不改写为优化提升；当前脚本已统一两轮前置流程，并已在新镜像完成复验。
- 首版属于 BF16 兼容适配，不声明优化奖励；后续优化版须在相同卸载策略下配对测试并回归正式精度。
