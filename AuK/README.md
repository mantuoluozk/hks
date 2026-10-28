# AuK · BW × SGLang-Omni 技术报告

## 1. 简介

- [SCNet 模型来源](https://www.scnet.cn/ui/aihub/models/sugon_scnet/AuK)
- [发布者模型说明](https://huggingface.co/tencent/AuK)
- [SCNet AuK 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2115345884192739329)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks) · 模型目录 `AuK/`

T04 语音生成适配，使用 AuK base 原版，搭配 Qwen2.5-Omni-3B 编码器与发布者 VAE，单张 BW。保留发布者权重与模型结构。本文报告语音合成、参考音频条件、HTTP编辑接口的功能检查，以及英文 Seed-TTS-Eval 的完整 WER 与首次性能实测。

| 验证项 | 实测结果 |
|---|---|
| 功能 | 两独立服务各9/9；七个有效音频用例跨进程PCM一致 |
| 公开精度自评 | Seed-TTS-Eval test-en，1088/1088，WER 1.7137% |
| 发布者精度对照 | WER 1.02%；增加0.6937个百分点 |
| 本报告精度判据 | WER增加不超过2个百分点，数值门槛满足 |
| 首次性能与复现 | 每进程32次计时请求、6次预热；平均延迟/音频吞吐最大偏差1.1583% |
| 镜像验证 | 新Notebook构建文件核对及一次真实中文语音HTTP检查通过 |

## 2. 环境

| 环境项 | 配置 |
|---|---|
| 平台 / 区域 | SCNet Notebook / 华中一区（A区） |
| 加速卡 | BW × 1，64GB |
| Python / DTK | Python3.10.12 / DTK26.04 |
| PyTorch | 2.11.0+dtk2604.2609201441.gf74cd7 |
| SGLang | 0.5.18+dtk2604.torch2110.2609231937.gc3bbbd |
| SGLang-Omni | 固定提交428b73225f808f0d93669140983e90839c9aea0e，模型局部兼容补丁 |
| Transformers / x-transformers | 5.12.1 / 1.31.14 |
| OpenAI Python | 2.30.0 |
| 计算精度 | 编码器BF16；DiT与VAE为FP32 |
| 镜像 | [AuK专用镜像](https://www.scnet.cn/ui/aihub/image/zack233/2115345884192739329)，运行内容位于 `/opt/auk` |

九个固定wheel隔离安装到镜像运行目录，不替换厂商PyTorch或全局SGLang。运行时将缓存和结果写入模型目录。构建记录见 `sources/image-build/`，组件与新镜像检查结果见 `results/new-image-quick-20261028/summary.json`。

## 3. 适配与配置

SGLang-Omni平台层原导入新版本SGLang的 `model_override_base.resolved_view`；当前镜像使用旧模块路径。通过模型目录内的五文件补丁回退到真实 `sglang.srt.arg_groups.overrides.resolved_view`，文件与改动哈希见 `sources/dtk-compat-files.json`，应用脚本为 `scripts/apply_dtk_compat.py`。

初始BF16 DiT出现同seed波形差异及一次NaN/HTTP500，历史证据保留于 `results/performance-20261028/`。正式配置改为编码器BF16、DiT/VAE FP32、Math SDPA与Torch确定性算法。NFE=32、CFG=2、sway=-1，最大batch=1，24kHz单声道。正式配置及稳定性处理分别见 `configs/inference.json`、`优化说明.md`。此适配不宣称完整BF16或未经验证的加速比。

## 4. 启动与复现

使用上述SCNet AuK镜像与代码目录。`HKS_ROOT` 指向含 `models/` 和 `platform/scnet-notebook/` 的仓库根目录；光合仓库采用每模型根目录布局，可用该根目录作为 `HKS_ROOT`。模型目录默认共享路径 `/root/private_data/hks/models/AuK`。

权重与数据独立准备，不随代码分发：

| 内容 | 固定来源 / 版本 | 文件要求与校验 |
|---|---|---|
| AuK base与VAE | tencent/AuK，790742b71a4430120daf2b2099192abae449eb9f | auk_base.safetensors、vae.safetensors、config.yaml、LICENSE；`sources/publisher-files.json` |
| 编码器 | Qwen/Qwen2.5-Omni-3B，f75b40e3da2003cdd6e1829b1f420ca70797c34e | 三个权重分片及完整处理器等14文件；`sources/conditioner-files.json` |
| 英文测试集 | zhaochenyang20/seed-tts-eval-arrow，27f4c1adee83b5b29b7c4b375f6b976324bda308 | `data/en-00000-of-00001.parquet`，247555423字节；`sources/seedtts-files.json` |
| WER识别器 | openai/whisper-large-v3，06f233fe06e710322aca913c1bc4249a0d71fce1 | 13个固定文件；`sources/whisper-large-v3-files.json` |

AuK与VAE的四项校验、编码器14文件与Whisper13文件校验均通过。英文parquet SHA256为 `5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca`。

```bash
cd /root/private_data/hks/models/AuK
export HKS_ROOT=/root/private_data/hks
export AUK_MODEL_PATH=/root/private_data/SothisAI/model/Aihub/AuK/main/AuK
export AUK_ENCODER_PATH="$PWD/weights/Qwen2.5-Omni-3B"
bash run.sh
```

另一终端可调用：

```bash
curl --noproxy '*' http://127.0.0.1:30005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，欢迎参加语音生成测试。","seed":42,"response_format":"wav","stage_params":{"auk_engine":{"gen_seconds":4}}}' \
  --output speech.wav
```

测试入口如下，各次运行使用新的输出目录。性能和全量精度控制器会启动并停止自有服务，运行前须停止手动启动的30005服务，避免端口冲突。

```bash
# 服务已启动时，单独功能检查
bash run-scnet.sh python scripts/http_smoke.py --output results/new-smoke
# 停止手动服务后，两独立服务完整性能检查
bash run-scnet.sh python scripts/run_performance.py --output results/new-performance
# 四个隔离评测依赖按requirements-eval.txt准备，完整英文集与Whisper放入下列目录
# data/seed-tts-eval-arrow/data/en-00000-of-00001.parquet
# evaluators/whisper-large-v3/
bash run-scnet.sh python scripts/run_quality_pilot.py --output results/new-seedtts-full --max-samples 0 --seed 1234
# 无GPU、无推理的已发布证据完整性检查
python scripts/verify_results.py
```

脚本名保留 `pilot`，参数 `--max-samples 0` 明确表示全1088条。源码与镜像重建步骤见 `重建说明.md`。

## 5. 功能测试

两独立服务均完成9项：英文生成、中文生成、seed42重复、seed43负对照、声音描述指令、参考音频条件、编辑接口、空输入拒绝、零时长拒绝。七项有效音频均为24kHz有限值非平坦波形；同seed逐采样一致，seed43产生不同PCM；七项PCM哈希也跨进程一致。编辑接口检查的是HTTP可执行性与音频合法性，没有将语义编辑效果当作已评分。

证据：`results/performance-fp32-20261028/{first,repeat}/smoke/summary.json`。专用镜像另在新Notebook核对588项构建内容及清单/校验器，单次中文HTTP200，输出4秒、96000采样点，实测PCM哈希与正式功能测试中文样本一致；测试自有服务已停止，见 `results/new-image-quick-20261028/summary.json`。

## 6. 精度测评

采用发布者公布成绩的 **Seed-TTS-Eval test-en完整1088条**，零样本参考音频语音合成；不使用其他数据集替代。发布者[技术报告表5](https://arxiv.org/html/2609.08936v1#A1.T5)中AuK base英文WER为1.02%。发布者报告三次运行均值，本次为固定seed1234的一次完整运行，未宣称随机重复次数完全一致。

| 指标 | 结果 |
|---|---:|
| 生成成功 / 评分成功 | 1088 / 1088 |
| 生成或评分执行错误 | 0 |
| 发布者英文WER | 1.02% |
| 本次英文WER | 1.7137189422% |
| WER增加 | 0.6937189422个百分点 |
| 本报告采用的允许增加量 | 不超过2个百分点 |
| 对应WER上限 / 数值判定 | 3.02% / 满足 |

“不超过2个百分点”按百分数直接相减：`1.7137189422 − 1.02 = 0.6937189422 ≤ 2`，不采用相对增加2%的计算方式。该判定为本报告采用口径下的数值对照，不替代主办方正式验收。

评分使用 `openai/whisper-large-v3` FP32，音频按官方实现以scipy重采样到16kHz，固定英文transcribe提示。评分源码取自 [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval/tree/752f4297f090c46bb1a55a1f7439e5944ddefe8d)，采用其 `process_one`：小写化、删除规定标点但保留撇号，不增加数字归一化。使用旧版BPE清理行为；各样本WER算术平均后乘100，不使用整个语料合并WER。完整保留1条WER大于50%的样本，不筛选高错样本或按正确性重试。

本次NFE32、CFG2、sway=-1；输出时长按参考音频长度与文本UTF-8长度比例估计，上限30秒，逐条参数保留。发布者完整随机运行配置未全部公开，不宣称协议逐项完全等同。只评价本节英文WER，没有测量或宣称SIM、中文CER及其他编辑基准。

汇总见 `results/seedtts-full-seed1234/generation/wer-summary.json`、`results/accuracy-comparison.json`；逐条评分、生成参数与原始文件哈希见该目录 `public/`。公开证据不分发测试集原文或参考音频，重新计算识别WER需另行获取上述固定数据。

## 7. 首次性能

计时范围为客户端HTTP请求到收到完整WAV，包含编码、生成、解码和响应传输；模型加载不计入请求延迟。两轮使用独立服务进程，完全相同的9项功能前缀；每轮测试3秒/5秒音频与并发1/2四种负载，各8次计时请求，预热按并发数每格1或2次，共32实测、6预热。服务最大batch=1，并发2不表示批量2。

| 音频时长 | 并发 | 首轮平均延迟(s) | 复跑平均延迟(s) | 首轮音频吞吐(s/s) | 复跑音频吞吐(s/s) |
|---|---:|---:|---:|---:|---:|
| 3s | 1 | 3.0147 | 3.0208 | 0.9908 | 0.9890 |
| 3s | 2 | 5.4160 | 5.4028 | 1.0290 | 1.0410 |
| 5s | 1 | 3.0783 | 3.0822 | 1.6038 | 1.6150 |
| 5s | 2 | 5.4173 | 5.4555 | 1.7236 | 1.7195 |

各负载平均延迟与音频吞吐的相对复跑偏差最大1.1583%，在20%以内。RTF、p95与逐请求延迟保留在 `results/performance-fp32-20261028/`；该最大偏差只指上述平均延迟和音频吞吐，不扩展为所有指标。没有相同负载的完整成功BF16基线，不宣称加速比。

## 8. 复现与限制

正式性能与精度在同一SCNet BW环境完成；专用镜像已完成新Notebook构建内容核对与一次真实中文HTTP快速复验，未在新Notebook重复完整性能矩阵或全量WER。两者范围分别记录，不能将快速复验当作全量复测。

报告使用的专用镜像见开头SCNet链接。权重、固定依赖、局部源码补丁、启动参数、原始计时数据和逐条评分均提供来源或证据。`results/evidence-manifest.json` 对发布文件逐项校验，`scripts/verify_results.py` 可在CPU核算题数、WER均值、性能汇总和比较门槛。本版满足本报告采用的WER增加不超过2个百分点的数值判据；赛事正式验收结论由主办方审核。
