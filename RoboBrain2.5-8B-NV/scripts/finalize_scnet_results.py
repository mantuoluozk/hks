"""Generate the eight-section first-release report only from complete SCNet evidence."""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    runtime = read('results/scnet/runtime/runtime-config.json')
    assert runtime['EXECUTION_PLATFORM'] == 'SCNet Notebook'
    assert runtime['CARD'] == 'BW' and runtime['HIP_VISIBLE_DEVICES'] == '0'
    assert runtime['ENABLE_DETERMINISTIC_INFERENCE'] == '1'
    weights = read('results/scnet/runtime/weights-check.json')
    assert weights['all_required_files_match'] and len(weights['files']) == 13
    kernel = read('results/scnet/runtime/deterministic-gemm-check.json')
    assert kernel['first_row_byte_exact_across_batch_sizes']
    assert all(c['first_batch_exact_match'] for c in kernel['bmm_checks'])
    patch = read('results/scnet/runtime/deterministic-lds-patch.json')
    assert patch['after_sha256'] == '9668a00e1f8d068d3528c5f6fea55b7ec6994259ebf42fc2edf031525aa87a1b'
    smoke_path = sorted((ROOT / 'results/scnet').glob('smoke-*/summary.json'))[-1]
    smoke = json.loads(smoke_path.read_text())
    assert smoke['execution_platform'] == 'SCNet Notebook'
    assert smoke['functional_pass'] and smoke['diagnostic_semantic_pass'] and len(smoke['cases']) == 8
    candidates = sorted((ROOT / 'bench').glob('scnet-matrix-*/audit.json'))
    assert candidates, 'Complete SCNet matrix audit missing'
    bench = candidates[-1].parent
    audit = json.loads((bench / 'audit.json').read_text())
    assert audit['complete'] and len(audit['workloads']) == 12
    assert sum(c['completed'] for c in audit['workloads']) == 608
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_matrix.py'), str(bench)],
                   check=True, env=dict(os.environ, EXECUTION_PLATFORM='SCNet Notebook'))
    rows = [json.loads(line) for line in (bench / 'round2.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == 6
    environment = read('results/scnet/runtime/environment.json')
    assert environment['visible_device_count'] == 1
    platform_path = ROOT / 'results/scnet/platform-spec.json'
    platform = json.loads(platform_path.read_text()) if platform_path.is_file() else {}
    optional = {}
    for name in ['stability', 'concurrency']:
        files = sorted((ROOT / 'results/scnet').glob(name + '-*/summary.json'))
        if files:
            optional[name] = {'path': str(files[-1].parent.relative_to(ROOT)), 'summary': json.loads(files[-1].read_text())}
    summary = {'created_at': datetime.now(timezone.utc).isoformat(),
               'model_id': runtime['MODEL_ID'], 'execution_platform': 'SCNet Notebook',
               'runtime': runtime, 'weights_verified': 13,
               'smoke': {'path': str(smoke_path.parent.relative_to(ROOT)), 'passed': 8, 'total': 8},
               'performance': {'path': str(bench.relative_to(ROOT)), 'rounds': 12, 'requests': 608, 'reported_round': 2},
               'same_service_diagnostics': optional,
               'official_accuracy_pass': None, 'official_acceptance_passed': None,
               'clean_image_full_matrix_deviation_pass': None}
    (ROOT / 'results/scnet/validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    verification = {'model_id': runtime['MODEL_ID'], 'execution_platform': 'SCNet Notebook',
                    'status': 'official_reference_missing', 'official_accuracy_pass': None,
                    'required_consistency_percent': 98,
                    'required': 'Official image/depth inputs, per-case GPU outputs and parser/comparison protocol.',
                    'functional_passed': 8, 'functional_total': 8,
                    'functional_checks_are_not_accuracy': True}
    for rel in ['verify_result.json', 'results/scnet/verify_result.json']:
        (ROOT / rel).write_text(json.dumps(verification, indent=2) + '\n')
    versions = '\n'.join(f'| {name} | `{environment["packages"][name]}` |' for name in ['torch', 'sglang', 'sglang-kernel', 'triton', 'transformers', 'flash_attn'])
    performance = '\n'.join(f'| {r["random_input_len"]} | {r["max_concurrency"]} | {r["completed"]} | {r["output_throughput"]:.2f} | {r["mean_ttft_ms"]:.2f} | {r["mean_tpot_ms"]:.2f} | {r["total_input_tokens"]} |' for r in sorted(rows, key=lambda r: (r['random_input_len'], r['max_concurrency'])))
    smokes = '\n'.join(f'| {c["case"]} | PASS |' for c in smoke['cases'])
    diagnostics = []
    if 'stability' in optional:
        cases = optional['stability']['summary']['cases']
        stable = sum(c['unique_answers'] == 1 for c in cases)
        diagnostics.append(f'同一服务重复具身提示：{stable}/{len(cases)} 组输出组内完全一致，逐次响应见 `{optional["stability"]["path"]}/`。')
    if 'concurrency' in optional:
        c = optional['concurrency']['summary']
        diagnostics.append(f'4 路并发响应与同服务串行结果逐字比较：{c["exact_matches"]}/{c["requests"]} 一致，请求错误 {c["errors"]}；原始记录见 `{optional["concurrency"]["path"]}/`。')
    diagnostic_text = '\n\n'.join(diagnostics)
    command = (ROOT / 'results/scnet/runtime' / ('server-' + runtime['SERVER_RUN_ID'] + '.log')).read_text().splitlines()[0].rstrip()
    region = platform.get('region', '运行环境未提供区域字段，以 Notebook 页面登记为准')
    specification = platform.get('specification', '容器 CPU 配额 15 核，内存上限 59 GiB；磁盘配额以 Notebook 页面为准')
    report = f'''# RoboBrain2.5-8B-NV on BW + SGLang 0.5.18

## 1. 简介

本项目完成 `BAAI/RoboBrain2.5-8B-NV` 在超算互联网单张 BW 加速卡上的 SGLang BF16 serving 适配。模型采用 Qwen3-VL 架构，支持文本、单图、多图输入，以及文本、坐标和轨迹输出；机器人实机任务不在本次部署范围。

- [模型下载](https://www.scnet.cn/ui/aihub/models/sugon_scnet/RoboBrain2.5-8B-NV)
- [发布者模型说明](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)
- [SCNet 镜像](https://www.scnet.cn/ui/aihub/image/zack233/2114287084172722177)
- [光合代码仓库](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)，模型目录 `RoboBrain2.5-8B-NV/`。

## 2. 验证环境

SCNet Notebook，BW × 1，65520 MiB、80 CU、gfx936，逻辑设备 0，TP=1、BF16。Ubuntu 22.04.5，驱动 6.3.31-V1.5.3.beta，Python 3.10.12、DTK 26.04、HIP 6.3.26113。

| 组件 | 版本 |
|---|---|
{versions}

区域：{region}。Notebook 规格：{specification}。服务仅监听 `127.0.0.1:30125`，未使用外部服务映射。

基础镜像：`image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`。

固定摘要：`sha256:98dca2a455d62b758a2455e674fb480bc39ac236a2b3b472de8a1525743ec8a3f`。启动时自动重放第 3 节的固定源码补丁；未额外安装或升级推理依赖包。

权重通过 SCNet AI 社区克隆获取，路径 `{runtime['MODEL_PATH']}`。13 个关键模型文件通过固定 SHA256 校验；说明文件和仓库属性文件不参与运行文件一致性校验。权重、结构、tokenizer 和图像处理配置未修改。环境及校验记录见 `env.log`、`results/scnet/runtime/`。

## 3. 适配优化方案

使用 Triton 文本注意力、SDPA 视觉注意力、确定性推理；上下文长度 32768，静态显存比例 0.75，最大运行请求数 64，chunked prefill 4096；完整解码图最大批次 32，关闭预填充图。

补丁作用于 `sglang/srt/batch_invariant_ops/batch_invariant_ops.py`。原确定性 MM/BMM 的流水配置申请 98304 字节共享内存，超过设备 65536 字节上限。修复根据 HIP 设备共享内存容量限制流水级数，并为 FP32 点积显式指定 IEEE 输入精度。

重放脚本为 `scripts/apply_deterministic_lds_patch.py`，实际差异及许可证位于 `patches/`。脚本校验补丁前 SHA256 `9a9fdb89d84f8c392ae9a46f46c0a846b332e78f4ae77c24f240f8db55e6a6e4`，补丁后 SHA256 `{patch['after_sha256']}`，拒绝修改其他版本的源码。

矩阵核检查中，BF16 GEMM 对 FP32 参考相对 L2 误差为 {kernel['relative_l2_error_vs_fp32']:.8f}，首行跨 batch 大小逐字节一致；BF16/FP32 BMM 数值及首批次一致性检查通过，记录见 `results/scnet/runtime/deterministic-gemm-check.json`。这些检查不能替代模型级 GPU 精度验收。

## 4. 服务启动脚本

```bash
bash run-scnet.sh
```

`run.sh` 校验权重、重放补丁、启动服务并轮询 `/health`。实际执行命令为：

```bash
{command}
```

`MODEL_PATH` 可覆盖。日志、PID 和实际参数写入 `results/scnet/runtime/`；`stop.sh` 仅停止 PID、模型路径、服务名称和端口均匹配的本模型服务。

## 5. Smoke 测试

`/health` 返回 200，`/v1/models` 包含 `robobrain25-8b`。8 项功能和诊断检查全部通过：

| 测试 | 结果 |
|---|---|
{smokes}

原始请求、响应、流式内容、图片与汇总见 `{smoke_path.parent.relative_to(ROOT).as_posix()}/`。坐标/轨迹检查验证格式和范围，不代表任务精度。

{diagnostic_text}

## 6. 精度测评

T02 具身 serving 验收要求在公示图像（含深度图）输入集上，解析动作/坐标与 GPU 参考输出一致率 ≥98%。当前缺少对应输入集、逐例 GPU 参考输出及比较协议，`official_accuracy_pass=null`，不填写 PASS。

发布者论文中的空间理解与轨迹基准成绩不能替代逐例 GPU 一致率；同服务重复稳定性和功能格式检查也不作为该一致率分数。正式状态与缺失项见 `verify_result.json`；来源索引见 `官方精度基线查证.md`。

## 7. 性能参考

每请求一张 1080p 随机图，文本长度设置 4096/16384、固定输出 1024，并发 1/8/64，N=max(8,2C)。每档连续两轮，报告第二轮；每档开始前清空 prefix cache。12/12 轮、608/608 个请求完成，生成 token 数和配置校验通过。

图像和模板会增加实际输入 token 数，下表的文本长度设置不是固定总多模态输入长度。TTFT/TPOT 为客户端测得的平均值。

| 文本长度设置 | 并发 | 请求数 | 输出 tokens/s | TTFT ms | TPOT ms | 实际输入 token 总数 |
|---:|---:|---:|---:|---:|---:|---:|
{performance}

原始两轮 JSONL、逐档日志、参数和审计见 `{bench.relative_to(ROOT).as_posix()}/`。本次结果作为平台首版性能基准，后续优化在相同卡型、框架和负载下比较。

## 8. 注意事项

在 SCNet Notebook 的共享存储中部署本模型目录，以及随代码提供的 `platform/scnet-notebook/env-scnet.sh`：

```bash
mkdir -p /root/private_data/hks/platform/scnet-notebook
cp platform/scnet-notebook/env-scnet.sh /root/private_data/hks/platform/scnet-notebook/
bash env_check.sh
bash run.sh                 # 在独立终端保持运行
bash smoke.sh
python verify.py            # 缺少官方参考时记录未判定并返回 2
BENCH_PROFILE=matrix bash bench.sh
```

性能和其他推理测试串行执行；16K、64 并发存在排队，延迟如实记录。首次启动会编译内核，启动超时为 1800 秒。尚未完成新 Notebook 中完整矩阵的性能偏差复现，不声明该项通过。权重不随代码仓库分发。
'''
    assert len(re.findall(r'^## [1-8]\. ', report, re.M)) == 8
    assert not any(word in report for word in ['用户', '自有', '历史', '备份', '协作'])
    (ROOT / 'README.md').write_text(report, encoding='utf-8', newline='\n')
    (ROOT / '优化说明.md').write_text('''# 部署配置与兼容性修复

基于固定 SCNet SGLang 0.5.18 / DTK 26.04 镜像，BF16、单卡 TP1。运行脚本自动校验并重放确定性 HIP MM/BMM 补丁，限制共享内存流水级数并使用 IEEE FP32 输入精度；不修改模型权重、结构、tokenizer 或图像处理配置。

Triton 文本注意力、SDPA 视觉注意力、确定性推理、full decode graph（最大批次 32），关闭 prefill graph；上下文长度 32768、静态显存比例 0.75、最大运行请求 64、chunked prefill 4096。

本版记录首次平台性能基准，未把其他硬件上的速度差异作为优化收益。具体环境、补丁前后 SHA256、矩阵核检查、功能、精度状态和性能原始证据见 README.md 与 results/scnet/。
''', encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
