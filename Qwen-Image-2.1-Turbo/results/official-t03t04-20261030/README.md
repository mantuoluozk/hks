# Qwen-Image-2.1-Turbo T03/T04 固定条件实测（2026-10-30）

- 固定输入：GenEval 60 条 prompt，每条使用 seed 42、2026、1234，共 180 项。
- 结果：180/180 成功，0 失败；每项均生成 2048×2048 PNG。
- 推理设置：单张 BW，BF16，8 步，`guidance_scale=1.0`，checkpoint `sample_sigmas`，`use_kv_cache=true`。
- KV cache 通过 SGLang `diffusers_kwargs` 传入 QwenImage21Pipeline；模型 `causal_condition=true`。`guidance_scale` 映射为 pipeline 参数 `true_cfg_scale`，数值均为 1.0。
- 运行环境与参数：见 `环境说明.txt`、`environment.json`；逐项官方反馈见 `反馈表.csv`；计数汇总见 `summary.json`。
- 生成脚本：模型目录 `scripts/run_official_conditions.py`。运行前需将官方条件清单放在脚本声明的数据路径；本仓库不分发条件数据集。
- 180 张原始 PNG 与以上反馈材料的完整 ZIP 保存在 SCNet Notebook 共享存储：`/root/private_data/hks/models/Qwen-Image-2.1-Turbo/artifacts/official-t03t04-20261030T105602Z/Qwen-Image-outputs-and-feedback.zip`（约 1.31 GB）。
- 本轮完成固定条件图片生成，未计算 CLIP score / FID。
