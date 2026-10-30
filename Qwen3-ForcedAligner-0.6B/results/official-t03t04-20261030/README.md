# Qwen3-ForcedAligner-0.6B T03/T04 补测材料

- 固定条件：英文 LibriSpeech test-clean 30 条，单条音频不超过 15 秒；BF16、seed 42，使用条件给定文本。
- 结果：30/30 成功，逐条输出词级时间戳 JSON；反馈 CSV 完整且无失败。
- 校验：条件 ID 和反馈表一致；输出词序与给定文本一致；时间戳单位为秒，单调并落在音频边界范围内。
- 完整输出包：`Qwen3-ForcedAligner-outputs-and-feedback.zip`，内含 30 个输出 JSON、反馈 CSV、环境 JSON 和生成汇总。
- 可读环境说明：`环境说明.txt`；逐项结果：`反馈表.csv`；校验摘要：`verified-summary.json`。
