# Qwen3-ASR-1.7B T03/T04 补测材料

- 固定条件任务：AISHELL-1、LibriSpeech test-clean、LibriSpeech test-other 各100条，共300条。
- 结果：300/300完成，300成功，0失败。
- 推理设置：BF16、temperature=0、最大输出1024、语种自动识别；模型请求未传参考文本或语种标签。
- 完整输出包：`Qwen3-ASR-outputs-and-feedback.zip`，包含300个逐条识别文本、官方反馈表、环境信息及汇总。
- 逐条输出位于 `outputs/`，反馈明细见 `反馈表.csv`。
- 可读环境说明见 `环境说明.txt`；统计汇总见 `summary.json`。
