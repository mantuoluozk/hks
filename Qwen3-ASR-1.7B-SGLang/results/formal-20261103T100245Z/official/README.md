# Qwen3-ASR-1.7B × SGLang · T04固定条件结果

AISHELL-1、LibriSpeech test-clean、test-other各100条，独立完成300个新请求；300成功、0失败、0截断。BF16、greedy、最大输出1024、seed42、自动语种；请求未输入参考文本或语种标签。执行完成不等同于识别精度或官方验收通过。

| 提交产物 | 位置 |
| --- | --- |
| 逐条识别文本，按原条件ID命名 | [outputs/（300个TXT）](outputs/) |
| 官方逐项反馈表，含状态及输出路径 | [反馈表.csv](反馈表.csv) |
| 可读环境说明 | [环境说明.txt](环境说明.txt) |
| 环境参数、版本及来源 | [environment.json](environment.json) |
| 执行数量与成功状态 | [summary.json](summary.json) |
| 独立输入/请求/输出核验 | [rescore.json](rescore.json) |
| 原始模型响应及条件参数 | [predictions.jsonl](predictions.jsonl) |
| 固定条件ID、顺序、参数及音频SHA映射 | [official-inputs.json](../../official-inputs.json) |

例如，[BAC009S0764W0121.txt](outputs/BAC009S0764W0121.txt)可与反馈表同ID的output_path逐项对应。所有300条文本均由本次SGLang推理生成，未复用原vLLM结果。较大的JSONL以可恢复原字节的证据引用保存，读取方法见[模型技术报告](../../../README.md#82-代码与原始数据)。
