# 输入与固定资源

模型来源为 XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B，固定 revision 2367e865d009c13ac81713a2878291d33ab28177。完整快照放于 env.sh 的 MODEL_PATH，15运行文件清单及SHA256见 sources/weight-files.json；env_check.sh 不下载权重。

功能图片由 scripts/smoke.py 本地生成，原始请求与结果保留在各服务 functional/。无需下载外部数据集。

性能使用镜像内 SGLang random-ids 原生成器、seed1、固定长度，直接发送 native token-ID 列表至 vLLM /v1/completions，无聊天模板或图片。脚本保存 inputs.json、逐请求 request/response、usage与计时。正式4096/16384输入、1024输出、并发1/8/64，每格N=max(8,2C)，各两次608条；另12个32-token预热。

发布视图可用 scripts/evidence_codec.py 将连续循环ID列表无损编码；解码后SHA必须匹配完整ID列表。图片按内容SHA去重，原始字节路径与转换映射见 sources/evidence-view-map.json。原始共享记录保持不变。

GSM8K仅测试与原SGLang相同的固定200题，不跑全量或第二个精度集。原始Parquet来自 openai/gsm8k revision740312add88f781978c0658806c59bc2815b9866，main/test-00000-of-00001.parquet，419088字节，SHA256 ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59。文件放入data/gsm8k/test.parquet或通过GSM8K_PARQUET指定；prepare_gsm8k.py按原seed20261023重建并严格核对200题JSONL SHA03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546。不自动下载。

完整请求载荷和数据文件保留在共享data/，不入库；发布逐题原始响应、数值预测、输入/请求hash、usage、计时和评分。通过固定数据、prompt函数及协议可重建完全相同的请求。取消的全量安排仅保留历史部分记录，不计入200题成绩。
