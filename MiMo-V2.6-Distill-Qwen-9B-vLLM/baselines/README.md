# 基线与比较协议

本目录以本次同 Notebook 的 vLLM eager 为基础优化基线。eager 和 decode graph 均为 BF16、FP32 SSM 状态、32768上下文、seq64、mem0.80、CPU4线程，关闭prefix与多模态缓存；原生512 token-ID/128输出，并发1/4，每档8计时请求，额外1/4个32-token预热。每候选只执行一遍短测。原始结果、运行参数及源码在 results/selection.json 指向的 screen 目录。

历史 SGLang 短测为512输入、128输出、并发1/4；正式矩阵为4096/16384输入、1024输出、并发1/8/64。[历史报告](https://developer.sourcefind.cn/codes/zhengke/gxgm-hks)中的 MiMo-V2.6-Distill-Qwen-9B/ 为跨框架历史参考，不充当本次同框架优化基线。

精度最低参考仍为 Qwen/Qwen3.5-9B。本版本仅评估与 SGLang SCNet 同一200题 GSM8K，seed20261023，准备文件SHA03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546。提示与评分函数从原 public_eval.py 原样提取，零样本、greedy、top_p1、thinking、max4096、workers8。sglang-gsm8k-200.json 保留历史179/200逐题判分与输入hash，本次成绩见精度任务 summary；历史MMLU成绩不迁移。它不是Qwen3.5-9B或发布者Agent基准同协议验收对照。
