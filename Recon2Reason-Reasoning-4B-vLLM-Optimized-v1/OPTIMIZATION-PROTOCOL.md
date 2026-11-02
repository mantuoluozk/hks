# Recon vLLM 优化版 v1 实验协议

2026-11-02。单 BW 64GB，原 Recon 专用镜像，TP1 BF16。只比较 eager 与 FULL_DECODE_ONLY；mode=0，capture sizes=1,2,4,8,16,32,64。保持32768上下文、mem0.80、max_num_seqs64、线程4、默认prefix cache/chunked prefill、多图上限2。

筛选严格复用原validate.py文本负载：提示重复32/256，128固定输出，temperature0、ignore_eos，每长度1次预热，每并发1/2/4各8请求；每配置仅一遍。分别执行原5项多模态功能检查，保存请求/响应与源码。根据实际completion_tokens重算吞吐；逐条对照请求、usage、可见文本。功能失败、图不兼容即停止该候选。

明确收益后才运行优化版单服务正式六档：4K/16K文本长度参数、1024输出、并发1/8/64、N=max(8,2C)、每档两次取第二次，总608请求，seed1、1张1080p随机JPEG、vllm-chat、stream_options.include_usage=true。原版完整1216请求不重跑，独立服务复现记null。保留九类指标和全部档位。

原版正式参考results/formal-20261031T163337Z/process1/round2.jsonl；短测不与正式矩阵混算。若新配置出现空可见文本，必须取得新配置实际token IDs，核对对应请求、输入及显示选项；不自动接受历史豁免、不删除样本。不新增精度套件。
