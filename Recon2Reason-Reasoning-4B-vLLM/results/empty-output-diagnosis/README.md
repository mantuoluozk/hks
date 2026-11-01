# 随机压测请求的特殊 token 输出核对

完整矩阵在16K文本长度、64并发的固定随机输入中，每次的索引71和98没有可见文本。四次重复共8条，全部仍计入1216条矩阵请求和吞吐分母，不删除、不替换原始输出。

`full-index{71,98}-skip{False,True}.response.json` 保存每条1024-token重放的后端 token IDs。两个请求均生成1024个ID151673，即模型词表中标记为special的 `<|feat_pad|>`；显示开关两侧token IDs完全相同。隐藏特殊token时文本为空，显示时为该标记的1024次重复。词表定义见 `sources/publisher-tokenizer-config.json`。

`publisher-reference.json` 是同一BF16权重、Transformers 5.12.1/SDPA路径的64新token独立参考。两个输入均生成64个ID151673，与vLLM输出的前64个ID相同。此项是对两个异常输入的定位，不是全量精度基准，也不宣称发布者路径完成了1024-token复跑。

原始官方客户端只根据可见内容设置TTFT：上述请求TTFT为0、ITL为空。这些值表示没有可见内容，不能解释为零耗时；正式表保留未修改客户端输出，说明该限制。没有改变压测请求、采样、模型权重或正式原始记录。

初次64-token控制样本70的两次回答不是逐ID相同，`driver.log` 中的断言失败保留，不作为显示开关逐bit一致的证据。随后独立完整定位仅针对异常71/98；`full-driver.log`保存四次1024-token回放。该脚本只对选定请求计算CPU输入计数，其他行的统计占位值不是性能数据；正式矩阵数据始终来自未修改的框架压测客户端。

CPU核验仅在上述实际token IDs、显示文本、词表定义、输入计数和发布者参考全部吻合时，才识别这两个已确认的特殊token输出；其他意外空输出仍拒绝。`scripts/test_formal_verifier.py`包含修改后端token与修改参考token的两项反例检查。
