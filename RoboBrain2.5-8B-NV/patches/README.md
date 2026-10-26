# 确定性模式兼容修复

重放入口：`scripts/apply_deterministic_lds_patch.py`。`run.sh` 在 `ENABLE_DETERMINISTIC_INFERENCE=1` 时自动调用；适用于本报告固定的 SGLang / DTK 镜像。

原始确定性 MM/BMM 使用三阶段流水，在 BW100 上申请 98304 字节共享内存，超过设备 65536 字节上限。修复降低 HIP 设备上的流水级数，并为 FP32 点积显式指定 IEEE 输入精度。修改范围、前后源文件 SHA256 见 `results/deterministic-lds-patch.json`，矩阵核测试见 `results/deterministic-gemm-check.json`。普通 BF16 权重、tokenizer、图像处理配置不变。

BF16 GEMM/BMM 的批次不变性与数值检查通过，FP32 BMM 对照通过；随后模型级功能、重复请求和图执行前后比较分别记录。不能把矩阵核通过当作正式 GPU 动作一致率通过。

修改发生在本模型容器内的框架运行文件；原始文件备份在模型缓存中。干净基础镜像通过相同脚本重放，不依赖已修改容器的快照。

底层代码来自 [SGLang](https://github.com/sgl-project/sglang)。`deterministic-mm-bmm.patch` 是针对固定镜像源码生成的实际差异，启动时使用 Python 重放脚本并校验前后 SHA256；安装包许可证如可用则保留为 `LICENSE-SGLang.txt`。


## 高分辨率 BMM 边界修复（2026-10-26）

CV-Bench 数据预跑触发确定性视觉 BMM 非法地址访问。新增 `deterministic-bmm-large-output.patch`，使输出 C 超过 2³¹ 元素时批次偏移使用 int64。完整重放后的框架 SHA256 为 `371f9816ae1614d944dce1616ee2e88a6d1b6dfed2ca660275ae28cb9785bb39`；原始源码及前版哈希固定，拒绝修改未知源码。

2,164,260,864 个输出元素的边界测试、小矩阵 BF16/FP32 数值及批次一致性测试通过，真实图片 8 例请求无错误。证据见 `results/scnet/publisher-runtime-c64-20261026/`。全量公开精度复测进行中，使用静态显存比例 0.5 为视觉临时矩阵留出空间；首版性能仍按原配置标注，不据此声明新版本性能收益。完整协议见 `公开基准复测.md`。
