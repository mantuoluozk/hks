# SCNet 镜像重建

构建基础镜像：`image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18`，使用此前已成功构建的内部地址，固定摘要待补充。保留 Jupyter、SSH 与厂商 Torch；离线重建当前 ASR 镜像的八个固定补充包，再添加 Qwen-Image 隔离依赖。私有社区拉取地址不作为构建 FROM。

模型依赖先由 `scripts/prepare_runtime_local.py` 准备，再由 `scripts/prepare_image_context.py` 核对固定 wheel 哈希并生成白名单构建目录。八个基础补充包来源及哈希见 `sources/base-extra-wheel-files.json`，由 `--base-extra` 指定已有 wheel 目录。Diffusers 固定源码提交，安装过程不联网、不安装模型权重、不全局覆盖推理框架。

SCNet 的 Dockerfile COPY 上下文固定使用家目录 `dockerFileTemp`，参见 [平台规则](https://www.scnet.cn/help/docs/mainsite/container-service/image-management/build-rules/)。提交构建路径：

```text
/public/home/zack233/dockerFileTemp/Dockerfile.qwen-image21-turbo
/public/home/zack233/dockerFileTemp/qwen-image21-turbo/
```

Dockerfile 仅 COPY `qwen-image21-turbo/`。安装前逐项核验 `SHA256.json` 中的 1,093 个源文件及 wheel：模型依赖安装到 `/opt/qwen-image21/site-packages`，八个基础补充包安装到 `/opt/qwen3-asr-deps`，均使用 `--no-deps --no-index`；Diffusers 源码放入 `/opt/qwen-image21/diffusers-py310`。构建目录没有权重、数据集、凭据或连接配置。

实际镜像为 `appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-image21-turbo:1.0.0`，版本 `1.0.0`，[版本页](https://www.scnet.cn/ui/aihub/image/zack233/2114893402085822465/versions)，华中一区（A区），保持私有，摘要待提交前补充。开发工具复用已安装的 `/opt/conda/bin/jupyter`。

新 Notebook 实测 1,093 个构建文件及 10 个 wheel 对应的 4,881 个已安装文件哈希通过；运行依赖来自镜像 `/opt/qwen-image21`，服务脚本与镜像内哈希一致。两个独立服务进程各8项功能检查通过，三档性能复跑最大偏差17.09%。证据分别见 `results/scnet/image-1.0.0-preflight/` 和 `results/scnet/service-validation-20261027T015635Z/`；正式CLIP/FID与GPU参考仍待官方代码，不声明官方精度通过。

新 Notebook 中执行：

```bash
cd /root/private_data/hks/models/Qwen-Image-2.1-Turbo
HKS_REQUIRE_BAKED_RUNTIME=1 bash env_check.sh
bash run.sh
# 另一个终端：
bash smoke.sh
```

preflight 须证明 Diffusers、Transformers 和 tokenizers 从镜像的 `/opt/qwen-image21` 加载，Torch 仍为厂商 DTK 版本。逐项复验后再将新镜像作为报告实际环境。
