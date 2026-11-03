# 运行输入

数据不随代码分发。`examples/`包含中英文演示WAV，来源与SHA256见`../sources/audio-sources.json`；`official-20261030/`包含官方30条条件与原音频，清单SHA256见`../sources/official-t03t04-20261030.json`。

在已准备首版共享输入的Notebook执行`bash prepare_data.sh`建立本目录的输入软链接，不复制或下载数据。其他环境可设置`ALIGNMENT_INPUT_ROOT`指向同样结构的现有数据根目录后执行。专项脚本在加载模型前校验条件及30条音频SHA256。
