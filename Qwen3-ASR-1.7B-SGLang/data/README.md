# 固定输入与准备

脚本不下载大文件。权重默认社区目录为 `/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B`，可设置MODEL_PATH。11运行文件长度/SHA及文件级修订见sources/weight-files.json；两片4698521512字节。

本目录优先读取data/examples/asr_en.wav、asr_zh.wav。来源及SHA见sources/audio-sources.json，分别2172458和134570字节，保持原始48kHz/16kHz输入。

完整test-clean源：ModelScope openslr/librispeech_asr固定修订2a4a7918cd87e2f0e315e576b405ae7b7a65d83f，clean/test/0000.parquet，350452636字节，SHA256 7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0。准备至data/librispeech-source/test-clean.parquet和data/LibriSpeech/test-clean/；LIBRI_PARQUET、LIBRI_TEST_CLEAN可覆盖。prepare_data.py只校验既有文件，对全部2620个FLAC及参考文本与源Parquet逐一核对，不重写音频。

官方300条件准备至data/official/formal/conditions/qwen3_asr.jsonl，保留其相对音频路径布局；OFFICIAL_CONDITIONS可覆盖。原顺序/ID/音频/seed42/temperature0/max1024/自动语种/BF16保持不变，AISHELL-1/test-clean/test-other各100。实际条件文件及音频SHA映射见results/official-inputs.json。

共享旧ASR数据可通过OLD_ASR_ROOT只读复用；独立目录输入优先，不需要运行旧vLLM模型。原始数据、参考文本及完整HTTP请求体保留共享data/，请求体按SHA去重；发布逐条输入hash、原始模型响应、计时、评分和条件映射。参考文本和语种标签不传入请求。
