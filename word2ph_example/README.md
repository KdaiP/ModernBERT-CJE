# 三语 BERT 特征对齐到音素示例

本示例基于 [Bert-VITS2 的文本前端](https://github.com/fishaudio/Bert-VITS2/tree/master/text)，将 ModernBERT-CJE 的中文（`ZH`）、日文（`JP`）、英文（`EN`）特征展开为 `[hidden_size, 音素数]`形状。

**本文所有命令都在当前目录下执行。** 默认使用 Hugging Face 上的 `KdaiP/ModernBERT-CJE-tiny`模型。

## 安装与运行

创建环境并准备前端：

```bash
pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict
python prepare_frontend.py
```

分别运行三种语言：

```bash
python example.py --language ZH --text '你好，世界！' --output zh.pt
python example.py --language JP --text 'こんにちは、世界！' --output jp.pt
python example.py --language EN --text 'Hello, world!' --output en.pt
```

使用 large、GPU，或启用 Bert-VITS2 的 blank 插入方式：

```bash
python example.py --model KdaiP/ModernBERT-CJE-large --device cuda --language ZH
python example.py --language EN --text "It's a beautiful day!" --add-blank --output en_blank.pt
```

## 映射过程

`word2ph[i]` 表示第 `i` 个 **模型输入 token** 对应多少个音素，包括 `[CLS]` 和 `[SEP]`这两个特殊 token。

例如中文 `你好!`：

```text
tokens:   [CLS]  你     好      !  [SEP]
word2ph:    1    2      2      1    1
phones:     _   n i   h ao     !    _
```

## 上游来源与改动

前端固定到 [Bert-VITS2 master 的提交 `3e710e9`](https://github.com/fishaudio/Bert-VITS2/tree/3e710e996f03d52476490c435899e76d688dcbf1/text)。[prepare_frontend.py](prepare_frontend.py) 保留Bert-VITS2的文本规范化、G2P、音素表和词典相关代码，仅做以下适配：

- 入口只导出音素表，移除自动下载三个语言的原始 BERT 的初始化流程。
- 移除英文、日文 tokenizer 的导入时加载。
- 日文按字符分配音素；英文用 `sep_text` 得到完整单词后发音，再在示例中分配到 CJE 子词。
