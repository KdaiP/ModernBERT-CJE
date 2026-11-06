# ModernBERT-CJE：现代中文基座模型

[English README](README_EN.md)

ModernBERT-CJE 是**面向中、日、英三语的现代文本编码器基座模型**，基于现代**ModernBERT 架构、从零开始预训练**，以支持 NLP、TTS 及 System 1 类模型（如 Jev）等应用场景。我们提供 **large（主干192.971M参数）** 和 **tiny（主干4.525M参数）** 两个版本，并开放模型权重、推理示例、预训练与蒸馏相关代码。

| 模型 | 主干参数量 | 最大序列长度 | 中文语料规模 | 语种支持 | 权重链接 |
| --- | ---: | ---: | ---: | --- | --- |
| [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | 302.309M | 512 | 5.4B | 中文 | — |
| **ModernBERT-CJE-large (ours)** | **192.971M** | **4096** | **约 120B** | **中文、英文、日文** | [🤗](https://huggingface.co/KdaiP/ModernBERT-CJE-large) |
| **ModernBERT-CJE-tiny (ours)** | **4.525M** | **512** | **约 120B** | **中文、英文、日文** | [🤗](https://huggingface.co/KdaiP/ModernBERT-CJE-tiny) |

与 RoBERTa-wwm-ext-large 相比，**ModernBERT-CJE-large 主干参数量减少约 36.2%**，**最大序列长度提升至 8 倍**（512 → 4,096），**CLUE 九项均分提升 0.38 分**（75.96 → 76.34），添加了更多训练数据，原生支持中、日、英三语。

同时，我们也提供蒸馏版本**ModernBERT-CJE-tiny**，采用了激活初始化和关系蒸馏等技术，相比于从零初始化训练的版本，更好地保留了教师的能力。

## CLUE 中文语言理解测评基准

基于 [CLUE v1](https://github.com/CLUEbenchmark/CLUE) 正式版九项任务评测：

| 模型 | 主干参数量 | 分类六项均分 | 阅读三项均分 | 九项均分 |
| --- | ---: | ---: | ---: | ---: |
| **ModernBERT-CJE-large** | 192.971M | **75.53** | **77.95** | **76.34** |
| [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | 302.309M | 75.10 | 77.68 | 75.96 |
| **ModernBERT-CJE-tiny** | 4.525M | 65.31 | 55.89 | **62.17** |
| [ALBERT tiny](https://huggingface.co/clue/albert_chinese_tiny) | 1.212M | 65.80 | 52.67 | 61.42 |
| [CLUE RoBERTa tiny](https://huggingface.co/clue/roberta_chinese_clue_tiny) | 4.689M | 64.96 | 52.76 | 60.90 |
| [CLUE RoBERTa 3L312](https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny) | 3.517M | 64.15 | 51.08 | 59.79 |
| [UER RoBERTa L6 H128](https://huggingface.co/uer/chinese_roberta_L-6_H-128) | 1.190M | 59.74 | 41.08 | 53.52 |

**ModernBERT-CJE-large** 的 CLUE 九项均分达到 **76.34**，超过 RoBERTa-wwm-ext-large 的 **75.96**；**ModernBERT-CJE-tiny** 九项均分达到 **62.17**，优于表中其他轻量级对比模型。


## 快速开始

### 1. 环境准备

模型可使用`transformers`直接调用，以下示例在 Python 3.12、PyTorch 2.11.0、Transformers 5.20.0 环境验证：

```bash
pip install torch transformers
```


### 2. 提取特征

以下示例加载 **ModernBERT-CJE-large**，提取**倒数第三层**的 token 级隐藏状态：

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path =  "KdaiP/ModernBERT-CJE-large"
text =  "你好，世界！Hello, World!  こんにちは、世界！"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    features = outputs.hidden_states[-3].cpu()

# [batch=1, token 数, hidden_size=1024]，包含 [CLS]、[SEP]。
print("文本：", text)
print("Tokens：", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))
print("特征形状：", tuple(features.shape))
print(features)
```

## 下游 TTS 任务示例

我们提供一个基于 Bert-VITS2 文本前端的 `word2ph` 示例，将**中文、日文、英文**的 ModernBERT-CJE token 级特征映射到音素长度，以用于TTS输入，适用于 [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)、[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 等模型。

使用方式见 [`word2ph_example/README.md`](word2ph_example/README.md)。**安装及运行命令均在 `word2ph_example/` 目录下执行。**

## 预训练与蒸馏

我们提供 **large 预训练**，**tiny 初始化** 和 **tiny 蒸馏**相关脚本。

完整说明及命令见 [`train/README.md`](train/README.md)。**训练命令在 `train/` 目录下执行。**


## CLUE 评测结果

以下是 [CLUE v1](https://github.com/CLUEbenchmark/CLUE) 正式版九任务：六项分类（AFQMC、TNEWS、IFLYTEK、CMNLI、CLUEWSC2020、CSL）和三项阅读理解（CMRC2018、CHID、C3）的评测结果：

### 逐任务结果

| 任务 | ModernBert-CJE-large | HFL RoBERTa-large | ModernBert-CJE-tiny | ALBERT tiny | CLUE RoBERTa tiny | CLUE 3L312 | UER L6 H128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AFQMC | **75.32** | 75.25 | 70.23 | 69.01 | 69.08 | 69.05 | 69.02 |
| TNEWS | 58.81 | **59.01** | 53.95 | 53.80 | 54.34 | 53.86 | 51.88 |
| IFLYTEK | 60.90 | **60.98** | 51.90 | 55.62 | 56.15 | 56.83 | 39.54 |
| CMNLI | **84.93** | 83.31 | 74.96 | 72.89 | 71.61 | 70.30 | 66.44 |
| CLUEWSC2020 | 88.60 | **89.47** | 65.35 | 67.21 | 66.34 | 64.69 | 63.60 |
| CSL | **84.64** | 82.56 | 75.46 | 76.23 | 72.24 | 70.16 | 67.98 |
| CMRC2018 EM | 67.97 | **72.04** | 54.81 | 48.82 | 53.12 | 49.72 | 34.83 |
| CHID | **86.48** | 86.06 | 65.64 | 58.22 | 57.37 | 54.82 | 43.30 |
| C3 | **79.39** | 74.93 | 47.20 | 50.97 | 47.80 | 48.71 | 45.12 |

### 微调与复现配置

每个模型、每个任务均以 **seed 42 完整搜索四个学习率**，选择开发集最高分；随后固定学习率，从同一预训练检查点独立微调 seed 43 和 44。每次训练按开发集指标选择最佳 epoch。

| 任务 | Epoch | Batch | 最大长度 | Warmup 比例 | 学习率网格 |
| --- | ---: | ---: | ---: | ---: | --- |
| AFQMC | 3 | 16 | 128 | 0.1 | `1e-5, 2e-5, 5e-5, 1e-4` |
| TNEWS | 3 | 16 | 128 | 0.1 | `1e-5, 2e-5, 5e-5, 1e-4` |
| IFLYTEK | 10 | 32 | 128 | 0.1 | `1e-5, 2e-5, 5e-5, 1e-4` |
| CMNLI | 2 | 64 | 128 | 0.1 | `2e-5, 3e-5, 5e-5, 1e-4` |
| CLUEWSC2020 | 50 | 8 | 128 | 0.1 | `1e-5, 2e-5, 5e-5, 1e-4` |
| CSL | 5 | 4 | 256 | 0.1 | `5e-6, 1e-5, 2e-5, 5e-5` |
| CMRC2018 | 3 | 32 | 512 | 0.1 | `2e-5, 3e-5, 1e-4, 2e-4` |
| CHID | 3 | 24 | 64 | 0.06 | `1e-5, 2e-5, 3e-5, 5e-5` |
| C3 | 8 | 24 | 512 | 0.1 | `2e-5, 5e-5, 1e-4, 2e-4` |

### 统一训练设置

| 配置项 | 值 |
| --- | --- |
| 优化器 | AdamW |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `0.01` |
| 梯度裁剪 | `1.0` |
| 学习率调度 | Linear warmup + linear decay |

## 特别鸣谢

- [zhiyongchenGREAT](https://github.com/zhiyongchenGREAT)
- [aizhiqi-work](https://github.com/aizhiqi-work)
- [ylzz1997](https://github.com/ylzz1997)
- [CNChTu](https://github.com/CNChTu)
- [Stardust·减](https://github.com/Stardust-minus)
- [PuddingEgg](https://github.com/PuddingEgg)
