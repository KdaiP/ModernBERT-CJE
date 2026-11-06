# ModernBERT-CJE: A Modern Chinese Foundation Model

[中文 README](README.md)

ModernBERT-CJE is a **modern text encoder foundation model for Chinese, Japanese, and English**, **pretrained from scratch using the ModernBERT architecture** to support applications such as NLP, TTS, and System 1 models (e.g., Jev). We provide two variants: **large (192.971M backbone parameters)** and **tiny (4.525M backbone parameters)**, along with model weights, inference examples, and code for pretraining and distillation.

| Model | Backbone Parameters | Maximum Sequence Length | Chinese Corpus Size | Supported Languages | Weights |
| --- | ---: | ---: | ---: | --- | --- |
| [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | 302.309M | 512 | 5.4B | Chinese | — |
| **ModernBERT-CJE-large (ours)** | **192.971M** | **4096** | **Approx. 120B** | **Chinese, English, Japanese** | [🤗](https://huggingface.co/KdaiP/ModernBERT-CJE-large) |
| **ModernBERT-CJE-tiny (ours)** | **4.525M** | **512** | **Approx. 120B** | **Chinese, English, Japanese** | [🤗](https://huggingface.co/KdaiP/ModernBERT-CJE-tiny) |

Compared with RoBERTa-wwm-ext-large, **ModernBERT-CJE-large has approximately 36.2% fewer backbone parameters**, supports **8 times the maximum sequence length** (512 → 4,096), and improves the **average score across nine CLUE tasks by 0.38 points** (75.96 → 76.34). It is trained on more data and natively supports Chinese, Japanese, and English.

We also provide a distilled variant, **ModernBERT-CJE-tiny**, which uses activation-based initialization and relation distillation to better preserve the teacher's capabilities compared with training from random initialization.

## CLUE Chinese Language Understanding Benchmark

Evaluation on the nine tasks in the official [CLUE v1](https://github.com/CLUEbenchmark/CLUE) release:

| Model | Backbone Parameters | Average: 6 Classification Tasks | Average: 3 Reading Comprehension Tasks | Average: All 9 Tasks |
| --- | ---: | ---: | ---: | ---: |
| **ModernBERT-CJE-large** | 192.971M | **75.53** | **77.95** | **76.34** |
| [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | 302.309M | 75.10 | 77.68 | 75.96 |
| **ModernBERT-CJE-tiny** | 4.525M | 65.31 | 55.89 | **62.17** |
| [ALBERT tiny](https://huggingface.co/clue/albert_chinese_tiny) | 1.212M | 65.80 | 52.67 | 61.42 |
| [CLUE RoBERTa tiny](https://huggingface.co/clue/roberta_chinese_clue_tiny) | 4.689M | 64.96 | 52.76 | 60.90 |
| [CLUE RoBERTa 3L312](https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny) | 3.517M | 64.15 | 51.08 | 59.79 |
| [UER RoBERTa L6 H128](https://huggingface.co/uer/chinese_roberta_L-6_H-128) | 1.190M | 59.74 | 41.08 | 53.52 |

**ModernBERT-CJE-large** achieves an average score of **76.34** across the nine CLUE tasks, exceeding RoBERTa-wwm-ext-large's **75.96**. **ModernBERT-CJE-tiny** achieves **62.17**, outperforming the other lightweight models listed above.

## Quick Start

### 1. Environment Setup

The models can be loaded directly with `transformers`. The following example was verified with Python 3.12, PyTorch 2.11.0, and Transformers 5.20.0:

```bash
pip install torch transformers
```

### 2. Feature Extraction

The following example loads **ModernBERT-CJE-large** and extracts token-level hidden states from the **third-to-last layer**:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "KdaiP/ModernBERT-CJE-large"
text = "你好，世界！Hello, World!  こんにちは、世界！"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    features = outputs.hidden_states[-3].cpu()

# [batch=1, number_of_tokens, hidden_size=1024], including [CLS] and [SEP].
print("Text:", text)
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))
print("Feature shape:", tuple(features.shape))
print(features)
```

## Downstream TTS Example

We provide a `word2ph` example based on the Bert-VITS2 text frontend. It maps ModernBERT-CJE token-level features for **Chinese, Japanese, and English** to phoneme-length sequences for use as TTS inputs in models such as [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) and [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).

See [`word2ph_example/README.md`](word2ph_example/README.md) (in Chinese) for usage instructions. **Run all installation and example commands from the `word2ph_example/` directory.**

## Pretraining and Distillation

We provide scripts for **large pretraining**, **tiny initialization**, and **tiny distillation**.

See [`train/README.md`](train/README.md) (in Chinese) for full instructions and commands. **Run training commands from the `train/` directory.**

## CLUE Evaluation Results

The following results cover the nine tasks in the official [CLUE v1](https://github.com/CLUEbenchmark/CLUE) release: six classification tasks (AFQMC, TNEWS, IFLYTEK, CMNLI, CLUEWSC2020, and CSL) and three reading comprehension tasks (CMRC2018, CHID, and C3).

### Results by Task

| Task | ModernBERT-CJE-large | HFL RoBERTa-large | ModernBERT-CJE-tiny | ALBERT tiny | CLUE RoBERTa tiny | CLUE 3L312 | UER L6 H128 |
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

### Fine-Tuning and Reproducibility Settings

For each model and task, we perform a **full search over four learning rates using seed 42** and select the learning rate with the highest development-set score. With that learning rate fixed, we then fine-tune independently from the same pretrained checkpoint using seeds 43 and 44. For each run, the best epoch is selected according to the development-set metric.

| Task | Epochs | Batch Size | Maximum Length | Warmup Ratio | Learning Rate Grid |
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

### Shared Training Settings

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight Decay | `0.01` |
| Gradient Clipping | `1.0` |
| Learning Rate Schedule | Linear warmup + linear decay |

## Special Thanks

- [zhiyongchenGREAT](https://github.com/zhiyongchenGREAT)
- [aizhiqi-work](https://github.com/aizhiqi-work)
- [ylzz1997](https://github.com/ylzz1997)
- [CNChTu](https://github.com/CNChTu)
- [Stardust·减](https://github.com/Stardust-minus)
- [PuddingEgg](https://github.com/PuddingEgg)
