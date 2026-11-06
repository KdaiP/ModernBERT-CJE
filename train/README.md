# ModernBert-CJE 预训练与蒸馏

本目录提供 ModernBERT-CJE 的完整预训练流程，包含 Large 模型预训练，Tiny 模型初始化和蒸馏的完整流程。

> 以下命令均在当前目录执行，配置文件中的相对路径也以该目录为基准。

## 1. 环境安装

项目在 **Python 3.12** 和 **PyTorch 2.11** 环境下测试，其他 python 和 pytorch 版本应该也是支持的。

安装完成pytorch后，通过以下命令安装其余依赖：

```bash
pip install -r requirements.txt
```

## 2. 数据准备

训练语料来自以下公开数据集：

| 语言 | 数据集 |
| --- | --- |
| 中文、英文 | [OpenBMB Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) |
| 日文 | [FineWeb2 Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) |

下载中文和英文训练集的 Parquet 文件到本地：

```bash
hf download openbmb/Ultra-FineWeb --repo-type dataset \
  --include 'data/ultrafineweb_zh/*.parquet' 'data/ultrafineweb_en/*.parquet' \
  --local-dir data/ultra-fineweb
```

下载日文训练集的 Parquet 文件到本地：

```bash
hf download hotchpotch/fineweb-2-edu-japanese --repo-type dataset \
  --include 'data/train-*.parquet' \
  --local-dir data/fineweb-ja
```

## 3. 预训练 Large 模型

编辑 [configs/pretrain.json](configs/pretrain.json) 后启动：

```bash
python pretrain.py
```

或者使用 8 张 GPU 进行分布式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=8 pretrain.py
```

训练完成后，模型和 tokenizer 默认保存在 `outputs/large/final`。

## 4. 初始化 Tiny 模型

如果需要训练tiny模型，我们推荐先使用large模型权重对tiny模型进行初始化，以达到更好的效果。

编辑 [`configs/distill.json`](configs/distill.json) 中的相关配置。`teacher` 默认指向上一步训练得到的 Large 模型，也可设为 `KdaiP/ModernBERT-CJE-large`，直接使用我们已经训练好的 Large 权重。

先在单个进程中执行激活校准与裁剪，以初始化tiny模型：

```bash
CUDA_VISIBLE_DEVICES=0 python prune.py
```

默认校准设置为 **1,000 个 batch**，每个 batch **4 篇文档**，每篇文档最多 **512 个 token**。

裁剪方法参考 [Minitron](https://arxiv.org/abs/2407.14679) 的激活重要性结构化裁剪思路。裁剪得到的 Tiny 初始模型默认保存在 `outputs/tiny-init`。

## 5. 蒸馏 Tiny 模型

完成裁剪初始化后，运行tiny模型的训练脚本：

```bash
python distill.py
```

或者使用 8 张 GPU 进行分布式蒸馏：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=8 distill.py
```

训练完成后，Tiny 编码器与 tokenizer 默认保存至 `outputs/tiny/final/`。辅助投影参数保存在训练 checkpoint 中，不需要随最终编码器一同导出。

## 6. Checkpoint 与断点续训

训练期间，每次保存会生成一个 `checkpoint-<step>/` 目录，其中包含模型参数、优化器状态等内容。默认仅保留**最近 3 个 checkpoint**。

从指定 checkpoint 恢复训练：

```bash
# 恢复 Large 预训练
python pretrain.py --resume outputs/large/checkpoint-4000

# 恢复 Tiny 蒸馏
python distill.py --resume outputs/tiny/checkpoint-10000
```

> **注意**：断点续训时，请保持原有配置、数据文件和 GPU 数量不变。

## 参考文献

1. Saurav Muralidharan et al. (2024). [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679).
2. Xiaoqi Jiao et al. (2020). [TinyBERT: Distilling BERT for Natural Language Understanding](https://aclanthology.org/2020.findings-emnlp.372/). *Findings of EMNLP*.
3. Wenhui Wang et al. (2021). [MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://aclanthology.org/2021.findings-acl.188/). *Findings of ACL-IJCNLP*.
4. Yiming Cui et al. [Pre-Training with Whole Word Masking for Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm).
