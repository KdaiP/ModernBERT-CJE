"""Streaming corpus mixing and Chinese whole-word masking."""

import json
import re
from glob import glob
from pathlib import Path

import jieba
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


HAN = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003347f]+")


class Corpus(IterableDataset):
    def __init__(self, config, tokenizer, max_length, seed):
        self.sources = json.loads(Path(config).read_text(encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed

    def documents(self, source):
        files = sorted({p for pattern in source["files"] for p in glob(pattern)})
        dataset = load_dataset(
            "parquet", data_files=files, split="train", streaming=True,
            columns=[source["text_column"]],
        ).shuffle(seed=self.seed, buffer_size=1024)
        epoch = 0
        while True:
            dataset.set_epoch(epoch)
            for row in dataset:
                text = row[source["text_column"]]
                if text and text.strip():
                    yield text
            epoch += 1

    def encode(self, text, language, segmenter):
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        spans = []
        if language == "zh":
            for run in HAN.finditer(text[:encoded["offset_mapping"][-2][1]]):
                start = run.start()
                for word in segmenter.cut(run.group(), HMM=False):
                    spans.append((start, start + len(word)))
                    start += len(word)
        groups, span_index, previous, group = [], 0, None, -1
        for index, ((start, end), special) in enumerate(zip(
            encoded.pop("offset_mapping"), encoded.pop("special_tokens_mask")
        )):
            if special or encoded["input_ids"][index] in self.tokenizer.all_special_ids:
                groups.append(-1)
                previous = None
                continue
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            in_word = span_index < len(spans) and spans[span_index][0] <= start < end <= spans[span_index][1]
            key = ("word", span_index) if in_word else ("token", index)
            if key != previous:
                group += 1
            groups.append(group)
            previous = key
        encoded["word_ids"] = groups
        return encoded

    def __iter__(self):
        segmenter = jieba.Tokenizer()
        streams = {lang: self.documents(source) for lang, source in self.sources.items()}
        tokens = dict.fromkeys(streams, 0)
        while True:
            # Balance actual token counts, rather than document counts.
            language = min(streams, key=lambda lang: tokens[lang] / self.sources[lang]["weight"])
            example = self.encode(next(streams[language]), language, segmenter)
            if max(example["word_ids"]) < 0:
                continue
            tokens[language] += len(example["input_ids"])
            yield example


class MaskingCollator:
    def __init__(self, tokenizer, seed, mask_probability=0.15):
        self.tokenizer = tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.mask_probability = mask_probability

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            [{k: v for k, v in x.items() if k != "word_ids"} for x in examples],
            return_tensors="pt",
        )
        selected = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        for row, example in enumerate(examples):
            groups = torch.tensor(example["word_ids"])
            choices = torch.rand(int(groups.max()) + 1, generator=self.generator) < self.mask_probability
            # At least one target keeps the MLM mean defined for short documents.
            if not choices.any():
                choices[torch.randint(len(choices), (1,), generator=self.generator)] = True
            selected[row, :len(groups)] = choices[groups.clamp_min(0)] & groups.ge(0)
        labels = batch["input_ids"].clone()
        labels[~selected] = -100
        replacement = torch.rand(labels.shape, generator=self.generator)
        batch["input_ids"][selected & (replacement < 0.8)] = self.tokenizer.mask_token_id
        random_positions = selected & (replacement >= 0.8) & (replacement < 0.9)
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, generator=self.generator)
        batch["input_ids"][random_positions] = random_tokens[random_positions]
        batch["labels"] = labels
        return batch
