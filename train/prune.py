"""Calibrate activations and copy selected channels, heads, and FFN neurons."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed

from data import Corpus


class Importance:
    def __init__(self, model):
        self.heads = model.config.num_attention_heads
        self.sums, self.handles = {}, []
        self.documents = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                self.handles.append(module.register_forward_hook(
                    lambda module, inputs, output, key="norm:" + name: self.add(key, output)))
        for i, layer in enumerate(model.layers):
            self.handles.append(layer.attn.Wo.register_forward_pre_hook(
                lambda module, inputs, key=f"head:{i}": self.add(key, inputs[0], heads=True)))
            self.handles.append(layer.mlp.Wo.register_forward_pre_hook(
                lambda module, inputs, key=f"mlp:{i}": self.add(key, inputs[0])))

    def add(self, key, values, heads=False):
        values = values.float()
        if heads:
            values = values.reshape(*values.shape[:2], self.heads, -1).norm(dim=-1)
        else:
            values = values.abs()
        mask = self.mask.unsqueeze(-1)
        means = (values * mask).sum(1) / mask.sum(1)
        score = means.double().square().sum(0).cpu()
        self.sums[key] = self.sums.get(key, 0) + score

    def result(self, depth):
        rms = {key: (value / self.documents).sqrt() for key, value in self.sums.items()}
        for handle in self.handles:
            handle.remove()
        return {
            "hidden": torch.stack([value for key, value in rms.items() if key.startswith("norm:")]).sum(0),
            "heads": torch.stack([rms[f"head:{i}"] for i in range(depth)]),
            "neurons": torch.stack([rms[f"mlp:{i}"] for i in range(depth)]),
        }


@torch.no_grad()
def initialize(teacher, config, scores, mapping):
    student = AutoModel.from_config(config, attn_implementation="sdpa")
    source, target = teacher.state_dict(), student.state_dict()
    tc, sc = teacher.config, student.config
    channels = scores["hidden"].topk(sc.hidden_size).indices.sort().values
    target["embeddings.tok_embeddings.weight"].copy_(source["embeddings.tok_embeddings.weight"][:, channels])
    for name in ("embeddings.norm.weight", "final_norm.weight"):
        target[name].copy_(source[name][channels])
    head_width = tc.hidden_size // tc.num_attention_heads
    report = {"teacher_layers": mapping, "residual_channels": channels.tolist(), "heads": [], "neurons": []}
    for i, j in enumerate(mapping):
        sp, tp = f"layers.{i}.", f"layers.{j}."
        heads = scores["heads"][j].topk(sc.num_attention_heads).indices.sort().values
        attention_channels = (heads[:, None] * head_width + torch.arange(head_width)).flatten()
        qkv = source[tp + "attn.Wqkv.weight"].reshape(3, tc.num_attention_heads, head_width, tc.hidden_size)
        target[sp + "attn.Wqkv.weight"].copy_(qkv[:, heads][:, :, :, channels].reshape(3 * sc.hidden_size, sc.hidden_size))
        target[sp + "attn.Wo.weight"].copy_(source[tp + "attn.Wo.weight"][channels][:, attention_channels])
        neurons = scores["neurons"][j].topk(sc.intermediate_size).indices.sort().values
        wi = source[tp + "mlp.Wi.weight"].reshape(2, tc.intermediate_size, tc.hidden_size)
        target[sp + "mlp.Wi.weight"].copy_(wi[:, neurons][:, :, channels].reshape(2 * sc.intermediate_size, sc.hidden_size))
        target[sp + "mlp.Wo.weight"].copy_(source[tp + "mlp.Wo.weight"][channels][:, neurons])
        for norm in ("attn_norm.weight", "mlp_norm.weight"):
            if sp + norm in target:
                target[sp + norm].copy_(source[tp + norm][channels])
        report["heads"].append(heads.tolist())
        report["neurons"].append(neurons.tolist())
    return student, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distill.json")
    args = json.loads(Path(parser.parse_args().config).read_text())
    set_seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"])
    tokenizer.padding_side = "right"
    teacher = AutoModel.from_pretrained(args["teacher"], attn_implementation="sdpa").to(device).eval()
    importance = Importance(teacher)
    calibration = args["calibration"]
    corpus = Corpus(args["data_config"], tokenizer, args["max_length"], calibration["data_seed"])

    def collate(examples):
        return tokenizer.pad(
            [{k: v for k, v in x.items() if k != "word_ids"} for x in examples], return_tensors="pt",
        )

    batches = iter(DataLoader(corpus, batch_size=calibration["batch_size"], collate_fn=collate))
    with torch.no_grad():
        for step in range(calibration["batches"]):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            importance.mask = batch["attention_mask"].bool()
            importance.documents += batch["input_ids"].shape[0]
            teacher(**batch)
            if (step + 1) % 100 == 0:
                print(f"Calibration {step + 1}/{calibration['batches']}", flush=True)
    scores = importance.result(teacher.config.num_hidden_layers)
    teacher.cpu()
    config = AutoConfig.from_pretrained(args["student_config"])
    student, report = initialize(teacher, config, scores, args["initialization_layers"])
    output = Path(args["student"])
    student.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "pruning.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved initialized student to {output}")


if __name__ == "__main__":
    main()
