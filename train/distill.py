"""Projected hidden-state distillation and pre-RoPE QQ/KK/VV relation distillation."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from common import arguments, setup, train


class Teacher(nn.Module):
    def __init__(self, path, layer):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(path, attn_implementation="sdpa")
        self.encoder.layers[layer].attn.Wqkv.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.qkv = output

    def forward(self, **inputs):
        states = self.encoder(**inputs, output_hidden_states=True).hidden_states
        return states, self.qkv


def relation_loss(student, teacher, mask, heads, positions):
    # Each row is one document; neither padding nor other documents enter the KL.
    total = student.new_zeros((), dtype=torch.float32)
    count = 0
    for s, t, valid in zip(student, teacher, mask):
        indices = valid.nonzero().flatten()
        indices = indices[torch.linspace(0, len(indices) - 1, min(len(indices), positions),
                                         device=indices.device).round().long()]

        def relations(qkv):
            width = qkv.shape[-1] // 3 // heads
            vectors = qkv[indices].reshape(-1, 3, heads, width).permute(1, 2, 0, 3).float()
            return F.log_softmax(vectors @ vectors.transpose(-1, -2) / math.sqrt(width), dim=-1)

        total = total + F.kl_div(relations(s), relations(t), log_target=True, reduction="sum")
        count += len(indices) * heads * 3
    return total / count


class Distillation(nn.Module):
    def __init__(self, args, teacher_width):
        super().__init__()
        self.student = AutoModel.from_pretrained(args.student, attn_implementation="sdpa")
        self.pairs = [(0, 0)] + [(i + 1, j + 1) for i, j in enumerate(args.hidden_layers)]
        self.projections = nn.ModuleList([
            nn.Linear(self.student.config.hidden_size, teacher_width, bias=False) for _ in self.pairs
        ])
        self.student.layers[args.relation_student_layer].attn.Wqkv.register_forward_hook(self.capture)
        self.args = args

    def capture(self, module, inputs, output):
        self.qkv = output

    def forward(self, input_ids, attention_mask, target):
        teacher_states, teacher_qkv = target
        states = self.student(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True).hidden_states
        valid = attention_mask.bool()
        hidden = states[0].new_zeros((), dtype=torch.float32)
        for projection, (s, t) in zip(self.projections, self.pairs):
            predicted = projection(states[s][valid][::self.args.hidden_stride]).float()
            expected = teacher_states[t][valid][::self.args.hidden_stride].float()
            hidden = hidden + F.mse_loss(F.layer_norm(predicted, (predicted.shape[-1],)),
                                         F.layer_norm(expected, (expected.shape[-1],)))
        hidden = hidden / len(self.pairs)
        relation = relation_loss(self.qkv, teacher_qkv, valid,
                                 self.args.relation_heads, self.args.relation_positions)
        return {"loss": hidden + relation, "hidden": hidden, "relation": relation}


def main():
    args = arguments("configs/distill.json")
    accelerator, tokenizer = setup(args)
    teacher = Teacher(args.teacher, args.relation_teacher_layer)
    model = Distillation(args, teacher.encoder.config.hidden_size)
    train(model, args, accelerator, tokenizer, teacher)


if __name__ == "__main__":
    main()
