"""Shared arguments, optimizer, schedule, and training loop."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import Corpus, MaskingCollator


def arguments(default_config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume", help="Checkpoint directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    return argparse.Namespace(**config, resume=args.resume)


def setup(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    accelerator.init_trackers("tensorboard")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "right"
    return accelerator, tokenizer


def data_loader(args, tokenizer):
    return DataLoader(
        Corpus(args.data_config, tokenizer, args.max_length, args.data_seed),
        batch_size=args.batch_size,
        collate_fn=MaskingCollator(tokenizer, args.data_seed),
        num_workers=0,
    )


def learning_rate_factor(step, args):
    if step < args.warmup_steps:
        return step / max(1, args.warmup_steps)
    decay_start = args.max_steps - args.decay_steps
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / args.decay_steps)
    return args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def train(model, args, accelerator, tokenizer, teacher=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, args.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, args))
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader(args, tokenizer), scheduler,
    )
    if teacher is not None:
        teacher.to(accelerator.device).eval().requires_grad_(False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (output / "run.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    batches = iter(loader)
    step = 0
    if args.resume:
        step = json.loads((Path(args.resume) / "progress.json").read_text())["step"]
        # Replay the deterministic stream and its masking generator, then restore model RNG.
        for _ in range(step * args.gradient_accumulation_steps):
            next(batches)
        accelerator.load_state(args.resume)
    model.train()
    optimizer.zero_grad()
    metrics = {}

    def save(directory):
        accelerator.save_state(str(directory))
        if accelerator.is_main_process:
            raw = accelerator.unwrap_model(model)
            encoder = raw.student if teacher is not None else raw
            encoder.save_pretrained(directory / "model")
            tokenizer.save_pretrained(directory / "model")
            (directory / "progress.json").write_text(json.dumps({"step": step}) + "\n")
            checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            for old in checkpoints[:-args.keep_checkpoints]:
                shutil.rmtree(old)
        accelerator.wait_for_everyone()

    while step < args.max_steps:
        batch = next(batches)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                if teacher is None:
                    losses = {"loss": model(**batch).loss}
                else:
                    inputs = {k: v for k, v in batch.items() if k != "labels"}
                    with torch.no_grad():
                        target = teacher(**inputs)
                    losses = model(**inputs, target=target)
                loss = losses["loss"]
            accelerator.backward(loss)
            for name, value in losses.items():
                metrics[name] = metrics.get(name, 0) + value.detach() / args.gradient_accumulation_steps
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            step += 1
            if step == 1 or step % args.log_every == 0:
                values = {name: round(accelerator.reduce(value, reduction="mean").item(), 5)
                          for name, value in metrics.items()}
                accelerator.print(json.dumps({"step": step, "lr": scheduler.get_last_lr()[0], **values}))
                accelerator.log({"lr": scheduler.get_last_lr()[0], **values}, step=step)
            metrics = {}
            if step % args.save_every == 0:
                save(output / f"checkpoint-{step}")
    if step % args.save_every:
        save(output / f"checkpoint-{step}")
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        encoder = raw.student if teacher is not None else raw
        encoder.save_pretrained(output / "final")
        tokenizer.save_pretrained(output / "final")
    accelerator.end_training()
