"""Train a randomly initialized ModernBERT with masked language modeling."""

from transformers import AutoConfig, AutoModelForMaskedLM

from common import arguments, setup, train


def main():
    args = arguments("configs/pretrain.json")
    accelerator, tokenizer = setup(args)
    config = AutoConfig.from_pretrained(args.model_config)
    model = AutoModelForMaskedLM.from_config(config, attn_implementation="sdpa")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    train(model, args, accelerator, tokenizer)


if __name__ == "__main__":
    main()
