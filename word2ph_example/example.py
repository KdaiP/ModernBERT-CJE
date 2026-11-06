"""将 ModernBERT-CJE 的中、日、英 token 特征展开到音素长度。"""

import argparse
from collections import Counter

import torch
from transformers import AutoModel, AutoTokenizer

from bv2_text import chinese, english, japanese

FRONTENDS = {"ZH": chinese, "JP": japanese, "EN": english}
EXAMPLES = {
    "ZH": "你好，世界！",
    "JP": "こんにちは、世界！",
    "EN": "Hello, world!",
}


def prepare_inputs(text, language, tokenizer):
    """音素先对齐到字符或单词，再分配到实际送入模型的 CJE token。"""
    frontend = FRONTENDS[language]
    norm_text = frontend.text_normalize(text)
    phones, tones, unit2ph = frontend.g2p(norm_text)

    if language == "ZH":
        units = list(norm_text)
    elif language == "JP":
        # OpenJTalk 可能改写字符；与上游 japanese_bert.py 使用相同的文本。
        units = list("".join(japanese.text2sep_kata(norm_text)[0]))
    else:
        units = english.sep_text(norm_text)

    # 分词和模型输入来自同一次调用，word_ids 给出每个 token 属于哪个单位。
    inputs = tokenizer(units, is_split_into_words=True, return_tensors="pt")
    token_counts = Counter(inputs.word_ids())
    word2ph = [1]  # [CLS] 对应句首的 _。
    for unit_id, phone_count in enumerate(unit2ph[1:-1]):
        token_count = token_counts[unit_id]
        quotient, remainder = divmod(phone_count, token_count)
        # 与上游 distribute_phone 相同：均匀分配，余数从左到右分配。
        word2ph.extend(quotient + (i < remainder) for i in range(token_count))
    word2ph.append(1)  # [SEP] 对应句尾的 _。
    return norm_text, phones, tones, word2ph, inputs


@torch.inference_mode()
def extract_features(model, inputs, word2ph):
    outputs = model(
        **inputs.to(model.device), output_hidden_states=True, return_dict=True
    )
    token_features = outputs.hidden_states[-3][0].cpu()
    # [token 数, hidden_size] -> [音素数, hidden_size] -> [hidden_size, 音素数]
    repeats = torch.tensor(word2ph)
    return token_features.repeat_interleave(repeats, dim=0).T.contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="KdaiP/ModernBERT-CJE-tiny")
    parser.add_argument("--language", choices=FRONTENDS, default="ZH")
    parser.add_argument("--text")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--add-blank", action="store_true")
    parser.add_argument("--output", default="features.pt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    text = args.text if args.text is not None else EXAMPLES[args.language]
    norm_text, phones, tones, word2ph, inputs = prepare_inputs(
        text, args.language, tokenizer
    )

    if args.add_blank:
        # 与 Bert-VITS2 的 intersperse(..., 0) 和 word2ph 更新方式一致。
        phones = ["_"] + [item for phone in phones for item in (phone, "_")]
        tones = [0] + [item for tone in tones for item in (tone, 0)]
        word2ph = [count * 2 for count in word2ph]
        word2ph[0] += 1

    features = extract_features(model, inputs, word2ph)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    torch.save(
        {
            "text": norm_text,
            "language": args.language,
            "tokens": tokens,
            "phones": phones,
            "tones": tones,
            "word2ph": word2ph,
            "features": features,
        },
        args.output,
    )
    print("规范化文本：", norm_text)
    print("Tokens：", tokens)
    print("音素：", phones)
    print("word2ph：", word2ph)
    print("音素数：", len(phones))
    print("特征形状：", tuple(features.shape))
    print("保存到：", args.output)


if __name__ == "__main__":
    main()
