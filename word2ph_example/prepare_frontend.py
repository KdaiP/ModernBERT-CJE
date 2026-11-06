"""下载 Bert-VITS2 文本前端，移除对原有 BERT tokenizer 的依赖。"""

from pathlib import Path
from urllib.request import urlopen

REVISION = "3e710e996f03d52476490c435899e76d688dcbf1"
BASE_URL = f"https://raw.githubusercontent.com/fishaudio/Bert-VITS2/{REVISION}"
DESTINATION = Path("bv2_text")
FILES = [
    "chinese.py",
    "english.py",
    "japanese.py",
    "symbols.py",
    "tone_sandhi.py",
    "opencpop-strict.txt",
    "cmudict.rep",
]


def main():
    DESTINATION.mkdir(exist_ok=True)
    for name in FILES:
        source = urlopen(f"{BASE_URL}/text/{name}").read().decode("utf-8")
        if name.endswith(".py"):
            source = source.replace("from text import", "from bv2_text import")
            source = source.replace("from text.", "from bv2_text.")

        if name == "english.py":
            source = source.replace(
                "tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)", ""
            )
            # 先按完整单词生成音素；CJE 子词的 word2ph 在 example.py 中计算。
            source = source.replace(
                "words = text_to_words(text)",
                "words = [[word] for word in sep_text(text)]",
            )

        if name == "japanese.py":
            source = source.replace(
                'tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")',
                "",
            )
            # 保留上游按词分配音素的逻辑，将分配单位明确为字符。
            source = source.replace("tokenizer.tokenize(i)", "list(i)")

        (DESTINATION / name).write_text(source, encoding="utf-8")

    # 上游 __init__.py 会下载三套 BERT；这里只需要音素表。
    (DESTINATION / "__init__.py").write_text(
        "from .symbols import *\n", encoding="utf-8"
    )
    (DESTINATION / "LICENSE").write_bytes(urlopen(f"{BASE_URL}/LICENSE").read())
    print(f"已准备 Bert-VITS2 文本前端：{DESTINATION}/（{REVISION}）")


if __name__ == "__main__":
    main()
