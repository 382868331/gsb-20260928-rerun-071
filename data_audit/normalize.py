"""文本规范化：只影响指纹计算，不改动原始样本。"""

from __future__ import annotations

import unicodedata

# 常见中英文标点；仅去掉这些字符与空白，不做词干化等激进改写。
_PUNCTUATION = (
    "，。！？；：、“”‘’（）《》〈〉【】「」『』·…—–-,.!?;:\\\"'()[]{}<>~`@#$%^&*_+=|/\\"
)
_PUNCT_TABLE = {ord(ch): None for ch in _PUNCTUATION}


def normalize_text(text: str) -> str:
    """NFKC 兼容分解 + 小写 + 去空白与标点，便于跨表面形式比对。"""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.translate(_PUNCT_TABLE)
    return "".join(text.split())
