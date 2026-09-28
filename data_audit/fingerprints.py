"""样本规范化与指纹计算。

指纹分三层，全部随 ``fingerprint_version`` 落库：
1. 规范化文本的 SHA-256：判定完全相同；
2. 词元集合（拉丁词 + 汉字 unigram/bigram + 标点保留 token）：
   对中文局部改写、调序更稳健，用于精确 Jaccard 相似度；
3. MinHash 签名：第一轮已留存，供更大规模数据上的近似检索复用。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import re as _re

_WHITESPACE_RE = _re.compile(r"\s+")
_LATIN_WORD_RE = _re.compile(r"[a-z0-9]+")
_CJK_RE = _re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class FingerprintConfig:
    ngram_size: int = 4
    num_perm: int = 128
    lowercase: bool = True
    nfkc: bool = True

    @property
    def version(self) -> str:
        return (
            f"fp:char{self.ngram_size}gram:minhash{self.num_perm}:"
            f"lc={int(self.lowercase)}:nfkc={int(self.nfkc)}:v1"
        )


def canonicalize(text: str, config: FingerprintConfig) -> str:
    text = unicodedata.normalize("NFKC", text) if config.nfkc else text
    text = text.lower() if config.lowercase else text
    return _WHITESPACE_RE.sub(" ", text).strip()


def _shingles(text: str, n: int) -> set[str]:
    """混合词元集合，同时适配中文与英文：

    - 英文/数字：完整单词；
    - 连续汉字：unigram 与 bigram（n>=2 时），捕捉字序同时容忍局部改写；
    - 标点：非空白单字符，保留句法语义提示。
    """
    tokens: set[str] = set()
    tokens.update(f"w:{m.group()}" for m in _LATIN_WORD_RE.finditer(text))
    cjk_spans = [
        text[m.start() : m.end()]
        for m in _re.compile(r"[\u4e00-\u9fff]+").finditer(text)
    ]
    for span in cjk_spans:
        tokens.update(f"c:{ch}" for ch in span)
        if n >= 2:
            tokens.update(f"c2:{span[i:i+2]}" for i in range(len(span) - 1))
    for ch in text:
        if not ch.isspace() and not _CJK_RE.match(ch) and not ch.isalnum():
            tokens.add(f"p:{ch}")
    return tokens


# 固定参数的 MinHash：跨运行、跨版本可复现、可比较。
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _permutation_params(num_perm: int, seed: int = 0x5EED) -> list[tuple[int, int]]:
    params: list[tuple[int, int]] = []
    counter = 0
    while len(params) < num_perm:
        raw = hashlib.blake2b(
            f"minhash-{seed}-{counter}".encode(), digest_size=16
        ).digest()
        a = int.from_bytes(raw[:8], "little") % _MERSENNE_PRIME
        b = int.from_bytes(raw[8:], "little") % _MERSENNE_PRIME
        if a != 0:
            params.append((a, b))
        counter += 1
    return params


@dataclass
class TextFingerprint:
    sample_id: str
    canonical_sha256: str
    char_length: int
    shingles: set[str]
    minhash: list[int]


class TextFingerprinter:
    def __init__(self, config: FingerprintConfig | None = None) -> None:
        self.config = config or FingerprintConfig()
        self._params = _permutation_params(self.config.num_perm)

    def fingerprint(self, sample_id: str, raw_text: str) -> TextFingerprint:
        canonical = canonicalize(raw_text, self.config)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        shingle_set = _shingles(canonical, self.config.ngram_size)
        signature = [_MAX_HASH] * self.config.num_perm
        for shingle in shingle_set:
            token = int.from_bytes(
                hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            for idx, (a, b) in enumerate(self._params):
                value = ((a * token + b) % _MERSENNE_PRIME) % (_MAX_HASH + 1)
                if value < signature[idx]:
                    signature[idx] = value
        return TextFingerprint(
            sample_id=sample_id,
            canonical_sha256=digest,
            char_length=len(canonical),
            shingles=shingle_set,
            minhash=signature,
        )

    @staticmethod
    def minhash_jaccard(left: list[int], right: list[int]) -> float:
        if len(left) != len(right):
            raise ValueError("MinHash 签名长度不一致，无法比较")
        hits = sum(a == b for a, b in zip(left, right))
        return hits / len(left) if left else 0.0
