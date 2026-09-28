"""指纹生成：精确哈希 + 字符 k-gram 片段集合。"""

from __future__ import annotations

import hashlib

from .models import Fingerprint, Sample
from .normalize import normalize_text

SHINGLE_SIZE = 2
SHINGLE_METHOD = f"char-{SHINGLE_SIZE}gram-jaccard"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def char_shingles(normalized: str, k: int = SHINGLE_SIZE) -> set[str]:
    """生成字符 k-gram 集合；短文本退化为整体一个片段。"""

    if not normalized:
        return set()
    if len(normalized) < k:
        return {normalized}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def make_fingerprint(sample: Sample, version_id: str) -> tuple[Fingerprint, set[str]]:
    """返回指纹对象和内存中的 shingle 集合（集合用于比对，列表用于落盘）。"""

    normalized = normalize_text(sample.text)
    shingles = char_shingles(normalized)
    fp = Fingerprint(
        sample_id=sample.sample_id,
        version_id=version_id,
        exact_hash=sample.content_hash,
        normalized_hash=sha256_hex(normalized),
        shingles=sorted(shingles),
        method=SHINGLE_METHOD,
    )
    return fp, shingles


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)
