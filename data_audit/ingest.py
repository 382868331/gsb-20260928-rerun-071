"""从 JSONL 清单导入一个数据版本。

每行格式：{"sample_id": ..., "split": "train|val|test", "text": ...}；
滚动验证还可选带 collected_at（采集时间）、label_available_at
（标签实际可见时间）与 source（来源），缺省按始终可见、无来源处理。
"""

from __future__ import annotations

import json
from pathlib import Path

from .fingerprint import make_fingerprint, sha256_hex
from .models import Sample


def load_dataset(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"重复的 sample_id: {sample_id} (第 {line_number} 行)")
            seen.add(sample_id)
            text = str(row["text"])
            raw_sources = row.get("sources")
            if raw_sources is None:
                raw_sources = [row["source"]] if row.get("source") else []
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            sources = [
                str(item).strip()
                for item in raw_sources
                if str(item).strip()
            ]
            samples.append(
                Sample(
                    sample_id=sample_id,
                    split=str(row["split"]),
                    text=text,
                    content_hash=sha256_hex(text),
                    collected_at=str(row.get("collected_at", "")),
                    label_available_at=str(row.get("label_available_at", "")),
                    source=str(row.get("source", "")),
                    sources=sources,
                )
            )
    return samples


def fingerprint_samples(
    samples: list[Sample], version_id: str
) -> tuple[list, dict[str, set[str]]]:
    fingerprints = []
    shingle_sets: dict[str, set[str]] = {}
    for sample in samples:
        fingerprint, shingles = make_fingerprint(sample, version_id)
        fingerprints.append(fingerprint)
        shingle_sets[sample.sample_id] = shingles
    return fingerprints, shingle_sets
