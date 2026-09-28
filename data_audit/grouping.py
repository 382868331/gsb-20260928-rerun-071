"""近重复样本分组：显式禁止相似关系无限传递。

分组规则（随 ``GroupingConfig.version`` 落库，可追溯）：

1. **完全相同**：规范化 SHA-256 相等即同组（相等天然可传递）。
2. **近重复边**：两条样本同时满足
   - n-gram Jaccard 相似度 >= ``jaccard_threshold``；
   - 较短/较长长度比 >= ``min_length_ratio``（防止“一小段公共片段”误判）。
3. **非传递分组（完全连接规则）**：样本只有在与组内**每一条**已有样本
   都存在近重复边时才能入组。链式局部相似 A~B~C 但 A≁C 时，A、C 不会被
   归为同一身份；B-C 这类跨组边只作为 ``local_similarity`` 证据保留。
4. 所有判定只产生证据与结论，不删除、不改动样本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .fingerprints import TextFingerprint


@dataclass(frozen=True)
class GroupingConfig:
    jaccard_threshold: float = 0.8
    min_length_ratio: float = 0.7

    @property
    def version(self) -> str:
        return (
            f"group:jaccard>={self.jaccard_threshold}:"
            f"lenratio>={self.min_length_ratio}:complete-linkage:v1"
        )


@dataclass
class SimilarityEdge:
    left_id: str
    right_id: str
    kind: str  # exact | near
    jaccard: float
    length_ratio: float
    longest_common_substring: str


@dataclass
class DuplicateGroup:
    group_id: str
    sample_ids: list[str] = field(default_factory=list)
    kind: str = "near"  # exact | near


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _longest_common_substring(left: str, right: str, max_chars: int = 80) -> str:
    match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match(
        0, len(left), 0, len(right)
    )
    snippet = left[match.a : match.a + match.size]
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    return snippet


def _candidate_pairs(fingerprints: list[TextFingerprint]) -> set[tuple[str, str]]:
    """倒排索引预过滤：至少共享一个 n-gram 的样本对才进入两两比较。"""
    posting: dict[str, list[int]] = {}
    for idx, fp in enumerate(fingerprints):
        for shingle in fp.shingles:
            posting.setdefault(shingle, []).append(idx)
    pairs: set[tuple[str, str]] = set()
    for indices in posting.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if a > b:
                    a, b = b, a
                pairs.add((fingerprints[a].sample_id, fingerprints[b].sample_id))
    return pairs


def build_groups(
    fingerprints: list[TextFingerprint],
    canonical_texts: dict[str, str],
    config: GroupingConfig | None = None,
) -> tuple[list[DuplicateGroup], list[SimilarityEdge], list[SimilarityEdge]]:
    """返回 (重复分组, 组内身份边, 跨组局部相似边)。"""
    config = config or GroupingConfig()
    fp_by_id = {fp.sample_id: fp for fp in fingerprints}
    ordered_ids = [fp.sample_id for fp in fingerprints]

    # 1) 完全相同：按规范化哈希分桶（相等关系，可安全传递）。
    exact_buckets: dict[str, list[str]] = {}
    for sample_id in ordered_ids:
        digest = fp_by_id[sample_id].canonical_sha256
        exact_buckets.setdefault(digest, []).append(sample_id)

    exact_groups: list[DuplicateGroup] = []
    grouped: dict[str, str] = {}
    for digest, members in exact_buckets.items():
        if len(members) <= 1:
            continue
        group_id = f"grp-exact-{digest[:12]}"
        exact_groups.append(DuplicateGroup(group_id, members, kind="exact"))
        for sample_id in members:
            grouped[sample_id] = group_id

    # 2) 近重复边（带长度比护栏）。
    near_edges: dict[tuple[str, str], SimilarityEdge] = {}
    for left_id, right_id in _candidate_pairs(fingerprints):
        left_fp = fp_by_id[left_id]
        right_fp = fp_by_id[right_id]
        if (
            left_fp.canonical_sha256 == right_fp.canonical_sha256
            or not left_fp.shingles
            or not right_fp.shingles
        ):
            continue
        score = jaccard(left_fp.shingles, right_fp.shingles)
        short, long_ = sorted((left_fp.char_length, right_fp.char_length))
        ratio = short / long_ if long_ else 0.0
        if score >= config.jaccard_threshold and ratio >= config.min_length_ratio:
            near_edges[(left_id, right_id)] = SimilarityEdge(
                left_id=left_id,
                right_id=right_id,
                kind="near",
                jaccard=round(score, 6),
                length_ratio=round(ratio, 6),
                longest_common_substring=_longest_common_substring(
                    canonical_texts[left_id], canonical_texts[right_id]
                ),
            )

    # 3) 完全连接贪心分组：入组必须与组内每个成员都有近重复边。
    #    不使用并查集/连通分量，避免 A~B~C 但 A≁C 时错误合并身份。
    near_groups: list[DuplicateGroup] = []
    membership: dict[str, int] = {}
    adjacency: dict[str, set[str]] = {sid: set() for sid in ordered_ids}
    for (a, b) in near_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for sample_id in ordered_ids:
        if sample_id in grouped:
            continue
        placed = False
        for group_idx, group in enumerate(near_groups):
            if all(other in adjacency[sample_id] for other in group.sample_ids):
                group.sample_ids.append(sample_id)
                membership[sample_id] = group_idx
                placed = True
                break
        if not placed:
            membership[sample_id] = len(near_groups)
            near_groups.append(
                DuplicateGroup(f"grp-near-{len(near_groups) + 1:04d}", [sample_id])
            )

    # 合并后仍只有 1 个成员的单例组不构成重复组。
    singleton = {
        idx
        for idx, group in enumerate(near_groups)
        if len(group.sample_ids) == 1
    }
    duplicate_groups = list(exact_groups)
    identity_edges: list[SimilarityEdge] = []
    local_edges: list[SimilarityEdge] = []

    # 完全相同组内的边
    for group in exact_groups:
        members = group.sample_ids
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                identity_edges.append(
                    SimilarityEdge(
                        a, b, "exact", 1.0, 1.0,
                        canonical_texts[a][:80] + ("…" if len(canonical_texts[a]) > 80 else ""),
                    )
                )

    for (a, b), edge in near_edges.items():
        ga = membership.get(a)
        gb = membership.get(b)
        if ga is not None and ga == gb and ga not in singleton:
            identity_edges.append(edge)
        else:
            edge.kind = "local"
            local_edges.append(edge)

    for idx, group in enumerate(near_groups):
        if idx not in singleton:
            duplicate_groups.append(group)

    identity_edges.sort(key=lambda e: (e.left_id, e.right_id))
    local_edges.sort(key=lambda e: (e.left_id, e.right_id))
    return duplicate_groups, identity_edges, local_edges
