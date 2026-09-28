"""近重复分组。

分组规则（刻意不做无限制的传递合并）：

1. 两两计算 char-2gram 的 Jaccard 相似度，只有相似度 ``>= threshold`` 时
   才建立一条近重复边；
2. 按相似度从强到弱做**完全连接（complete-link）凝聚聚类**，得到互斥分组：
   一个样本并入某个已有分组的条件，是它与该组中**每一个**成员都有近重复
   边，即组内始终构成一个相似团（clique）。两个分组合并同理，要求跨组的
   每一对成员都相似。因此 A~B、B~C 但 A 不~C 时，A 与 C 不会被归为同一
   身份，一串只局部相似的内容不会沿传递链塌缩成一个组；
3. 没能并入任何组的样本保持单例。未被采纳的边（局部相似证据）仍然保留在
   结论里，与最终分组（身份判定）分开列出，均可追溯。
"""

from __future__ import annotations

from itertools import combinations


def build_edges(
    sample_ids: list[str],
    shingle_sets: dict[str, set[str]],
    similarity_fn,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """返回所有达到阈值的无向边，按 (-sim, id1, id2) 排序，结果确定。"""

    edges: list[tuple[str, str, float]] = []
    for left_id, right_id in combinations(sample_ids, 2):
        similarity = similarity_fn(
            shingle_sets[left_id], shingle_sets[right_id]
        )
        if similarity + 1e-12 >= threshold:
            edges.append((left_id, right_id, similarity))
    edges.sort(key=lambda edge: (-edge[2], edge[0], edge[1]))
    return edges


def _clique_compatible(
    sample_id: str,
    group: set[str],
    adjacency: dict[str, set[str]],
) -> bool:
    """完全连接判据：sample_id 必须与组内每个成员都有近重复边。"""

    return group <= adjacency[sample_id]


def complete_link_groups(
    sample_ids: list[str],
    edges: list[tuple[str, str, float]],
) -> list[list[str]]:
    """强边优先的完全连接凝聚聚类，输出成员 >=2 的组。

    每个样本在任何时刻至多属于一个组；无法满足团条件的边被拒绝，但边本身
    仍由调用方保留作为证据。
    """

    adjacency: dict[str, set[str]] = {sid: set() for sid in sample_ids}
    for left_id, right_id, _ in edges:
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)

    groups: list[set[str]] = []
    owner: dict[str, int] = {}

    for left_id, right_id, _ in edges:
        left_group = owner.get(left_id)
        right_group = owner.get(right_id)

        if left_group is None and right_group is None:
            index = len(groups)
            groups.append({left_id, right_id})
            owner[left_id] = owner[right_id] = index
        elif left_group is None:
            if _clique_compatible(left_id, groups[right_group], adjacency):
                groups[right_group].add(left_id)
                owner[left_id] = right_group
        elif right_group is None:
            if _clique_compatible(right_id, groups[left_group], adjacency):
                groups[left_group].add(right_id)
                owner[right_id] = left_group
        elif left_group != right_group:
            if _groups_can_merge(
                groups[left_group], groups[right_group], adjacency
            ):
                merged = groups[left_group] | groups[right_group]
                keep, drop = left_group, right_group
                groups[keep] = merged
                for node in merged:
                    owner[node] = keep
                groups[drop] = set()

    result = [sorted(group) for group in groups if len(group) >= 2]
    result.sort(key=lambda group: (group[0], len(group)))
    return result


def _groups_can_merge(
    group_a: set[str],
    group_b: set[str],
    adjacency: dict[str, set[str]],
) -> bool:
    return all(
        _clique_compatible(node, group_b, adjacency) for node in group_a
    )
