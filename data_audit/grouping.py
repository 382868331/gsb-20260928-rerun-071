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
   结论里，与最终分组（身份判定）分开列出，均可追溯；
4. 第二轮加入人工身份判定：人工确认的“不同身份”对即使相似也不参与成组；
   人工确认的“同身份”对强制成团。每次成组/并入/合并都记录成组依据
   （similarity / human_confirmed）。
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
    excluded_pairs: set[tuple[str, str]] | None = None,
    forced_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[list[str]], list[dict]]:
    """强边优先的完全连接凝聚聚类，返回 (成员 >=2 的组, 成组依据)。

    每个样本在任何时刻至多属于一个组；无法满足团条件的边被拒绝，但边本身
    仍由调用方保留作为证据。

    ``excluded_pairs``：人工确认的“不同身份”对，即使相似度达标也不成边、
    不参与合并（相似证据由调用方另行保留在 excluded_edges 中）。
    ``forced_pairs``：人工确认的“同身份”对，以最高优先级强制成团/合并，
    成组依据标记为 human_confirmed。
    """

    adjacency: dict[str, set[str]] = {sid: set() for sid in sample_ids}
    excluded = {tuple(sorted(pair)) for pair in (excluded_pairs or set())}
    forced = {tuple(sorted(pair)) for pair in (forced_pairs or set())}

    auto_edges: list[tuple[str, str, float]] = []
    for left_id, right_id, similarity in edges:
        if tuple(sorted((left_id, right_id))) in excluded:
            continue
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)
        auto_edges.append((left_id, right_id, similarity))

    groups: list[set[str]] = []
    owner: dict[str, int] = {}
    basis: list[dict] = []

    def open_group(left_id, right_id, source, sim=None):
        index = len(groups)
        groups.append({left_id, right_id})
        owner[left_id] = owner[right_id] = index
        record = {
            "pair": sorted((left_id, right_id)),
            "source": source,
            "group": index,
        }
        if sim is not None:
            record["similarity"] = sim
        basis.append(record)

    def attach(index, node, source):
        groups[index].add(node)
        owner[node] = index
        basis.append(
            {"sample_id": node, "source": source, "group": index}
        )

    def merge_groups(left_group, right_group, source):
        merged = groups[left_group] | groups[right_group]
        groups[left_group] = merged
        for node in merged:
            owner[node] = left_group
        groups[right_group] = set()
        basis.append({"merge_groups": sorted(merged), "source": source})

    def force_pair(left_id, right_id):
        left_group = owner.get(left_id)
        right_group = owner.get(right_id)
        if left_group is None and right_group is None:
            open_group(left_id, right_id, "human_confirmed")
        elif left_group is None:
            attach(right_group, left_id, "human_confirmed")
        elif right_group is None:
            attach(left_group, right_id, "human_confirmed")
        elif left_group != right_group:
            merge_groups(left_group, right_group, "human_confirmed")

    for left_id, right_id in sorted(forced):
        force_pair(left_id, right_id)

    for left_id, right_id, similarity in auto_edges:
        left_group = owner.get(left_id)
        right_group = owner.get(right_id)

        if left_group is None and right_group is None:
            open_group(left_id, right_id, "similarity", round(similarity, 6))
        elif left_group is None:
            if _clique_compatible(left_id, groups[right_group], adjacency):
                attach(right_group, left_id, "similarity")
        elif right_group is None:
            if _clique_compatible(right_id, groups[left_group], adjacency):
                attach(left_group, right_id, "similarity")
        elif left_group != right_group:
            if _groups_can_merge(
                groups[left_group], groups[right_group], adjacency
            ):
                merge_groups(left_group, right_group, "similarity")

    result = [sorted(group) for group in groups if len(group) >= 2]
    result.sort(key=lambda group: (group[0], len(group)))

    # 用最终稳定的组号重写 basis，便于报告引用。
    stable_index = {tuple(group): index for index, group in enumerate(result)}
    for record in basis:
        if "merge_groups" in record:
            members = record.pop("merge_groups")
            record["group"] = stable_index[tuple(members)]
            continue
        if "sample_id" in record:
            group = next(g for g in result if record["sample_id"] in g)
            record["group"] = stable_index[tuple(group)]
        elif "pair" in record:
            group = next(g for g in result if record["pair"][0] in g)
            record["group"] = stable_index[tuple(group)]
    return result, basis


def _groups_can_merge(group_a, group_b, adjacency):
    return all(
        _clique_compatible(node, group_b, adjacency) for node in group_a
    )
