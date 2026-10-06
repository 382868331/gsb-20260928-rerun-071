"""有限标注预算下的逐来源配额补齐。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Sample
from .rolling import overrides_as_of


def sample_sources(sample: Sample) -> list[str]:
    sources = list(dict.fromkeys(sample.sources))
    if not sources and sample.source:
        sources = [sample.source]
    return sources


def state_as_of(events: list[dict[str, Any]], cutoff: str) -> tuple[dict[str, str], set[str]]:
    return overrides_as_of(events, cutoff)


def _label_time(sample: Sample, labels: dict[str, str]) -> str:
    return labels.get(sample.sample_id) or sample.label_available_at or sample.collected_at


def identity_units(samples: list[Sample], families: list[list[str]]) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {sample.sample_id: [sample.sample_id] for sample in samples}
    for family in families:
        members = [sid for sid in family if sid in units]
        for sid in members:
            units[sid] = members
    return units


def basis_signature(
    samples: list[Sample],
    families: list[list[str]],
    events: list[dict],
    cutoff: str,
    spent_ids: set[str],
) -> str:
    labels, withdrawn = state_as_of(events, cutoff)
    payload = {
        "cutoff": cutoff,
        "spent": sorted(spent_ids),
        "families": sorted(sorted(family) for family in families),
        "samples": [
            {
                "id": s.sample_id,
                "sources": sample_sources(s),
                "collected_at": s.collected_at,
                "label_at": _label_time(s, labels),
                "withdrawn": sorted(set(sample_sources(s)) & withdrawn),
            }
            for s in sorted(samples, key=lambda item: item.sample_id)
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_inventory(
    samples: list[Sample],
    families: list[list[str]],
    events: list[dict],
    cutoff: str,
    spent_ids: set[str],
) -> tuple[dict[str, int], dict[str, list[str]], dict[str, list[str]], list[dict]]:
    labels, withdrawn = state_as_of(events, cutoff)
    by_id = {sample.sample_id: sample for sample in samples}
    units = identity_units(samples, families)
    confirmed: dict[str, int] = {}
    pending: dict[str, list[str]] = {}
    unavailable: dict[str, list[str]] = {}
    seen_units: set[str] = set()
    candidates: list[dict] = []

    for sample in sorted(samples, key=lambda item: item.sample_id):
        member_ids = sorted(units[sample.sample_id])
        unit_key = "|".join(member_ids)
        active_sources: list[str] = []
        aliases = []
        unit_labeled = True
        unit_collected = True
        unit_spent = False

        for member_id in member_ids:
            member = by_id[member_id]
            aliases.append(member_id)
            if member.collected_at and member.collected_at > cutoff:
                unit_collected = False
            if _label_time(member, labels) > cutoff:
                unit_labeled = False
            if member_id in spent_ids:
                unit_spent = True
            for source in sample_sources(member):
                if source in withdrawn:
                    unavailable.setdefault(source, []).append(f"{member_id}:来源撤销")
                elif member.collected_at and member.collected_at > cutoff:
                    unavailable.setdefault(source, []).append(f"{member_id}:截点后采集")
                else:
                    active_sources.append(source)

        active_sources = list(dict.fromkeys(active_sources))
        if unit_key in seen_units or not active_sources:
            continue
        seen_units.add(unit_key)

        if not unit_collected:
            continue
        if unit_labeled or unit_spent:
            for source in active_sources:
                confirmed[source] = confirmed.get(source, 0) + 1
            continue

        representative = min(member_ids)
        candidate = {
            "sample_id": representative,
            "identity_aliases": aliases,
            "sources": active_sources,
            "label_available_at": max(_label_time(by_id[sid], labels) for sid in member_ids),
        }
        candidates.append(candidate)
        for source in active_sources:
            pending.setdefault(source, []).append(representative)

    return confirmed, pending, unavailable, candidates


def _remaining_after(candidate: dict, remaining: dict[str, int]) -> dict[str, int]:
    after = dict(remaining)
    for source in candidate["sources"]:
        if after.get(source, 0) > 0:
            after[source] -= 1
    return after


def _cover(candidate: dict, remaining: dict[str, int]) -> int:
    return sum(1 for source in candidate["sources"] if remaining.get(source, 0) > 0)


def _minimum(candidates: list[dict], gaps: dict[str, int], budget: int) -> list[dict]:
    best: list[dict] | None = None
    memo: dict[tuple, int] = {}

    def search(index: int, remaining: tuple[int, ...], chosen: list[dict]) -> None:
        nonlocal best
        if not any(remaining):
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if index >= len(candidates) or len(chosen) >= budget:
            return
        if best is not None and len(chosen) >= len(best):
            return
        key = (index, remaining)
        if key in memo and memo[key] <= len(chosen):
            return
        memo[key] = len(chosen)

        candidate = candidates[index]
        covered = _cover(candidate, dict(zip(gaps.keys(), remaining)))
        if covered:
            next_remaining = tuple(_remaining_after(candidate, dict(zip(gaps.keys(), remaining))).values())
            search(index + 1, next_remaining, chosen + [candidate])
        search(index + 1, remaining, chosen)

    ordered = sorted(candidates, key=lambda c: c["sample_id"])
    search(0, tuple(gaps.values()), [])
    if best is not None:
        return best

    chosen = []
    remaining = dict(gaps)
    pool = list(ordered)
    while any(remaining.values()) and len(chosen) < budget and pool:
        candidate = max(pool, key=lambda item: (_cover(item, remaining), -len(item["sources"]), item["sample_id"]))
        if _cover(candidate, remaining) == 0:
            break
        chosen.append(candidate)
        remaining = _remaining_after(candidate, remaining)
        pool.remove(candidate)
    return chosen


def _shortage_first(
    candidates: list[dict], gaps: dict[str, int], needs: dict[str, int], budget: int
) -> list[dict]:
    chosen = []
    remaining = dict(gaps)
    pool = list(candidates)
    while any(remaining.values()) and len(chosen) < budget:
        target = max(
            remaining,
            key=lambda source: (remaining[source] / max(needs[source], 1), remaining[source], tuple(-ord(ch) for ch in source)),
        )
        usable = [item for item in pool if target in item["sources"] and _cover(item, remaining)]
        if not usable:
            break
        candidate = max(usable, key=lambda item: (_cover(item, remaining), -len(item["sources"]), item["sample_id"]))
        chosen.append(candidate)
        pool.remove(candidate)
        remaining = _remaining_after(candidate, remaining)
    return chosen


def _strategy_result(
    strategy: str,
    chosen: list[dict],
    gaps: dict[str, int],
    budget: int,
    pending: dict[str, list[str]],
    unavailable: dict[str, list[str]],
) -> dict:
    remaining = dict(gaps)
    for candidate in chosen:
        remaining = _remaining_after(candidate, remaining)
    reasons = {}
    for source, gap in remaining.items():
        if gap <= 0:
            continue
        details = []
        if len(chosen) >= budget:
            details.append("标注预算不足")
        if pending.get(source):
            details.append("仍有待标签候选 " + ", ".join(pending[source]))
        if unavailable.get(source):
            details.extend(unavailable[source])
        if not details:
            details.append("该来源样本总量不足")
        reasons[source] = "；".join(details)
    return {
        "strategy": strategy,
        "sample_ids": [item["sample_id"] for item in chosen],
        "candidates": chosen,
        "new_label_cost": len(chosen),
        "remaining_gaps": {source: gap for source, gap in remaining.items() if gap > 0},
        "feasible": not any(gap > 0 for gap in remaining.values()),
        "reasons": reasons,
    }


def plan_source_quota(
    *,
    samples: list[Sample],
    families: list[list[str]],
    events: list[dict],
    cutoff: str,
    needs: dict[str, int],
    budget: int,
    spent_ids: set[str] | None = None,
) -> dict[str, Any]:
    spent_ids = spent_ids or set()
    confirmed, pending, unavailable, candidates = _build_inventory(
        samples, families, events, cutoff, spent_ids
    )
    gaps = {source: max(0, need - confirmed.get(source, 0)) for source, need in sorted(needs.items())}
    minimum = _minimum(candidates, gaps, budget)
    shortage = _shortage_first(candidates, gaps, needs, budget)
    return {
        "cutoff": cutoff,
        "budget": budget,
        "basis_signature": basis_signature(samples, families, events, cutoff, spent_ids),
        "source_status": {
            source: {
                "need": needs[source],
                "confirmed": confirmed.get(source, 0),
                "gap": gaps[source],
                "pending_candidates": pending.get(source, []),
                "unavailable": unavailable.get(source, []),
            }
            for source in sorted(needs)
        },
        "strategies": {
            "minimum_new_labels": _strategy_result(
                "minimum_new_labels", minimum, gaps, budget, pending, unavailable
            ),
            "most_shortage_first": _strategy_result(
                "most_shortage_first", shortage, gaps, budget, pending, unavailable
            ),
        },
    }
