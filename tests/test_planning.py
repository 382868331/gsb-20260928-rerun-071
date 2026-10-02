"""第四轮：候选划分约束检查、迁移建议逐项采纳、冲突与撤回影响。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_audit.ingest import fingerprint_samples, load_dataset
from data_audit.models import Decision
from data_audit.planning import (
    apply_moves,
    bound_units,
    evaluate_constraints,
    find_conflicts,
    load_constraints,
    suggest_moves,
)
from data_audit.rules import run_near_duplicate_rule
from data_audit.storage import AuditStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "planning_samples.jsonl"
CONSTRAINTS = ROOT / "data" / "planning_constraints.json"


def _make_store(tmp_path):
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(DATA)
    fps, _ = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fps)
    return store


def _context(store, split_id=""):
    samples = store.load_samples("v1")
    fps = store.load_fingerprints("v1")
    overrides = (
        dict(store.load_split(split_id)["assignments"]) if split_id else {}
    )
    conclusion = run_near_duplicate_rule(
        samples,
        {fp.sample_id: set(fp.shingles) for fp in fps},
        audit_id="t",
        version_id="v1",
        decisions=store.load_active_decisions("v1"),
        split_id=split_id,
        split_overrides=overrides,
        normalized_hashes={fp.sample_id: fp.normalized_hash for fp in fps},
    )
    split_of = {s.sample_id: s.split for s in samples}
    split_of.update(overrides)
    return samples, split_of, conclusion


def _units(store, constraints, conclusion, samples):
    decisions = store.load_active_decisions("v1")
    distinct = {
        tuple(sorted((d.sample_a, d.sample_b)))
        for d in decisions
        if d.kind == "distinct"
    }
    conflicts = find_conflicts(constraints, distinct, len(samples))
    units = bound_units(
        sorted(s.sample_id for s in samples),
        conclusion.groups,
        constraints,
        conflicts,
    )
    return units, conflicts


def _decide(store, a, b, kind, decision_id="h1"):
    store.append_decision_event(
        "v1",
        {
            "event": "add",
            "at": "2026-10-02T00:00:00Z",
            "decision": Decision(
                decision_id=decision_id, sample_a=a, sample_b=b, kind=kind
            ).to_dict(),
        },
    )


def test_check_reports_violations_with_samples(tmp_path):
    store = _make_store(tmp_path)
    samples, split_of, conclusion = _context(store)
    constraints = load_constraints(CONSTRAINTS)
    units, conflicts = _units(store, constraints, conclusion, samples)
    assert conflicts == []
    violations = evaluate_constraints(samples, split_of, units, constraints)
    kinds = {v["kind"] for v in violations}
    families = {
        tuple(v["family"]) for v in violations if v["kind"] == "family_split"
    }
    assert ("p1", "p2") in families
    assert ("q1", "q2") in families
    source_hits = [v for v in violations if v["kind"] == "source_overlap"]
    assert source_hits
    involved = set()
    for hit in source_hits:
        for ids in hit["samples"].values():
            involved |= set(ids)
    assert "p1" in involved and "r2" in involved
    assert "min_count" not in kinds


def test_conflicts_reported_not_relaxed(tmp_path):
    store = _make_store(tmp_path)
    _decide(store, "p1", "p2", "distinct")
    samples, split_of, conclusion = _context(store)
    constraints = load_constraints(CONSTRAINTS)
    constraints["families"] = [["p1", "p2"]]
    units, conflicts = _units(store, constraints, conclusion, samples)
    assert any(
        c["kind"] == "family_vs_human_distinct" and c["pair"] == ["p1", "p2"]
        for c in conflicts
    )
    assert not any({"p1", "p2"} <= set(u["samples"]) for u in units)
    constraints2 = load_constraints(CONSTRAINTS)
    constraints2["min_counts"] = {"train": 5, "val": 5}
    conflicts2 = find_conflicts(constraints2, set(), len(samples))
    assert any(c["kind"] == "min_counts_infeasible" for c in conflicts2)


def test_suggest_moves_show_improves_and_breaks(tmp_path):
    store = _make_store(tmp_path)
    samples, split_of, conclusion = _context(store)
    constraints = load_constraints(CONSTRAINTS)
    units, _ = _units(store, constraints, conclusion, samples)
    moves = suggest_moves(
        samples, split_of, conclusion.edges, units, constraints
    )
    assert moves
    by_unit = {tuple(m["unit"]): m for m in moves}
    move = by_unit[("p1", "p2")]
    assert move["to"] == "val"
    assert ["p1", "p2"] in move["improves"]["leakages_resolved"]
    for m in moves:
        assert "violations_added" in m["breaks"]
        assert "leakages_added" in m["breaks"]
    assert any(
        v["kind"] == "source_overlap"
        for v in move["breaks"]["violations_added"]
    )


def test_partial_adoption_saved_and_reopenable(tmp_path):
    store = _make_store(tmp_path)
    samples, split_of, conclusion = _context(store)
    constraints = load_constraints(CONSTRAINTS)
    units, conflicts = _units(store, constraints, conclusion, samples)
    moves = suggest_moves(
        samples, split_of, conclusion.edges, units, constraints
    )
    adopt = [
        m["id"] for m in moves if m["unit"] == ["p1", "p2"] and m["to"] == "val"
    ]
    assignments = apply_moves(split_of, moves, adopt)
    assert assignments == {"p1": "val"}
    store.save_split("sp-plan", "v1", assignments)
    plan = {
        "plan_id": "plan1",
        "version_id": "v1",
        "base_split_id": "",
        "result_split_id": "sp-plan",
        "stage": "partial",
        "constraints": constraints,
        "assignments": assignments,
        "moves": [
            {**m, "status": "adopted" if m["id"] in adopt else "pending"}
            for m in moves
        ],
        "undecided_conflicts": conflicts,
        "decisions_snapshot": [],
        "created_at": "2026-10-02T00:00:00Z",
    }
    store.save_plan(plan)

    _, _, migrated = _context(store, split_id="sp-plan")
    leak_pairs = {
        tuple(sorted((e["sample_a"], e["sample_b"])))
        for e in migrated.leakages
    }
    assert ("p1", "p2") not in leak_pairs
    assert ("q1", "q2") in leak_pairs
    assert {
        tuple(sorted((e["sample_a"], e["sample_b"]))) for e in migrated.edges
    } == {
        tuple(sorted((e["sample_a"], e["sample_b"]))) for e in conclusion.edges
    }
    assert migrated.groups == conclusion.groups
    loaded = store.load_plan("plan1")
    assert any(m["status"] == "pending" for m in loaded["moves"])
    with pytest.raises(FileExistsError):
        store.save_plan(plan)


def test_withdraw_distinct_restates_impact(tmp_path):
    store = _make_store(tmp_path)
    _decide(store, "q1", "q2", "distinct")
    samples, split_of, conclusion = _context(store)
    constraints = load_constraints(CONSTRAINTS)
    units, conflicts = _units(store, constraints, conclusion, samples)
    assert not any(
        {"q1", "q2"} <= set(u["samples"]) and u["kind"] != "singleton"
        for u in units
    )
    snapshot = [d.to_dict() for d in store.load_active_decisions("v1")]
    store.save_plan(
        {
            "plan_id": "plan2",
            "version_id": "v1",
            "base_split_id": "",
            "result_split_id": "",
            "stage": "candidate",
            "constraints": constraints,
            "assignments": {},
            "moves": [],
            "undecided_conflicts": conflicts,
            "decisions_snapshot": snapshot,
            "created_at": "2026-10-02T00:00:00Z",
        }
    )
    store.append_decision_event(
        "v1",
        {
            "event": "withdraw",
            "at": "2026-10-02T01:00:00Z",
            "withdrawn_decision_id": "h1",
            "sample_a": "q1",
            "sample_b": "q2",
        },
    )
    plan = store.load_plan("plan2")
    before = {
        tuple(sorted((d["sample_a"], d["sample_b"])))
        for d in plan["decisions_snapshot"]
    }
    after = {
        tuple(sorted((d.sample_a, d.sample_b)))
        for d in store.load_active_decisions("v1")
    }
    assert ("q1", "q2") in before - after
    samples2, split_of2, conclusion2 = _context(store)
    units2, _ = _units(store, constraints, conclusion2, samples2)
    assert any(set(u["samples"]) == {"q1", "q2"} for u in units2)
    violations2 = evaluate_constraints(samples2, split_of2, units2, constraints)
    assert any(
        v["kind"] == "family_split" and v["family"] == ["q1", "q2"]
        for v in violations2
    )


def test_cli_plan_flow(tmp_path):
    store_dir = tmp_path / "store"

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-m", "data_audit.cli", *argv],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    run("ingest", "--store", str(store_dir), "--version-id", "v1",
        "--input", str(DATA))
    out = run("plan-check", "--store", str(store_dir), "--version-id", "v1",
              "--constraints", str(CONSTRAINTS))
    assert "family_split" in out and "source_overlap" in out
    out = run("plan-suggest", "--store", str(store_dir), "--version-id", "v1",
              "--constraints", str(CONSTRAINTS))
    assert "改善" in out and "破坏约束" in out
    out = run("plan-apply", "--store", str(store_dir), "--version-id", "v1",
              "--constraints", str(CONSTRAINTS), "--plan-id", "pl1",
              "--stage", "partial", "--adopt", "m1",
              "--new-split-id", "sp1")
    assert "pl1" in out
    out = run("audit", "--store", str(store_dir), "--version-id", "v1",
              "--audit-id", "a1", "--split-id", "sp1", "--no-base")
    assert "泄漏" in out
    out = run("plan-status", "--store", str(store_dir), "--plan-id", "pl1")
    assert "pl1" in out
