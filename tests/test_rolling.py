"""第四轮：滚动验证划分（时间截点 + 后到标签）核对。

覆盖：迟到标签不进早期训练、验证保留当时真正未知的窗口、家族/来源/
最低保留数量约束、提案逐项连带影响与部分采纳（冲突不静默放宽）、
人工确认不同身份仍按判定处理且相似证据保留、标签更正/来源撤回/判定
变化后的轮次重审与带限制保留、重开后各轮采用范围与证据快照的对应。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_audit.ingest import fingerprint_samples, load_dataset
from data_audit.rules import run_near_duplicate_rule
from data_audit.storage import AuditStore

ROOT = Path(__file__).resolve().parents[1]
ROLLING_DATA = ROOT / "data" / "rolling_samples.jsonl"

ROUNDS = [
    {"round_id": "r1", "window_start": "2026-01-01T00:00:00Z",
     "cutoff": "2026-01-31T00:00:00Z"},
    {"round_id": "r2", "window_start": "2026-02-01T00:00:00Z",
     "cutoff": "2026-02-28T00:00:00Z"},
    {"round_id": "r3", "window_start": "2026-03-01T00:00:00Z",
     "cutoff": "2026-03-31T00:00:00Z"},
]


def _audit(store, audit_id, version_id="rv1"):
    samples = store.load_samples(version_id)
    fps = store.load_fingerprints(version_id)
    conclusion = run_near_duplicate_rule(
        samples,
        {fp.sample_id: set(fp.shingles) for fp in fps},
        audit_id=audit_id,
        version_id=version_id,
        decisions=store.load_active_decisions(version_id),
        normalized_hashes={fp.sample_id: fp.normalized_hash for fp in fps},
    )
    store.save_conclusion(conclusion)
    return conclusion


def _make_store(tmp_path):
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(ROLLING_DATA)
    fps, _ = fingerprint_samples(samples, "rv1")
    store.save_version("rv1", samples, fps)
    _audit(store, "ra1")
    return store


def _rounds_file(tmp_path):
    path = tmp_path / "rounds.json"
    path.write_text(json.dumps(ROUNDS), "utf-8")
    return str(path)


def _create_plan(store, tmp_path, plan_id="p1", audit_id="ra1"):
    from data_audit.cli import main

    main([
        "plan-create", "--store", str(store.root), "--plan-id", plan_id,
        "--version-id", "rv1", "--audit-id", audit_id,
        "--rounds", _rounds_file(tmp_path),
    ])
    return store.load_plan(plan_id)


def _violations(plan, vtype, round_id=None):
    return [
        v for v in plan["violations"]
        if v["type"] == vtype and (round_id is None or v["round_id"] == round_id)
    ]


def test_late_label_never_in_early_training(tmp_path):
    store = _make_store(tmp_path)
    plan = _create_plan(store, tmp_path)
    # b2 的标签 2026-04-02 才可见：r3 截点（03-31）前不可见 -> 排除，
    # 且任何轮次都不会把它放进 train。
    assert plan["excluded"]["r3"]["b2"] == "label_not_visible"
    for round_id in ("r1", "r2", "r3"):
        assert plan["assignments"][round_id].get("b2") != "train"
    # 未采集的样本同样排除，不进入任何一侧。
    assert plan["excluded"]["r1"]["a2"] == "not_collected_yet"


def test_validation_keeps_truly_unknown_window(tmp_path):
    store = _make_store(tmp_path)
    plan = _create_plan(store, tmp_path)
    val_of = {
        rid: sorted(s for s, sp in assign.items() if sp == "val")
        for rid, assign in plan["assignments"].items()
    }
    # 每轮验证集恰好是该轮窗口内采集、当时真正未知的样本。
    assert val_of["r1"] == ["a1", "a3", "b1", "u1", "x1"]
    assert val_of["r2"] == ["a2", "b2", "x2"]
    assert val_of["r3"] == ["u2"]


def test_family_source_and_min_count_violations_listed(tmp_path):
    store = _make_store(tmp_path)
    plan = _create_plan(store, tmp_path)
    # 家族 A（a1/a3 train + a2 val）与家族 B（b1 train + b2 val）在 r2 跨侧。
    family_splits = _violations(plan, "family_split", "r2")
    assert {tuple(v["members"]) for v in family_splits} >= {
        ("a1", "a2", "a3"), ("b1", "b2"),
    }
    # 来源隔离：S1/S2 在 r2 横跨 train/val。
    straddles = _violations(plan, "source_straddle", "r2")
    assert {v["source"] for v in straddles} >= {"S1", "S2"}
    # 每组最低保留数量：b 家族在 r2 两侧各只有 1 个。
    min_hits = _violations(plan, "min_per_group", "r2")
    assert any(v["count"] == 1 and v["required"] == 2 for v in min_hits)


def test_proposal_impact_and_partial_adoption(tmp_path):
    store = _make_store(tmp_path)
    _create_plan(store, tmp_path)
    from data_audit.cli import main

    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps({"round_id": "r2", "sample_id": "b2", "split": "train"})
        + "\n",
        "utf-8",
    )
    main([
        "plan-propose", "--store", str(store.root), "--plan-id", "p1",
        "--proposal-id", "prop1", "--input", str(items),
    ])
    proposal = store.load_proposal("prop1")
    impact = proposal["impacts"][0]
    # 连带影响：解决了 b 家族跨侧 / S2 来源横跨 / 最低保留数量，
    # 但引入了迟到标签进训练的冲突——明确列出，不静默放宽。
    assert any(v["type"] == "family_split" for v in impact["resolved"])
    assert any(v["type"] == "source_straddle" for v in impact["resolved"])
    introduced_types = {v["type"] for v in impact["introduced"]}
    assert introduced_types == {"late_label_in_train"}

    main([
        "plan-apply", "--store", str(store.root), "--plan-id", "p1",
        "--proposal-id", "prop1", "--new-plan-id", "p2", "--items", "0",
    ])
    p2 = store.load_plan("p2")
    assert p2["parent_plan_id"] == "p1"
    assert p2["assignments"]["r2"]["b2"] == "train"
    # 引入的冲突保留在违规列表中；未触及的 a 家族冲突也不会消失。
    assert _violations(p2, "late_label_in_train", "r2")
    a_splits = [
        v for v in _violations(p2, "family_split", "r2")
        if "a2" in v["members"]
    ]
    assert a_splits
    # 父方案原样保留。
    assert store.load_plan("p1")["assignments"]["r2"]["b2"] == "val"


def test_human_distinct_respected_and_evidence_kept(tmp_path):
    store = _make_store(tmp_path)
    # 人工确认 x1/x2 是不同身份后重新审查：分组排除该对，但相似证据保留。
    store.append_decision_event(
        "rv1",
        {
            "event": "add",
            "at": "2026-02-15T00:00:00Z",
            "decision": {
                "decision_id": "h1",
                "sample_a": "x1",
                "sample_b": "x2",
                "kind": "distinct",
                "note": "内容相同但确认为不同来源身份",
                "created_at": "2026-02-15T00:00:00Z",
            },
        },
    )
    conclusion = _audit(store, "ra2")
    assert not any(
        "x1" in group and "x2" in group for group in conclusion.groups
    )
    assert any(
        {edge["sample_a"], edge["sample_b"]} == {"x1", "x2"}
        for edge in conclusion.excluded_edges
    )
    plan = _create_plan(store, tmp_path, plan_id="px", audit_id="ra2")
    # 规划遵循人工判定：x1/x2 不再构成家族跨侧违规。
    assert not any(
        "x1" in v.get("members", []) or "x2" in v.get("members", [])
        for v in _violations(plan, "family_split")
    )


def test_review_lists_rounds_after_label_source_and_decision_changes(tmp_path):
    store = _make_store(tmp_path)
    _create_plan(store, tmp_path)
    from data_audit.cli import main

    main([
        "label-correct", "--store", str(store.root), "--version-id", "rv1",
        "--sample-id", "b2", "--label-available-at", "2026-03-10T00:00:00Z",
    ])
    main([
        "source-withdraw", "--store", str(store.root), "--version-id", "rv1",
        "--source", "S3",
    ])
    # 人工身份判定变化（新增 a1/a2 不同身份的判定）。
    main([
        "decide", "--store", str(store.root), "--version-id", "rv1",
        "--decision-id", "h9", "--sample-a", "a1", "--sample-b", "a2",
        "--kind", "distinct",
    ])
    main(["plan-review", "--store", str(store.root), "--plan-id", "p1"])
    plan = store.load_plan("p1")
    assert plan["status"] == "restricted"
    affected = plan["restrictions"][-1]["affected_rounds"]
    by_round = {entry["round_id"]: entry["reasons"] for entry in affected}
    # 标签更正：b2 在 r2 被采用；r3 可见性相对截点发生变化。
    assert any(r["type"] == "label_correction" for r in by_round["r2"])
    r3_label = [r for r in by_round["r3"] if r["type"] == "label_correction"]
    assert r3_label and r3_label[0]["visibility_changed"]
    # 来源撤回：S3 的 u1 在 r1/r2/r3 均被采用。
    assert any(r["type"] == "source_withdrawal" for r in by_round["r1"])
    # 判定变化：a1/a2 在 r2 同轮被采用。
    assert any(r["type"] == "decision_changed" for r in by_round["r2"])
    # 带限制保留：旧方案的各轮划分内容不变。
    assert plan["assignments"]["r2"]["b2"] == "val"


def test_reopen_recomputes_and_stays_consistent(tmp_path):
    store = _make_store(tmp_path)
    _create_plan(store, tmp_path)
    from data_audit.cli import main

    main([
        "label-correct", "--store", str(store.root), "--version-id", "rv1",
        "--sample-id", "b2", "--label-available-at", "2026-03-10T00:00:00Z",
    ])
    main([
        "source-withdraw", "--store", str(store.root), "--version-id", "rv1",
        "--source", "S3",
    ])
    main(["plan-review", "--store", str(store.root), "--plan-id", "p1"])
    main([
        "plan-reopen", "--store", str(store.root), "--plan-id", "p1",
        "--new-plan-id", "p2",
    ])
    p1 = store.load_plan("p1")
    p2 = store.load_plan("p2")
    # 旧方案带限制保留，新方案与其谱系对应。
    assert p1["status"] == "restricted"
    assert p2["parent_plan_id"] == "p1"
    assert p2["rounds"] == p1["rounds"]
    # 决定时可见的证据快照：更正后的标签可见时间。
    assert (
        p2["evidence_snapshot"]["label_visibility"]["b2"]
        == "2026-03-10T00:00:00Z"
    )
    assert p2["evidence_snapshot"]["withdrawn_sources"] == ["S3"]
    # 最终划分与可见性一致：b2 标签已在 r3 截点前可见 -> 进入 train；
    # S3 来源的样本全部因撤回被排除。
    assert p2["assignments"]["r3"]["b2"] == "train"
    for round_id in ("r1", "r2", "r3"):
        assert p2["excluded"][round_id]["u1"] == "source_withdrawn"
        assert "u1" not in p2["assignments"][round_id]
    # 重开后各轮 train 中不存在截点后才可见标签的样本。
    for spec in p2["rounds"]:
        for sample_id, split in p2["assignments"][spec["round_id"]].items():
            if split == "train":
                label_at = p2["evidence_snapshot"]["label_visibility"][sample_id]
                assert not label_at or label_at <= spec["cutoff"]


def test_cli_end_to_end_rolling_flow(tmp_path):
    store_dir = tmp_path / "store"
    rounds_path = tmp_path / "rounds.json"
    rounds_path.write_text(json.dumps(ROUNDS), "utf-8")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "data_audit.cli", *args],
            check=True, capture_output=True, text=True, cwd=ROOT,
        ).stdout

    run("ingest", "--store", str(store_dir), "--version-id", "rv1",
        "--input", str(ROLLING_DATA))
    run("audit", "--store", str(store_dir), "--version-id", "rv1",
        "--audit-id", "ra1", "--no-base")
    out = run("plan-create", "--store", str(store_dir), "--plan-id", "p1",
              "--version-id", "rv1", "--audit-id", "ra1",
              "--rounds", str(rounds_path))
    assert "待解决违规" in out
    out = run("label-correct", "--store", str(store_dir), "--version-id",
              "rv1", "--sample-id", "b2",
              "--label-available-at", "2026-03-10T00:00:00Z")
    assert "标签更正" in out
    out = run("plan-review", "--store", str(store_dir), "--plan-id", "p1")
    assert "restricted" in out and "r3" in out
    out = run("plan-reopen", "--store", str(store_dir), "--plan-id", "p1",
              "--new-plan-id", "p2")
    assert "重开方案 p2" in out
    out = run("plan-show", "--store", str(store_dir), "--plan-id", "p2")
    assert "b2" in out
