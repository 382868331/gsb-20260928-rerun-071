# 训练数据审查系统

四轮实现：

- **第一轮**：近重复样本分组与训练/验证（测试）划分泄漏检查；
- **第二轮**：增量加入样本后的近重复组修订、人工确认同/异身份（可撤回）、
  修订差异报告。系统只标记并给出人工复核建议，**不会自动删除或修改任何样本**。
- **第三轮**：同一数据版本上的多个**划分版本**（训练/测试安排）：相似关系与
  人工身份判定跨划分版本保留，泄漏结论按各自划分版本重算、不跨版本沿用；
  按样本家族查看迁移影响（区分完全复制 / 局部相似 / 人工确认不同身份）；
  支持跨版本复制式删除样本。撤回判定、删除样本、恢复旧划分后的增量结果
  与该版本完整重算一致。
- **第四轮**：按多个时间截点的**滚动验证划分方案**。样本带采集时间与标签
  实际可见时间：某轮训练只能用截点前标签已可见的样本，迟到标签绝不进入
  早期训练；各轮验证集保留当时真正未知的窗口范围。规划同时考虑样本家族
  （近重复组）、来源隔离与每组最低保留数量；调整以逐项提案进行，每个条目
  列出跨全部轮次的连带影响（解决/引入的冲突），用户逐项采纳，冲突不会
  通过静默放宽条件消失。标签更正、来源撤回、人工判定变化会列出需要重新
  审查的轮次，旧方案带限制保留，重开生成新方案版本并记录决定时可见的
  证据快照。

## 设计原则

- **三类数据分开保存，均可追溯原样本**：数据版本（原始样本）、样本指纹、
  审查结论分别落盘，通过 `version_id` / `sample_id` 关联。
- **版本不可变，增量靠跨版本复制**：新版本完整复制父版本样本后追加新样本，
  父版本原样保留；不能通过删掉/改写原始样本让问题消失。
- **相似关系不无限制传递**：局部相似的边不会沿传递链把一串内容塌缩成一个
  身份；新样本即使同时连接两个旧组（桥接），旧组也只在跨组成员构成团时
  才合并，未采纳的桥接边作为“未合并局部相似证据”保留。
- **证据与判定分开**：近重复边（含局部公共段落）是“局部相似证据”，分组是
  “身份判定”，泄漏边是“跨划分证据”；每次成组都记录依据
  （`similarity` / `human_confirmed`）。
- **人工判定只增不改、可撤回**：判定是 append-only 事件日志（add/withdraw）；
  每次审查结论内嵌当时的判定快照，撤回并重算后**旧报告仍可追溯、可复现**。
  审查结论本身也不允许覆盖，修订必须使用新的 `audit_id`。
- **泄漏结论对应各自划分版本**：划分版本（`splits/<split_id>.json`）只重新
  安排样本的 train/val/test 归属，不可覆盖；相似边、分组与人工判定与划分
  无关、继续保留，泄漏证据在每个划分版本下独立重算。自动基线按
  （数据版本, 划分版本）匹配，不会把另一种划分的结论当作基线沿用。
  恢复旧划分 = 再次引用旧的 `split_id` 重算。
- **删除也是跨版本复制**：`remove-samples` 复制父版本并剔除指定样本生成
  新版本，父版本原样保留；两端样本都存活的人工判定事件复制到新版本日志，
  引用被删样本的判定随样本一起不再生效。

## 划分迁移与家族视图

`audit --split-id` 按指定划分版本重算泄漏；与基线审查（可为另一种划分）
对比时，`changes.families` 给出**按样本家族（近重复组）的迁移影响**：

- 每个家族列出触及它的全部相似证据，分类为 `exact_copy`（完全复制，
  规范化哈希一致）/ `partial_similarity`（局部相似）/
  `human_distinct`（人工确认不同身份）/ `human_confirmed`（人工确认同身份）；
- 每条证据标注两个划分版本下的划分归属（`split_before` / `split_after`）
  与泄漏结论变化（`leakage_before` / `leakage_after`）；
- 跨家族的局部相似边与人工排除边也列入触及家族的证据
 （`within_family=false`），桥接样本仍只在跨组构成团时才合并。

## 滚动验证划分方案（第四轮）

样本可带 collected_at（采集时间）、label_available_at（标签实际可见时间）
与 source（来源），均为可选、缺省无限制。方案由若干轮次组成，每轮有
窗口 (window_start, cutoff]：

- 默认归属：窗口内采集的样本进 val（当时真正未知的范围）；更早采集且
  标签在截点前已可见的样本进 train；标签迟到的样本排除
  （label_not_visible），绝不进入早期训练；未采集的样本排除
  （not_collected_yet）；来源已撤回的样本排除（source_withdrawn）。
- 约束（只标记、不静默放宽）：late_label_in_train（迟到标签进训练）、
  val_outside_window（验证超出未知窗口）、family_split（家族同轮跨
  train/val）、source_straddle（来源同轮跨侧）、min_per_group（家族在
  某侧数量低于最低保留数量）。家族直接采用审查结论的分组，因此人工
  确认不同身份的样本对不参与家族约束，其相似证据仍保留在审查结论中。
- 调整提案：plan-propose 对每个条目独立计算跨全部轮次的连带影响
  （resolved / introduced）；plan-apply 逐项采纳并生成新方案版本，
  未采纳或新引入的冲突保留在违规列表中。
- 变更与重审：label-correct / source-withdraw 只追加事件（原始样本
  不变）；plan-review 对比方案的证据快照与当前事件、人工判定，列出
  需要重新审查的轮次并把方案标记为 restricted（内容不变、带限制保留）；
  plan-reopen 按当前有效标签与判定重算，生成新方案版本，父方案原样
  保留，各轮采用范围、决定时可见的证据快照与最终划分始终可对应。

## 分组规则（R001）

1. 文本先做规范化（NFKC、小写、去空白与常见中英文标点），**只影响指纹，
   不改原样本**。
2. 对规范化文本生成字符 2-gram 片段集合，用 Jaccard 相似度两两比对；
   `sim >= 0.45`（默认阈值，CLI 可调）时建立一条近重复边，边附带两段原文的
   **局部公共段落**（最长公共连续子串）作为人读证据。
3. 按相似度从强到弱做**完全连接（complete-link）凝聚聚类**：一个样本并入某组，
   必须与该组**每一个**成员都有近重复边（构成团）。`A~B、B~C` 但 `A≁C` 时，
   C 不会并入 A/B 的身份；被拒绝的边仍保留在 `edges` 中。
4. 近重复边一端在 `train`、另一端在 `val/validation/dev/test` 即标记为
   **划分泄漏证据**；`val~test` 也提示风险。
5. **人工身份判定**：
   - `distinct`（不同身份 / 排除关系）：即使相似也不参与成组、不计泄漏；
     原始相似边保留在 `excluded_edges` 中；撤回后重算即恢复。
   - `duplicate`（同身份）：强制并入同一身份，成组依据记为人工确认；
     跨划分时同样计为泄漏（相似度不伪造，低于阈值时按 0 记录并标记
     `human_forced`）。
   同一对样本同时存在两类判定会被拒绝。

## 增量修订与差异报告

`add-samples` 复制父版本并追加新样本生成新版本；`audit` 可用
`--base-audit-id` 指定基线（默认自动取同一版本最近一次审查）。结论中的
`changes` 与报告“相对基线的修订”一节指出：

- 分组：新增 / 消失 / **扩张（旧身份保留并并入新成员）** / 收缩 / 不变；
- 泄漏证据：新增、撤销或不再成立的样本对；
- 桥接但未合并证据：桥接样本、两侧组、阻断合并的跨组对（低于阈值或仅局部
  相似 / 人工排除）。

## 核对数据

`data/samples.jsonl`（误报/漏报）：完全相同 `t1/v1`、刻意改写 `t2`、只有公共
免责声明的 `t3/v2`（不成组）、链式局部相似 `c1~c2~c3`（c3 不并入但泄漏边保留）、
无关样本 `u1`。

`data/base_incremental.jsonl` + `data/incremental_samples.jsonl`（增量核对）：

- `a1/a2/a3` 为同身份旧组（train），`b1/b2` 为另一身份（b2 在 val，
  基线即有 1 条泄漏）；两组字符集完全不相交（跨组相似度 0）；
- 新样本 `d1` 是 `a1` 的**跨划分完整复制**（val）：必须新增泄漏；
- 新样本 `br1` 是**桥接样本**，与两个旧组成员都恰好达到阈值并保留局部公共
  段落，但跨组对不构成团，两个旧身份**不被整链合并**，桥接边单独留证；
- 测试还校验：增量版本重算与一次性全量导入重算结论一致（跨划分复制核对），
  且父/新版本的原始样本数量都不减少。

## 使用方法

```bash
# 1) 导入基线版本并生成指纹
python -m data_audit.cli ingest --store ./store --version-id v1 --input data/base_incremental.jsonl

# 2) 基线审查
python -m data_audit.cli audit --store ./store --version-id v1 --audit-id a1 --no-base

# 3) 增量加入样本（跨版本复制 + 追加；父版本保留）
python -m data_audit.cli add-samples --store ./store --version-id v2 \
    --parent-version-id v1 --input data/incremental_samples.jsonl

# 4) 人工确认不同身份（排除关系）/ 同身份
python -m data_audit.cli decide --store ./store --version-id v2 --decision-id h1 \
    --sample-a b2 --sample-b br1 --kind distinct --note "仅公共片段，非同一身份"
python -m data_audit.cli decide --store ./store --version-id v2 --decision-id h2 \
    --sample-a t3 --sample-b v2 --kind duplicate

# 5) 相对基线重算并出报告（分组与泄漏证据的变化会单列）
python -m data_audit.cli audit --store ./store --version-id v2 --audit-id a2 \
    --base-audit-id a1 --report

# 6) 撤回人工判定（只追加 withdraw 事件），再用新 audit_id 重算；a1/a2 报告不变
python -m data_audit.cli withdraw --store ./store --version-id v2 \
    --sample-a b2 --sample-b br1
python -m data_audit.cli audit --store ./store --version-id v2 --audit-id a3 \
    --base-audit-id a2

# 7) 第三轮：保存另一种训练/测试安排（划分版本），比较迁移影响
echo '{"sample_id": "d1", "split": "train"}' > /tmp/splits.jsonl
python -m data_audit.cli split-assign --store ./store --version-id v2 \
    --split-id sp1 --input /tmp/splits.jsonl
python -m data_audit.cli audit --store ./store --version-id v2 --audit-id a4 \
    --split-id sp1 --base-audit-id a3 --report   # 含按样本家族的迁移影响

# 8) 删除样本（跨版本复制，父版本保留；存活的人工判定随之保留）
python -m data_audit.cli remove-samples --store ./store --version-id v3 \
    --parent-version-id v2 --sample-ids d1

# 9) 随时查看任意历史审查报告
python -m data_audit.cli report --store ./store --audit-id a1

# 10) 第四轮：滚动验证划分方案（时间截点 + 后到标签）
cat > /tmp/rounds.json <<'EOF'
[{"round_id": "r1", "window_start": "2026-01-01T00:00:00Z", "cutoff": "2026-01-31T00:00:00Z"},
 {"round_id": "r2", "window_start": "2026-02-01T00:00:00Z", "cutoff": "2026-02-28T00:00:00Z"}]
EOF
python -m data_audit.cli ingest --store ./store --version-id rv1 --input data/rolling_samples.jsonl
python -m data_audit.cli audit --store ./store --version-id rv1 --audit-id ra1 --no-base
python -m data_audit.cli plan-create --store ./store --plan-id p1 +    --version-id rv1 --audit-id ra1 --rounds /tmp/rounds.json --min-per-group 2

# 调整提案：逐项列出跨轮次连带影响，再逐项采纳（冲突不静默放宽）
echo '{"round_id": "r2", "sample_id": "b2", "split": "train"}' > /tmp/items.jsonl
python -m data_audit.cli plan-propose --store ./store --plan-id p1 +    --proposal-id prop1 --input /tmp/items.jsonl
python -m data_audit.cli plan-apply --store ./store --plan-id p1 +    --proposal-id prop1 --new-plan-id p2 --items 0

# 标签更正 / 来源撤回后：列出需重审的轮次，旧方案带限制保留，再重开
python -m data_audit.cli label-correct --store ./store --version-id rv1 +    --sample-id b2 --label-available-at 2026-03-10T00:00:00Z
python -m data_audit.cli source-withdraw --store ./store --version-id rv1 --source S3
python -m data_audit.cli plan-review --store ./store --plan-id p1
python -m data_audit.cli plan-reopen --store ./store --plan-id p1 --new-plan-id p3
python -m data_audit.cli plan-show --store ./store --plan-id p3
```

## 存储布局

```
store/
  versions/<version_id>/
    samples.jsonl       # 数据版本：原始样本（永不被审查流程改动）
    fingerprints.jsonl  # 样本指纹：精确哈希、规范化哈希、2-gram 片段
    version.json        # 版本元信息；增量版本记录 parent_version_id 与新增样本
  conclusions/<audit_id>.json  # 审查结论（不可覆盖）：分组、成组依据、
                               # 全部边（含公共段落）、泄漏、排除边、桥接、
                               # 判定快照、相对基线的 changes
  splits/<split_id>.json       # 划分版本（不可覆盖）：样本划分的重新安排
  decisions/<version_id>.jsonl # 人工判定事件日志（add / withdraw，只追加）
  index.json
  plans/<plan_id>.json         # 滚动划分方案（内容不可变，仅状态/限制可更新）：
                               # 各轮划分、排除原因、约束、违规、证据快照
  plans/proposals/<id>.json    # 调整提案（逐项连带影响 resolved/introduced）
  plans/events.jsonl           # 标签更正 / 来源撤回事件（只追加）
```

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：完全相同/改写/公共段落的误报漏报、链式不传递、人工排除与强制同身份、
撤回恢复、桥接不合并、跨划分复制新增泄漏、增量与全量一致、原始样本不删除、
CLI 端到端可追溯；第三轮补充：划分迁移重算泄漏且分组保留、家族视图区分
完全复制/局部相似/人工确认不同身份、桥接在划分迁移下仍不合并、
删除样本与恢复旧划分后的增量结果与完整重算一致、
自动基线不跨划分版本沿用。
第四轮补充：迟到标签不进早期训练、验证保留当时真正未知的窗口、
家族/来源/最低保留数量约束、提案逐项连带影响与部分采纳（冲突不静默
放宽）、人工确认不同身份仍按判定处理且相似证据保留、标签更正/来源
撤回/判定变化后的轮次重审与带限制保留、重开后各轮采用范围与证据
快照及最终划分的对应、CLI 端到端。
