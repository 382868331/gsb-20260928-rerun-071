# 训练数据审查系统

五轮实现：

- **第一轮**：近重复样本分组与训练/验证（测试）划分泄漏检查；
- **第二轮**：增量加入样本后的近重复组修订、人工确认同/异身份（可撤回）、
  修订差异报告。系统只标记并给出人工复核建议，**不会自动删除或修改任何样本**。
- **第三轮**：同一数据版本上的多个**划分版本**（训练/测试安排）：相似关系与
  人工身份判定跨划分版本保留，泄漏结论按各自划分版本重算、不跨版本沿用；
  按样本家族查看迁移影响（区分完全复制 / 局部相似 / 人工确认不同身份）；
  支持跨版本复制式删除样本。撤回判定、删除样本、恢复旧划分后的增量结果
  与该版本完整重算一致。
- **第四轮**：滚动验证划分与后到标签。按多个时间截点形成滚动训练轮次，
  每轮只能采用截点前已可见的信息：采集晚于截点的样本不参与该轮，
  采集在截点前但标签实际可见时间晚于截点的样本只进入该轮验证范围
  （当时真正未知的范围），后来补到的标签不会回流进早期训练。规划同时
  考虑样本家族同侧、来源隔离与每轮最低保留数量；调整项逐项列出对全部
  轮次的连带影响（改善某轮泄漏可能破坏另一轮数量约束），由用户逐项
  采纳，冲突只列出、绝不静默放宽。标签更正、来源撤回或人工身份判断
  改变时列出需要重新审查的轮次，旧方案可标记限制后保留。
- **第五轮**：有限标注预算下的来源配额补齐。样本可用 `sources` 列出其实际
  涉及的多个来源，跨来源人工确认同身份时作为一个身份单位只支付一次标注成本；
  系统比较“最少新增标注”和“优先补最紧缺来源”两种方案。选择严格按指定截点
  可见的标签、撤回事件与身份快照计算，迟到资料不回流。来源撤销或身份合并后，
  旧批次中尚未执行的选择只作废重算，已执行的标注成本作为事实保留；同基础、
  同方案重开不会重复安排，预算不足时逐来源返回缺口与原因。

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

## 滚动验证（第四轮）

- **可见性**：样本可带 collected_at（采集时间）、label_available_at
  （标签实际可见时间）与 source（来源）。每轮截点把样本分为
  train（采集与标签均不晚于截点）/ val（采集不晚于截点、标签后到，
  即当时真正未知的范围）/ future（采集晚于截点）/ withdrawn（来源已撤回）。
- **约束**：family_together（家族同侧，家族来自证据快照审查结论，
  已确认不同身份的样本仍按该判断处理、原始相似证据不删除）、
  isolated_sources（来源隔离）、min_train / min_val（每轮最低保留
  数量）、min_train_per_source / min_val_per_source（**逐来源族**
  最低保留量）：每个来源族在每轮 train/val 侧分别达标，整体数量
  足够不能代替逐组达标，也不借用别组样本凑数。逐组不达标时保留
  失败原因与可用数量，并区分“标签尚未可见”（只能留在当时未知的
  验证范围，之后可能转为可训练）与“来源已撤回”（全部轮次不可用）；
  每轮落盘 source_status 明细（train / val(标签未到) / future /
  val(身份隔离) / val(人工调整) / 身份隔离剔除 / 人工调整剔除 /
  来源撤销），分别记录四类数量缺口：**标签尚未可见**（标签在截点后
  才可见，自然落在当时未知的验证范围，之后可能转为可训练）、
  **人工划分调整**（来源隔离等人工调整把样本让入另一侧或剔除）、
  **身份隔离**（样本标签截点前已可见，仅因同一实体跨来源重复出现、
  为同实体隔离而被人工移到验证侧或剔除；**不计 pending_label、不
  解释成等待标签**）、**来源撤销**（全部轮次不可用；人工移动后又
  撤销的按撤销记录）。实际 train/val 数量始终按最终划分如实统计。
  同一实体跨来源重复出现（人工确认同身份）时，家族同侧约束保证它
  不会同时泄漏到 train 与 val 两侧。冲突全部列出，不会通过静默放宽
  条件消失。
- **调整项与连带影响**：rolling-eval 给出候选调整项，每项的 effects
  列出各轮冲突变化（解决了什么、新引入了什么）；rolling-adopt 逐项
  采纳并生成新修订（rev_n 不可覆盖，旧修订原样保留）。
- **事件与重审**：rolling-event 追加标签更正 / 来源撤回 / 判定改变
  （只追加）；rolling-review 列出每个事件触发后需要重新审查的轮次；
  rolling-restrict 把旧修订标记为带限制保留；rolling-verify 重算
  最新修订并核对各轮采用范围、决定时可见的证据快照与最终划分一致。

```bash
# 1) 导入带时间与来源的样本并出基线审查（证据快照）
python -m data_audit.cli ingest --store ./store --version-id v1 --input data/rolling_samples.jsonl
python -m data_audit.cli audit --store ./store --version-id v1 --audit-id a1 --no-base

# 2) 创建滚动方案：两个截点、最低保留数量、来源隔离
python -m data_audit.cli rolling-init --store ./store --version-id v1 --audit-id a1 \
    --plan-id rp1 --rounds r1=2026-02-01,r2=2026-03-01 \
    --min-train 5 --min-val 1 --isolated-sources srcB \
    --min-train-per-source srcA=2,srcB=1 --min-val-per-source srcB=1

# 3) 查看候选调整项（含各轮连带影响），逐项采纳
python -m data_audit.cli rolling-eval --store ./store --plan-id rp1
python -m data_audit.cli rolling-adopt --store ./store --plan-id rp1 --items i2

# 4) 标签更正 / 来源撤回 / 判定改变：列出需要重新审查的轮次
python -m data_audit.cli rolling-event --store ./store --plan-id rp1 \
    --kind label_correction --sample-ids rd1 --label-available-at 2026-01-28
python -m data_audit.cli rolling-event --store ./store --plan-id rp1 \
    --kind source_withdrawn --source srcB
python -m data_audit.cli rolling-review --store ./store --plan-id rp1

# 5) 旧方案带限制保留；重新采纳后核对一致性
python -m data_audit.cli rolling-restrict --store ./store --plan-id rp1 \
    --revision 1 --note 标签更正后待重审
python -m data_audit.cli rolling-verify --store ./store --plan-id rp1
```

## 来源配额标注（第五轮）

样本可在 JSONL 中使用 `sources: ["srcA", "srcB"]` 表示同一样本实际涉及多个
来源；这不是把样本复制成两份，因此一次标注只消耗一次预算，并同时保留其来源
身份关系。人工确认为同一身份的跨来源样本也按同一身份单位处理。

```bash
# 在 r1 截点比较两种补齐策略，并保存所选策略的下一轮候选批次
python -m data_audit.cli rolling-quota --store ./store --plan-id rp1 \
    --cutoff 2026-02-01 --needs srcA=2,srcB=1 --budget 3 \
    --strategy minimum_new_labels
python -m data_audit.cli rolling-quota --store ./store --plan-id rp1 \
    --cutoff 2026-02-01 --needs srcA=2,srcB=1 --budget 3 \
    --strategy most_shortage_first

# 人工实际核对后登记成本；不登记时后续变化可作废尚未执行的选择
python -m data_audit.cli rolling-quota-execute --store ./store --plan-id rp1 \
    --sample-ids multi --at 2026-02-02
```

历史事件必须给出生效时刻；默认当前时刻只影响不早于该时刻的截点：

```bash
python -m data_audit.cli rolling-event --store ./store --plan-id rp1 \
    --kind source_withdrawn --source srcB --at 2026-01-31
```

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
  rolling/<plan_id>/plan.json  # 滚动方案：轮次截点、约束、证据快照 audit_id
  rolling/<plan_id>/rev_<n>.json  # 每次采纳生成一个修订（不可覆盖）：
                               # 已采纳调整项、各轮采用范围、候选调整项
  rolling/<plan_id>/events.jsonl  # 标签更正/来源撤回/判定改变（只追加）
  rolling/<plan_id>/restrictions.jsonl  # 旧修订的限制标记（只追加）
  rolling/<plan_id>/quota_batches.jsonl # 标注方案、作废记录与已执行成本（只追加）
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
第四轮补充：迟到标签只进当时未知的验证范围、同家族跨截点与来源隔离
冲突、调整项连带影响（含跨轮）、部分采纳后冲突仍列出、标签更正/
来源撤回/判定改变触发轮次重审、带限制旧方案保留、重开后采用范围与
证据快照核对一致。
逐来源族约束补充：整体数量足够不能代替逐组达标、缺口原因区分
“标签尚未可见”/“人工划分调整”/“身份隔离”/“来源撤销”四类并保留
可用数量（标签截点前已可见、仅为同实体隔离而人工移动的样本不算
等待标签）、晚标签+来源撤销+跨来源身份合并共同发生时的分配与保存
恢复（人工移动后撤销按撤销记录、标签时间更正后缺口消失并重算、
保存重开核对一致、新截点派生新方案后旧方案划分与旧解释不变、
带限制旧修订保留，旧修订不被新解释覆盖）。
