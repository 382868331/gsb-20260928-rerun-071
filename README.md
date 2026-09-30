# 训练数据审查系统

三轮实现：

- **第一轮**：近重复样本分组与训练/验证（测试）划分泄漏检查；
- **第二轮**：增量加入样本后的近重复组修订、人工确认同/异身份（可撤回）、
  修订差异报告。系统只标记并给出人工复核建议，**不会自动删除或修改任何样本**。
- **第三轮**：同一数据版本上的**多划分版本**（多套 train/val/test 安排）与
  划分迁移对比；泄漏结论对应各自划分版本，按样本家族查看迁移影响；
  删除样本同样靠跨版本复制，父版本原样保留。

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
- **泄漏结论对应各自划分版本**：相似关系（近重复边、分组）与人工身份判定
  与划分无关、跨划分版本保留；但泄漏结论按各自划分版本计算并记录
  `split_version_id`，自动基线只在同一划分版本内选取，不能跨划分版本
  直接沿用。划分版本本身不可变：迁移生成新划分版本，旧划分版本原样保留，
  恢复旧划分后重算与旧划分版本的完整重算一致。
- **桥接不整链合并**：新样本同时连接两个家族时，只靠连接存在不会合并全部
  成员；只有跨组成员构成团才合并，未采纳的桥接边保留为未合并证据。

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

### 划分版本与迁移对比（第三轮）

```bash
# 1) 为数据版本定义一套划分安排（JSONL：每行 {"sample_id", "split"}）
python -m data_audit.cli define-split --store ./store --split-version-id sp1 \
    --version-id v1 --input data/split_base.jsonl

# 2) 迁移到新安排：复制父划分版本并应用变更，旧划分版本原样保留
python -m data_audit.cli revise-split --store ./store --split-version-id sp2 \
    --parent-split-version-id sp1 --input data/split_changes.jsonl

# 3) 分别在各划分版本下审查：泄漏结论各自独立（split_version_id 落盘）
python -m data_audit.cli audit --store ./store --version-id v1 \
    --audit-id a-sp1 --split-version-id sp1 --no-base
python -m data_audit.cli audit --store ./store --version-id v1 \
    --audit-id a-sp2 --split-version-id sp2

# 4) 迁移对比：按样本家族查看影响，区分完全复制 / 局部相似 /
#    人工确认不同身份三类证据
python -m data_audit.cli compare --store ./store --version-id v1 \
    --from-split sp1 --to-split sp2 --compare-id cmp1

# 5) 删除样本：跨版本复制父版本并去掉指定样本，父版本原样保留；
#    新版本重算与剩余样本一次性导入的完整重算一致
python -m data_audit.cli remove-samples --store ./store --version-id v3 \
    --parent-version-id v1 --sample-ids b2
```

### 基础流程（第一、二轮）

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

# 7) 随时查看任意历史审查报告
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
  decisions/<version_id>.jsonl # 人工判定事件日志（add / withdraw，只追加）
  splits/<split_version_id>.json   # 划分版本：sample_id -> split 安排（不可变），
                                   # 记录父划分版本与应用的变更
  comparisons/<compare_id>.json    # 划分迁移对比结论（不可覆盖）：
                                   # 按家族的证据分类与泄漏变化
  index.json
```

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：完全相同/改写/公共段落的误报漏报、链式不传递、人工排除与强制同身份、
撤回恢复、桥接不合并、跨划分复制新增泄漏、增量与全量一致、原始样本不删除、
CLI 端到端可追溯；多划分版本下泄漏结论各自独立、按家族区分完全复制/局部
相似/人工确认不同身份、桥接在任意划分下不整链合并、撤回判定/删除样本/
恢复旧划分后与完整重算一致、划分迁移 CLI 端到端。
