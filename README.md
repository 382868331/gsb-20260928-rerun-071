# 训练数据审查系统（第一轮）

第一轮实现两件事：**近重复样本分组**与**训练/验证（测试）划分泄漏检查**。
系统只标记并给出人工复核建议，**不会自动删除或修改任何样本**。

## 设计原则

- **三类数据分开保存，均可追溯原样本**：数据版本（原始样本）、样本指纹、
  审查结论分别落盘，通过 `version_id` / `sample_id` 关联。以后加入更多质量
  规则时，原始样本与既有指纹都可以复用、重放。
- **相似关系不无限制传递**：局部相似的边不会沿传递链把一串内容塌缩成一个
  身份（见下）。
- **证据与判定分开**：近重复边是“局部相似证据”，分组是“身份判定”，泄漏边
  是“跨划分证据”；三者都写入结论文件与报告。

## 分组规则（R001）

1. 文本先做规范化（NFKC、小写、去空白与常见中英文标点），**只影响指纹，
   不改原样本**。
2. 对规范化文本生成字符 2-gram 片段集合，用 Jaccard 相似度两两比对；
   `sim >= 0.45`（默认阈值，CLI 可调）时建立一条近重复边。
3. 按相似度从强到弱做**完全连接（complete-link）凝聚聚类**，得到互斥分组：
   一个样本并入某组，必须与该组**每一个**成员都有近重复边（构成团）。
   - 因此 `A~B、B~C` 但 `A≁C` 时，C 不会并入 A/B 的身份；
   - 被拒绝的边仍作为证据保留在 `edges` 中。
4. 一条近重复边若一端在 `train`、另一端在 `val/validation/dev/test`，即标记为
   **划分泄漏证据**；`val~test` 也会提示风险，同划分近重复则只提示去重复核。

> 为什么不用并查集/连通分量？连通分量会无限传递相似关系，链式局部相似会被
> 错误归为同一身份。完全连接保证组内两两足够相似。

## 误报/漏报核对数据（`data/samples.jsonl`）

| 样本 | 用途 | 预期 |
| --- | --- | --- |
| `t1`(train) / `v1`(val) | **完全相同** | 相似度 1.0，报跨划分泄漏 |
| `t2`(train) | 对 `t1` 的**刻意改写**（同义词/语序） | 与 t1/v1 同身份，报泄漏 |
| `t3` / `v2` | **只有公共片段**（相同免责声明，主体不同） | 不成组、不报泄漏（防误报） |
| `c1~c2`（同 train）、`c2~c3`(c3 在 val) | 链式局部相似 | c1/c2 同身份，c3 不并入；c2~c3 边单独作为泄漏证据 |
| `u1` | 无关样本 | 无任何近重复边 |

## 使用方法

```bash
# 1) 导入数据版本（JSONL：每行 {sample_id, split, text}），生成并保存指纹
python -m data_audit.cli ingest --store ./store --version-id v1 --input data/samples.jsonl

# 2) 运行近重复 + 泄漏检查（可选 --threshold，--report 直接打印报告）
python -m data_audit.cli audit --store ./store --version-id v1 --audit-id a1 --report

# 3) 之后随时重看某次审查报告
python -m data_audit.cli report --store ./store --audit-id a1
```

## 存储布局

```
store/
  versions/<version_id>/
    samples.jsonl       # 数据版本：原始样本（永不被审查流程改动）
    fingerprints.jsonl  # 样本指纹：精确哈希、规范化哈希、2-gram 片段
    version.json        # 版本元信息
  conclusions/<audit_id>.json   # 审查结论：规则、分组、边、泄漏证据、建议
  index.json
```

## 测试

```bash
python -m pytest tests/ -q
```
