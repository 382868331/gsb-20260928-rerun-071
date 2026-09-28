# 训练数据审查系统（第一轮）

第一轮实现两个质量规则：**近重复样本分组** 与 **训练/验证数据划分泄漏检查**。
纯 Python 标准库实现，无第三方依赖。

## 设计原则

### 1. 分层存储，结论可追溯

运行一次审查会在存储目录（默认 audit_store/）写入四类 JSONL：

| 文件 | 内容 |
| --- | --- |
| versions.jsonl | 数据版本、来源文件、sample_id、划分、**原始文本** |
| fingerprints.jsonl | 规范化 SHA-256、长度、词元数、MinHash 签名，带 fingerprint_version |
| findings.jsonl | 审查结论与证据，带 run_id、dataset_version、规则版本 |
| runs.jsonl | 每次运行的配置版本、规则清单与统计 |

后续新增质量规则时，原样本版本不变，新结论通过 sample_id / dataset_version
即可回溯到原样本；历史结论只追加、不覆盖。

### 2. 相似关系不做无限传递

这是本轮最关键的规则。相似（Jaccard 高）不是等价关系：
A 与 B 局部相似、B 与 C 局部相似，并不意味着 A 与 C 是同一条样本。
因此**不使用并查集/连通分量**做传递闭包，而采用完全连接（complete-linkage）规则：

1. **完全相同**：规范化（NFKC、小写、空白折叠）后 SHA-256 相等 → 同组。
   相等天然可传递，按哈希直接分桶。
2. **近重复边**：两条样本同时满足
   - 混合词元（拉丁词 + 汉字 unigram/bigram + 标点）Jaccard
     >= 0.8（--threshold 可调）；
   - 较短/较长文本长度比 >= 0.7（--min-length-ratio 可调），
     防止“只有一小段公共片段”的样本被误判。
3. **入组条件**：样本必须与组内**每一条**已有成员都存在近重复边。
   链式数据 A~B~C 但 A≁C 时，A 与 C 不会归为同一身份。
4. 未并入身份组的近重复边保留为 cross_split_local_similarity（severity=info）
   证据，既不丢弃也不升级为泄漏。

### 3. 只标记，不删除

所有结论的 action 均为 flag_only_no_auto_deletion。系统不修改、不移动、
不删除任何输入样本，由人工根据证据决定后续处理。

## 划分泄漏证据

当一条 exact/near 身份边跨越 train 与 validation（或任意两个不同 split）时，
逐条输出 split_leak_pair（severity=high），证据包含：

- 两侧 sample_id 与所属划分；
- Jaccard 分数、长度比、最长公共子串（原文片段）；
- 所属重复组 ID 与分组规则版本。

## 使用

```bash
# 审查（支持 .jsonl 与 .json，每条样本含 text、split，可选 id）
python -m data_audit.cli audit examples/demo_samples.jsonl \
    --version v2026-09-29 --store-dir audit_store

# 汇总查看
python -m data_audit.cli report audit_store

# 测试
python -m pytest tests/ -q
```

也可在代码中使用：

```python
from data_audit import AuditPipeline, AuditStore

result = AuditPipeline(AuditStore("audit_store")).run("samples.jsonl", "v1")
for group in result.groups:
    print(group.kind, group.sample_ids)
```

## 对照验证（误报 / 漏报）

examples/demo_samples.jsonl 与 tests/test_audit.py 覆盖四类刻意构造的数据：

- **完全相同**：train-a1/train-a2/val-a3 三处同文 → exact 组 + 跨划分泄漏；
- **刻意改写**：train-b1/val-b2 调序、替换少量连接词（Jaccard≈0.81）
  → near 组 + 泄漏，验证不漏报；
- **只有公共片段**：train-c1/val-c2 仅共享“系统调用接口”（Jaccard≈0.14）
  → 不成组、不报泄漏；完全无关样本 train-d1/val-d2 同样不误报；
- **链式局部相似**：train-chain-a ~ train-chain-b ~ val-chain-c
  （0.82/0.82/0.67）→ A、C 不合并身份，B-C 只保留 info 级证据。
