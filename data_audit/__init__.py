"""训练数据审查系统（第一轮）：近重复分组与数据划分泄漏检查。

设计要点：
- 数据版本、样本指纹、审查结论分层存储，后续新增质量规则时仍可追溯原样本。
- 相似关系不做无限传递：组内必须两两相似（完全连接），局部相似只记录为证据。
- 只产出审查结论，不自动删除或修改任何原始样本。
"""

from .fingerprints import FingerprintConfig, TextFingerprinter
from .grouping import GroupingConfig, SimilarityEdge, build_groups
from .pipeline import AuditPipeline, AuditResult
from .store import AuditStore

__all__ = [
    "FingerprintConfig",
    "TextFingerprinter",
    "GroupingConfig",
    "SimilarityEdge",
    "build_groups",
    "AuditPipeline",
    "AuditResult",
    "AuditStore",
]
