# 本地数据检索引擎

这是一个无外部服务依赖的 Python 本地检索引擎。文档以调用方提供的稳定 `doc_id` 标识身份；新增、修改和删除进入事务暂存区，在 `commit()` 成功后原子可见。

## 能力

- 不可变索引段与 JSON 清单原子切换，构建中断时继续使用上一份索引。
- 字段级倒排链和词位置，支持词项、短语、`AND` / `OR` / `NOT` 布尔查询。
- 查询语法支持 `字段:词` 和 `字段:"完整短语"`。
- 排序器可替换；排序器不会改变或重新定义文档身份。
- 重启后直接从本地目录读取已提交数据。

## 快速使用

```python
from local_data_retrieval import LocalIndex, PhraseQuery, TermQuery, And

index = LocalIndex("data/index")
with index.writer() as writer:
    writer.upsert_document("doc-1", {
        "title": "local engine",
        "body": "build a local retrieval engine",
    })
    writer.delete_document("old-doc")
    writer.commit()

hits = index.search(And(
    TermQuery("title", "engine"),
    PhraseQuery("body", "local retrieval"),
))
```

自定义排序：

```python
hits = index.search(query, ranker=lambda hit: hit.doc_id, reverse=True)
```

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
