# 本地数据检索引擎

只依赖 Python 标准库的本地检索引擎。文档 ID 是稳定身份；新增、修改和删除先进入暂存区，在一次提交中原子可见。本版本增加同义词扩展、可解释排序、可查询的历史快照和索引整理。

## 结构

- `local_search/engine.py`：提交、快照、规则版本、索引整理和查询执行。
- `local_search/segment.py`：不可变索引段、倒排表位置信息和崩溃安全文件写入。
- `local_search/matcher.py`：严格/扩展匹配，记录每个词是原词还是同义词命中。
- `local_search/analysis.py`：分词与不可变同义词规则表。
- `local_search/query.py`：词项、字段条件、短语查询和 `~` 扩展语法。
- `local_search/ranking.py`：可替换排序接口，常量分和带逐词贡献说明的 BM25。
- `tests/reference_scan.py`：基于同一匹配器的顺序扫描参照实现。
- `tests/test_search_engine.py`：重复词、歧义同义词、修改删除、版本固定、整理中断和排序身份测试。

## 同义词与短语

同义词规则是不可变版本：用 `set_synonyms` 暂存并随下一次 `commit()` 发布新的 `rules_version`。

```python
engine.set_synonyms({
    "york": ["metropolis"],        # 单词同义词，可参与短语扩展
    "new": ["fresh", "first"],     # 一词多义：说明中报告实际命中的词义
    "park": ["municipal garden"],  # 多词同义词，只参与宽松 AND 子句
})
engine.commit()
```

匹配模式：

- 严格短语（`"new york"`、`title:"new york"`）：只按原词序匹配相邻位置，从不使用同义词。
- 扩展短语（`~"new york"`、`title:~"new york"`，或查询以 `~` 开头）：每个位置允许原词或单词同义词替换；多词同义词不参与短语。
- 宽松子句（`~park`）：允许同义词（含多词短语同义词）命中。
- 短语只在同一个字段内检查连续位置，跨字段拼接永远不算短语；一个 AND 子句的多个词也必须落在同一字段。

查询结果默认带 `explanation`，逐词标注：

- `match`：`original` 或 `synonym`。
- `matched`：文档中实际命中的词（同义词命中时与查询词不同）。
- `positions`：命中文档字段内的位置（重复词逐个报告）。
- `score`：BM25 的 `tf`、`idf`、`contribution` 和总分加和；常量排序时全部为 0。

```python
engine.search('title:~"new york"', explain=True)
engine.search("york", explain=False)  # 只要 id/score/fields
```

## 版本与快照

- 规则更新后新查询自动使用最新版本；在途查询应持有 `snapshot()` 返回的不可变快照，快照固定文档集合与规则版本。
- 每次发布把旧 manifest 归档到 `history/manifest-N.json`，旧段文件不删除，旧快照始终可查询：

```python
old = engine.snapshot(1)          # 第一次提交时的完整视图
old.search("york")                # 即使文档后来被修改/删除/整理
engine.snapshots()                # 可查询的版本号
```

- `compact()` 把当前可见文档合并成一个新段：先写新段并归档旧 manifest，最后原子替换 manifest；整理中断只留下未引用的临时/新段文件，已发布快照和当前 manifest 不受影响。

排序只改变顺序：同分按稳定文档 ID 排序，文档身份始终是提交时使用的 ID，不由分数或内容重新生成。

## 基础 API

```python
from local_search import SearchEngine

engine = SearchEngine("./data/index")
engine.upsert("doc-1", {"title": "new york news", "body": "city park opens"})
engine.commit()

engine.search("new york")
engine.search('title:"new york"')
engine.search("title:news body:park")
engine.get("doc-1")
```

空白分隔的条件是 AND 关系。修改文档复用原 ID；删除传同一个 ID。多个操作在一次 `commit()` 中一起生效。

## 命令行

```bash
python -m local_search.cli --index ./data/index upsert doc-1 '{"title":"new york news"}'
python -m local_search.cli --index ./data/index synonyms '{"york":["metropolis"]}'
python -m local_search.cli --index ./data/index search 'title:~"new york"'
python -m local_search.cli --index ./data/index search york --snapshot 1
python -m local_search.cli --index ./data/index snapshots
```

单条写入/删除和同义词发布立即提交；Python API 可在一次 `commit()` 前暂存多个操作。

## 验证

```bash
python -m unittest discover -v
```

测试把含重复词和歧义同义词的文档集与直接顺序扫描对照，验证严格短语拒绝跨字段拼接、扩展匹配逐词说明、修改删除后旧快照仍可查询、排序变化不改变文档身份，以及索引段写入和整理中断后旧 manifest/旧索引依旧可读。
