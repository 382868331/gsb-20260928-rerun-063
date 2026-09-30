# 本地数据检索引擎

这是一个只依赖 Python 标准库的本地检索引擎，不依赖外部搜索服务。文档 ID 是稳定身份；新增、修改和删除先进入暂存区，在一次提交中原子可见。

支持版本化同义词扩展与可解释排序：短语可以严格按原词序匹配，也可以在词项位置上用同义词扩展匹配；结果会解释每个词命中的是原词还是具体哪个同义词。任何已发布快照都保存其当时的完整同义词规则版本，规则升级后仍可独立查询。

## 结构

- `local_search/engine.py`：提交、读取、重启加载和查询执行。
- `local_search/segment.py`：不可变索引段、倒排表位置信息和崩溃安全文件写入。
- `local_search/query.py`：词项、字段条件和短语查询解析。
- `local_search/ranking.py`：可替换排序接口，内置常量分和 BM25。
- `local_search/synonyms.py`：不可变、带版本号的同义词规则（允许歧义词组）。
- `local_search/view.py`：不可变查询视图，负责原词/同义词短语链匹配和命中解释；实时索引与快照共用同一视图。
- `tests/reference_scan.py`：小文档集顺序扫描参照实现。
- `tests/test_search_engine.py`：重复词、同义词歧义、修改删除对照、在途查询版本、快照与整理中断测试。

## Python API

```python
from local_search import SearchEngine

engine = SearchEngine("./data/index")
engine.upsert("doc-1", {"title": "new york news", "body": "city park opens"})
engine.commit()

engine.search("new york")
engine.search('title:"new york"')
engine.search("title:news body:park")

# 同义词扩展：词项前或字段后加 ~ 表示允许同义词命中
from local_search import SynonymRules

engine.set_synonym_rules(
    SynonymRules.create(1, [["new", "fresh"], ["york", "city"]])
)
engine.search('~"new york"')   # 扩展短语：fresh city 也命中
engine.search('~title:fresh')  # 等价写法 title:~fresh
engine.search('"new york"')    # 不带 ~：严格原词序，只用原词
```

查询中的空白分隔条件是 AND 关系。短语只在同一个字段内检查连续位置，因此重复词和跨字段词序不会混算（同义词扩展也只替换同字段同一位置上的词，跨字段不会拼成短语）。

每条结果带 `explanation`：`kind` 为 `original` 或 `synonym`，`matched_as` 给出实际命中的词，`synonym_version` 标明所用规则版本，BM25 下还带每个命中词的分数贡献。

## 规则版本、在途查询与快照

`set_synonym_rules()` 要求版本号严格递增；新版本立即对新查询生效。已经解析并通过 `query.with_rules(旧规则)` 绑定的查询对象保持自己的完整规则版本，在规则升级后继续按旧语义执行。

`publish_snapshot(name)` 固化当前可见文档集合、段引用、删除标记和完整同义词规则；之后的文档修改删除、规则升级和索引整理都不影响快照。`compact()` 通过“写新段 + 原子换 manifest”整理索引，旧段文件保留给已发布快照，整理中断时已发布 manifest、段和快照均可继续查询。

```python
engine.publish_snapshot("v1")
engine.search_snapshot("v1", '~"new york"')
engine.compact()
```

修改文档时复用原 ID；删除传入同一个 ID。多个操作只有调用 `commit()` 后才一起生效。排序器可通过构造函数替换，但排序结果不会改变或重新生成文档身份。

## 命令行

```bash
python -m local_search.cli --index ./data/index upsert doc-1 '{"title":"new york news"}'
python -m local_search.cli --index ./data/index search 'title:"new york"'
python -m local_search.cli --index ./data/index snapshot v1
python -m local_search.cli --index ./data/index snapshot-search v1 '~"new york"'
```

命令行的单条写入和删除会立即提交；Python API 可在一次 `commit()` 前暂存多个操作。

## 验证

```bash
python -m unittest discover -v
```

测试会先把小文档集（含重复词、歧义词组、修改与删除）在两种规则版本下与直接顺序扫描的结果逐条对照，检查解释中的原词/同义词标注、跨字段短语不拼接、在途查询保留旧版本，再模拟索引整理中断确认已发布快照与旧 manifest 仍可查询。
