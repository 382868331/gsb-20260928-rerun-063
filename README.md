# 本地数据检索引擎

这是一个只依赖 Python 标准库的本地检索引擎，不依赖外部搜索服务。文档 ID 是稳定身份；新增、修改和删除先进入暂存区，在一次提交中原子可见。

## 结构

- `local_search/engine.py`：提交、读取、重启加载和查询执行。
- `local_search/segment.py`：不可变索引段、倒排表位置信息和崩溃安全文件写入。
- `local_search/query.py`：词项、字段条件和短语查询解析。
- `local_search/ranking.py`：可替换排序接口，内置常量分和 BM25。
- `tests/reference_scan.py`：小文档集顺序扫描参照实现。
- `tests/test_search_engine.py`：短语、字段、重复词、批量提交、构建中断和重启测试。

## Python API

```python
from local_search import SearchEngine

engine = SearchEngine("./data/index")
engine.upsert("doc-1", {"title": "new york news", "body": "city park opens"})
engine.commit()

engine.search("new york")
engine.search('title:"new york"')
engine.search("title:news body:park")
```

查询中的空白分隔条件是 AND 关系。短语只在同一个字段内检查连续位置，因此重复词和跨字段词序不会混算。

修改文档时复用原 ID；删除传入同一个 ID。多个操作只有调用 `commit()` 后才一起生效。排序器可通过构造函数替换，但排序结果不会改变或重新生成文档身份。

## 命令行

```bash
python -m local_search.cli --index ./data/index upsert doc-1 '{"title":"new york news"}'
python -m local_search.cli --index ./data/index search 'title:"new york"'
```

命令行的单条写入和删除会立即提交；Python API 可在一次 `commit()` 前暂存多个操作。

## 验证

```bash
python -m unittest discover -v
```

测试会先把小文档集直接顺序扫描的结果和索引查询结果对照，再模拟新索引段写入中断，确认旧 manifest 和旧索引仍可读取，最后重新创建引擎实例验证更新持久化。
