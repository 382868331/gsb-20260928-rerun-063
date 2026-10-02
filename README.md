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
engine.search('"york new"~2') # 间隔短语：保持词序、总间隔不超过 2 个词
engine.search('new -york')     # 排除词：命中 new 但不含 york 的文档
engine.search('~"new york" -body:park  # 连续短语 + 间隔短语 + 排除词可混用
```

## 固定视角分页与排名变化

`start_search()` 会自动发布一个仅供本次检索使用的不可变快照，并把查询文本、排序器参数、同义词规则、文档版本、页大小和稳定名次写入 `runs/<run-id>.json`。返回体中的 `next_cursor` 可在同一进程、重开索引或 `compact()` 后继续使用；已经确认的页只从持久化记录返回，不会因新增、删除、修改或规则升级而漏项、重复或改变语义。

```python
page = engine.start_search("~new york", page_size=20)
while page["next_cursor"] is not None:
    page = engine.fetch_page(page["run_id"], page["next_cursor"])
```

每条结果都带 `document_version`，解释中保留命中字段、实际词、短语位置和 `synonym_version`；页返回体的 `perspective` 同时保存完整同义词规则和全部文档版本。`budget` 限制每次继续最多检查多少候选文档，未检查完时 `exhausted` 为 `false`，`unexamined_candidates` 给出剩余候选数。无预算限制时，逐页拼接结果与同一视角一次性 `search_snapshot()` 的完整排序一致。

同分结果使用文档 ID 作为最终稳定次序。两次运行可用 `explain_rank_changes(old_run_id, new_run_id)` 对比，原因分为：

- `match_condition_added` / `match_condition_removed`：文档新进入或离开命中集合。
- `match_condition_changed`：原词/同义词、字段、位置或短语链等命中条件改变。
- `score_changed`：命中条件相同但排序分值改变。
- `relative_position_changed`：分值和证据相同，但其他文档进入或离开造成相对名次变化。

对比结果中的旧/新证据分别引用各自运行的快照，因此位置和短语解释可回到当时的文档与规则版本。也可以把已发布快照名传给 `start_search(..., snapshot="v1")`，在同一个人工研究视角上启动多次运行。

查询中的空白分隔条件是 AND 关系。带 `-` 前缀的条件是排除条件，命中文档将被剔除；字段限定（如 `-title:news`）和短语（如 `-"new york"`）同样支持排除。`"..."~N` 是间隔短语：各词必须保持查询中的先后顺序，允许中间插入其他词，但所有间隔加起来不超过 `N`（`~0` 等价于严格连续，也可以省略）。同义词 `~` 与距离限制可以叠加，例如 `~"fresh park"~3`。

短语只在同一个字段内检查位置，因此跨字段的片段不会拼成短语（同义词扩展也只替换同字段同一位置上的词）。一个查询里重复出现的词必须落在不同位置上：`"repeat repeat"` 要求两个相邻的 `repeat`，`"repeat repeat"~1` 要求中间至多隔一个词，重复词不能重复占用同一个位置。间隔短语会从全部合法位置链中确定性地选择“原词最多、跨度最小、起点最早”的一条，因此无论段、倒排表还是子句以什么顺序执行，结果与解释都一致。

每条结果带 `explanation`：`kind` 为 `original` 或 `synonym`，`matched_as` 给出实际命中的词，`position` 给出该词实际占用的字段内位置（每个子句还汇总 `positions` 与 `slop`），用户可以据此区分原词匹配和规则扩展；`synonym_version` 标明所用规则版本，BM25 下还带每个命中词的分数贡献。`excluded_clauses` 列出本次查询实际应用的排除条件原文。

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
python -m local_search.cli --index ./data/index search-page '~"new york"' --page-size 20
```

命令行的单条写入和删除会立即提交；Python API 可在一次 `commit()` 前暂存多个操作。

## 验证

```bash
python -m unittest discover -v
```

测试会先把小文档集（含重复词、歧义词组、修改与删除）在两种规则版本下与直接顺序扫描的结果逐条对照，检查解释中的原词/同义词标注与实际位置、跨字段短语不拼接、间隔短语的距离边界、排除词、多段索引下任意子句执行顺序结果相同、在途查询保留旧版本，再模拟索引整理中断确认已发布快照与旧 manifest 仍可查询，并对快照在规则升级、文档修改和整理前后的全部混合查询逐条核对直接扫描结果。
