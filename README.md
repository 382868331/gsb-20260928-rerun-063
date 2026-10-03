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
```

命令行的单条写入和删除会立即提交；Python API 可在一次 `commit()` 前暂存多个操作。

## 验证

```bash
python -m unittest discover -v
```

测试会先把小文档集（含重复词、歧义词组、修改与删除）在两种规则版本下与直接顺序扫描的结果逐条对照，检查解释中的原词/同义词标注与实际位置、跨字段短语不拼接、间隔短语的距离边界、排除词、多段索引下任意子句执行顺序结果相同、在途查询保留旧版本，再模拟索引整理中断确认已发布快照与旧 manifest 仍可查询，并对快照在规则升级、文档修改和整理前后的全部混合查询逐条核对直接扫描结果。

## 多查询研究集

研究集把若干查询在同一研究视角下形成的候选文档固定保存，之后逐页审阅。创建时可指定组合方式：`mode="any"` 保留满足任意查询的文档，`mode="all"` 只保留同时满足全部查询的文档。每篇文档逐查询保留自己的命中来源、分数和位置证据（`per_query`），跨查询分数绝不相加成统一相关性；研究集内的相对顺序按“各查询自己的最优名次 → 命中查询数 → 完整名次向量 → 文档 ID”确定性确定。

```python
research = engine.create_research_set(
    "study-2026-10",
    {"q1": '~"new york"', "q2": "fresh -body:park"},
    mode="any",
    budget=200,          # 每个查询最多完整评分多少候选；None 表示一次穷尽
)
research.page(0, page_size=20)      # 按稳定顺序翻页，不重复不遗漏
research.read_page(20)              # 读取下一页并标记为已审阅
research.review_status()            # 已审阅/未审阅/人工排除/待评分候选
research.decide("d9", "excluded", "人工复核：与研究问题无关")
research.decide("d12", "included", "例外：作者人工标注，虽未命中查询")
research.save()                     # 持久化；重开后顺序、已读位置、决定都保持
```

研究集保存时固定查询文本、完整同义词规则版本以及当时的文档体（pinned 视图），所以之后实时资料新增、修改、删除、同义词升级或整理索引，已确认的页都保持原顺序，证据位置也不漂移。

**预算与未穷尽**：`budget` 未覆盖全部候选时，`exhausted` 为 `False`，未评分的命中进入 `pending_unscored`，页载荷的 `complete` 也是 `False`。暂时没出现的文档不能判断为不满足；调用 `continue_scan(更大预算或 None)` 在同一固定视角上继续评分，已确认页不变，之后的文档按同一顺序规则接续——逐页读取与一次性完整读取的顺序完全一致，同分文档按文档 ID 稳定打破平局。

**修订对比**：用修订后的查询重新生成研究集时，`revise_research_set()` 返回每篇文档 `entered`/`left`/`rank_moved`/`stable` 及原因，区分 `document_changed`（文档新增、删除或正文变化）、`query_meaning_changed`（查询文本或同义词版本变化）和 `relative_order_from_other_candidates`（只是被其他候选的进出带动了相对名次）。旧研究集的人工排除/例外只在依据仍相符时继承：系统比较当时记录的逐查询证据签名（具体命中词、字段、位置），文档删除或证据变化则放入 `dropped_decisions` 要求重新审阅，绝不会只按文档 ID/名称复制决定。

命令行：

```bash
python -m local_search.cli --index ./data/index research-new study '{"q1":"~new","q2":"york"}' --mode any --budget 200
python -m local_search.cli --index ./data/index research-page study --page-size 20
python -m local_search.cli --index ./data/index research-read study --page-size 20
python -m local_search.cli --index ./data/index research-status study
python -m local_search.cli --index ./data/index research-decide study d9 excluded "人工复核：无关"
python -m local_search.cli --index ./data/index research-continue study --budget 1000
python -m local_search.cli --index ./data/index research-revise study '{"q1":"~fresh","q2":"york"}'
```
