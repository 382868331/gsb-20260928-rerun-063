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
python -m local_search.cli --index ./data/index research-status study
python -m local_search.cli --index ./data/index research-decide study d9 excluded "人工复核：无关"
python -m local_search.cli --index ./data/index research-continue study --budget 1000
python -m local_search.cli --index ./data/index research-revise study '{"q1":"~fresh","q2":"york"}'
```

## 多视角审阅与合并

同一个冻结研究集可以挂多个独立审阅视角（perspective）。每个视角有自己的已读位置和决定：一个审阅者翻页或下判断，不会改变另一个视角的未读队列，也不会覆盖其对同一条证据的相反决定。不指定视角时，所有既有 API 默认作用于 `main` 视角，行为与单审阅者完全一致。

```python
research = engine.open_research_set("study")
alice = research.perspective("alice")
bob = research.perspective("bob")
alice.read_page(10)                     # 只推进 alice 的已读位置
bob.read_page(20)                       # bob 可以按不同节奏阅读
alice.decide("d1", "included", "alice：直接相关")
bob.decide("d1", "excluded", "bob：与研究问题无关")   # 相反决定并存
research.save()
```

`merge_perspectives()` 把各视角的决定合并成分类报告，绝不改写任何原始判断：

- `consensus`：所有参与视角都审过且结论一致；
- `conflict`：视角之间存在真实分歧，报告逐条保留每个视角的决定与理由；
- `unreviewed`：没有任何视角下过决定；
- `partial`：部分视角已决定、其余仍未审；
- `resolved`：分歧已被显式解决。

分歧不靠最后一次写入覆盖：某视角再次写入只改自己的决定。要解决分歧必须调用 `resolve(doc_id, action, reason)`，它生成一条新的解决决定，通过 `references` 引用每个视角的原始判断，原始判断本身保留为证据。

```python
report = research.merge_perspectives()
research.resolve("d1", "excluded", "负责人复核：超出范围")
```

研究者可以一次性撤回自己的多条意见，撤回必须填写原因，并且只会撤回指定视角自己的当前意见；其他研究者意见和负责人裁决都不会被删除。撤回的原意见进入撤回批次归档，当前页面与状态不再把它当作有效排除/纳入依据：

```python
alice = research.perspective("alice")
batch = alice.withdraw_decisions(
    ["d1", "d7"],
    "重新核对来源后发现单位和时间范围判断错误",
)
batch.batch_id
research.save()
```

如果负责人历史裁决的 `references` 引用了该批次中的意见，该裁决不会被删除，而是标记为 `pending_recheck`；裁决动作、原因、证据签名和原始引用都继续保留在当前裁决与 `resolution_history` 中。`merge_perspectives()` 会把这类文档归入 `recheck`，`review_status()` 的 `recheck_resolutions` 也会列出待复核清单。负责人完成复核后可再次调用 `resolve()` 写入新裁决，旧裁决仍作为历史依据保留。

索引更新后，`revise_research_set()`（查询不变时即派生新研究）逐视角继承：内容未变、证据签名仍相符的记录继承原决定（含已读位置，去重且不保留已删除文档）；内容变更的记录进入 `dropped_decisions`（`evidence_basis_changed`），已删除记录标记 `document_absent`，都要求重新审阅。解决决定只有在它引用的全部原判断都仍可用（有效或属于匹配的撤回批次）且证据签名仍相符时才随之继承；引用撤回意见的裁决以 `pending_recheck` 状态继承，无关裁决保持有效。撤回批次会随派生集保存，但影响范围仅限批次内意见和实际引用这些意见的结论。派生后的新研究集可以继续挂视角、再合并。

```bash
python -m local_search.cli --index ./data/index research-page study --page-size 20 --perspective alice
python -m local_search.cli --index ./data/index research-decide study d1 excluded "无关" --perspective bob
python -m local_search.cli --index ./data/index research-merge study
python -m local_search.cli --index ./data/index research-resolve study d1 excluded "负责人复核"
python -m local_search.cli --index ./data/index research-withdraw study '["d1","d7"]' "来源复核后撤回" --perspective alice
```

## 有依赖关系的多批撤回方案

直接撤回（`withdraw_decisions`）是立即生效的单批操作；撤回方案（withdrawal plan）则把一次撤回拆成"起草 → 预览 → 确认/取消"三步，并在起草时推导依赖：

```python
alice = research.perspective("alice")
plan = alice.plan_withdrawal(
    ["d1", "d7"],
    "重新核对来源后撤回部分意见",
    keep=["d3"],          # d3 的已确认裁决必须保持确认
    references=["d9"],    # 只引用、不撤回 d9 的意见
)
plan.report["withdraw"]               # 可行的最小撤回范围
plan.report["recheck_resolutions"]    # 只依赖被撤意见、将转为待复核的裁决
plan.report["supported_resolutions"]  # 仍有其他研究者独立支持、保持不变的裁决
plan.report["unchanged_opinions"]     # 本视角保持不变的意见
plan.report["conflicts"]              # 与 keep 冲突的依据（不自动覆盖裁决）
research.save()                        # 方案随研究集持久化，可重开后续核
research.apply_withdrawal_plan(plan.plan_id)   # 或 research.cancel_withdrawal_plan(...)
```

- **依赖推导**：裁决的 `references` 可带 `doc_id` 形成跨文档引用。方案推导时，只有当其引用的全部意见都被撤回（或已失效）时才转为 `pending_recheck`；只要仍引用其他研究者的有效意见，就视为有独立支持而保持不变。真正撤回原意见时会扫描全部裁决的跨文档引用，不会遗漏。
- **引用 ≠ 撤回**：`references` 列出的意见只作为方案依据被引用，应用方案不会触碰原意见；多个批次引用同一份意见时，撤回其中一个引用不会删除原意见。
- **最小可行范围**：`keep` 指定的已确认裁决必须保持确认。若撤回某意见会使其失去全部支持，该意见被移出撤回范围并记入 `blocked`/`conflicts`（附裁决与引用依据）；完全无法满足时应用阶段直接报冲突中止，绝不自动覆盖裁决。
- **重开续核**：方案保存后可重新打开再 `apply`。应用时逐条重新核对：已撤回的意见跳过（不重复撤回），制定方案后被改写的新意见按 `opinion_changed_since_plan` 跳过（不混入旧方案）；已应用或已取消的方案不能再次应用。

```bash
python -m local_search.cli --index ./data/index research-plan-withdraw study '["d1","d7"]' "来源复核" --perspective alice --keep '["d3"]'
python -m local_search.cli --index ./data/index research-plan-show study plan-1
python -m local_search.cli --index ./data/index research-plan-apply study plan-1
python -m local_search.cli --index ./data/index research-plan-cancel study plan-1
```
