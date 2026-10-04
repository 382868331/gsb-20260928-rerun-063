import argparse
import json
from typing import Sequence

from .engine import SearchEngine


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-search")
    parser.add_argument("--index", default=".search_index")
    commands = parser.add_subparsers(dest="command", required=True)

    upsert = commands.add_parser("upsert")
    upsert.add_argument("id")
    upsert.add_argument("fields_json")
    commands.add_parser("commit")
    delete = commands.add_parser("delete")
    delete.add_argument("id")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int)
    get = commands.add_parser("get")
    get.add_argument("id")
    commands.add_parser("docs")
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("name")
    snapshot_search = commands.add_parser("snapshot-search")
    snapshot_search.add_argument("name")
    snapshot_search.add_argument("query")
    snapshot_search.add_argument("--limit", type=int)

    research_new = commands.add_parser("research-new")
    research_new.add_argument("name")
    research_new.add_argument("queries_json", help='{"label": "query text", ...}')
    research_new.add_argument("--mode", choices=["any", "all"], default="any")
    research_new.add_argument("--budget", type=int)
    research_page = commands.add_parser("research-page")
    research_page.add_argument("name")
    research_page.add_argument("--offset", type=int, default=0)
    research_page.add_argument("--page-size", type=int)
    research_page.add_argument("--perspective")
    research_status = commands.add_parser("research-status")
    research_status.add_argument("name")
    research_status.add_argument("--perspective")
    research_decide = commands.add_parser("research-decide")
    research_decide.add_argument("name")
    research_decide.add_argument("doc_id")
    research_decide.add_argument("action", choices=["included", "excluded"])
    research_decide.add_argument("reason")
    research_decide.add_argument("--perspective")
    research_withdraw = commands.add_parser("research-withdraw")
    research_withdraw.add_argument("name")
    research_withdraw.add_argument("doc_ids_json", help='["doc-id-1", "doc-id-2"]')
    research_withdraw.add_argument("reason")
    research_withdraw.add_argument("--perspective", required=True)
    research_withdraw.add_argument("--batch-id")
    research_merge = commands.add_parser("research-merge")
    research_merge.add_argument("name")
    research_merge.add_argument("--perspective", action="append", dest="perspectives")
    research_resolve = commands.add_parser("research-resolve")
    research_resolve.add_argument("name")
    research_resolve.add_argument("doc_id")
    research_resolve.add_argument("action", choices=["included", "excluded"])
    research_resolve.add_argument("reason")
    research_continue = commands.add_parser("research-continue")
    research_continue.add_argument("name")
    research_continue.add_argument("--budget", type=int)
    research_revise = commands.add_parser("research-revise")
    research_revise.add_argument("name")
    research_revise.add_argument("queries_json", nargs="?")
    research_revise.add_argument("--mode", choices=["any", "all"])
    research_revise.add_argument("--budget", type=int)
    research_revise.add_argument("--new-name")

    args = parser.parse_args(argv)
    engine = SearchEngine(args.index)

    if args.command == "upsert":
        engine.upsert(args.id, json.loads(args.fields_json))
        engine.commit()
    elif args.command == "delete":
        engine.delete(args.id)
        engine.commit()
    elif args.command == "commit":
        engine.commit()
    elif args.command == "search":
        print(json.dumps(engine.search(args.query, args.limit), ensure_ascii=False, indent=2))
    elif args.command == "get":
        print(json.dumps(engine.get(args.id), ensure_ascii=False, indent=2))
    elif args.command == "docs":
        print(json.dumps(engine.all_documents(), ensure_ascii=False, indent=2))
    elif args.command == "snapshot":
        print(json.dumps(engine.publish_snapshot(args.name), ensure_ascii=False, indent=2))
    elif args.command == "snapshot-search":
        print(json.dumps(engine.search_snapshot(args.name, args.query, args.limit), ensure_ascii=False, indent=2))
    elif args.command == "research-new":
        research = engine.create_research_set(
            args.name, json.loads(args.queries_json), mode=args.mode, budget=args.budget
        )
        print(json.dumps(research.review_status(), ensure_ascii=False, indent=2))
    elif args.command == "research-page":
        research = engine.open_research_set(args.name)
        if args.perspective:
            research.add_perspective(args.perspective)
        print(json.dumps(research.page(args.offset, args.page_size, args.perspective), ensure_ascii=False, indent=2))
    elif args.command == "research-status":
        research = engine.open_research_set(args.name)
        if args.perspective:
            research.add_perspective(args.perspective)
        print(json.dumps(research.review_status(args.perspective), ensure_ascii=False, indent=2))
    elif args.command == "research-decide":
        research = engine.open_research_set(args.name)
        if args.perspective:
            research.add_perspective(args.perspective)
        research.decide(args.doc_id, args.action, args.reason, args.perspective)
        research.save()
        print(json.dumps(research.review_status(args.perspective), ensure_ascii=False, indent=2))
    elif args.command == "research-withdraw":
        research = engine.open_research_set(args.name)
        research.add_perspective(args.perspective)
        batch = research.withdraw_decisions(
            json.loads(args.doc_ids_json),
            args.reason,
            args.perspective,
            batch_id=args.batch_id,
        )
        research.save()
        print(json.dumps({
            "batch_id": batch.batch_id,
            "perspective": batch.perspective,
            "reason": batch.reason,
            "doc_ids": sorted(batch.decisions),
            "status": research.review_status(args.perspective),
            "merge": research.merge_perspectives(),
        }, ensure_ascii=False, indent=2))
    elif args.command == "research-merge":
        research = engine.open_research_set(args.name)
        print(json.dumps(research.merge_perspectives(args.perspectives), ensure_ascii=False, indent=2))
    elif args.command == "research-resolve":
        research = engine.open_research_set(args.name)
        research.resolve(args.doc_id, args.action, args.reason)
        research.save()
        print(json.dumps(research.merge_perspectives(), ensure_ascii=False, indent=2))
    elif args.command == "research-continue":
        research = engine.open_research_set(args.name)
        research.continue_scan(args.budget)
        research.save()
        print(json.dumps(research.review_status(), ensure_ascii=False, indent=2))
    elif args.command == "research-revise":
        queries = None if args.queries_json is None else json.loads(args.queries_json)
        report = engine.revise_research_set(
            args.name,
            queries,
            mode=args.mode,
            budget=args.budget,
            new_name=args.new_name,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
