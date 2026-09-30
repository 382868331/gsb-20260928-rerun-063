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
    search.add_argument("--snapshot", type=int)
    search.add_argument("--no-explain", action="store_true")
    search.add_argument("--expand", action="store_true")
    get = commands.add_parser("get")
    get.add_argument("id")
    commands.add_parser("docs")
    commands.add_parser("snapshots")
    synonyms = commands.add_parser("synonyms")
    synonyms.add_argument("rules_json")

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
        target = engine.snapshot(args.snapshot)
        print(json.dumps(
            target.search(args.query, args.limit, explain=not args.no_explain, expand=args.expand or None),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.command == "get":
        print(json.dumps(engine.get(args.id), ensure_ascii=False, indent=2))
    elif args.command == "docs":
        print(json.dumps(engine.all_documents(), ensure_ascii=False, indent=2))
    elif args.command == "snapshots":
        print(json.dumps(engine.snapshots(), ensure_ascii=False, indent=2))
    elif args.command == "synonyms":
        version = engine.set_synonyms(json.loads(args.rules_json))
        engine.commit()
        print(json.dumps({"rules_version": version}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
