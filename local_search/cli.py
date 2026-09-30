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


if __name__ == "__main__":
    main()
