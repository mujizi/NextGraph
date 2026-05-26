#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pymilvus import Collection, MilvusClient, connections

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import Settings


def _print_json(title: str, payload: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _field_names(collection_name: str) -> list[str]:
    collection = Collection(collection_name)
    return [field.name for field in collection.schema.fields]


def _sample_query(
    *,
    collection_name: str,
    expr: str,
    output_fields: list[str],
    limit: int,
) -> dict[str, Any]:
    collection = Collection(collection_name)
    collection.load()
    result = {
        "collection": collection_name,
        "requested_fields": output_fields,
        "rows": [],
        "error": None,
    }
    try:
        rows = collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit,
        )
        result["rows"] = rows
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Milvus connectivity and collection schema.")
    parser.add_argument("--uri", default=None, help="Milvus URI. Defaults to NEXTGRAPH_MILVUS_URI / settings.")
    parser.add_argument("--db", default=None, help="Milvus database name. Defaults to NEXTGRAPH_MILVUS_DB / settings.")
    parser.add_argument("--token", default=None, help="Optional Milvus token.")
    parser.add_argument("--user-id", default="admin_user", help="User scope for sample queries.")
    parser.add_argument("--kb-id", default="text0515", help="Knowledge base scope for sample queries.")
    parser.add_argument("--limit", type=int, default=3, help="Sample row limit.")
    args = parser.parse_args()

    settings = Settings()
    uri = args.uri or settings.milvus_uri
    db_name = args.db or settings.milvus_db
    token = args.token if args.token is not None else settings.milvus_token

    print("Milvus test config:")
    print(f"  uri     = {uri}")
    print(f"  db      = {db_name}")
    print(f"  user_id = {args.user_id}")
    print(f"  kb_id   = {args.kb_id}")
    print(f"  token   = {'<set>' if token else '<empty>'}")

    try:
        bootstrap_kwargs: dict[str, Any] = {"uri": uri}
        if token:
            bootstrap_kwargs["token"] = token

        print("\n[1/5] Connecting with MilvusClient ...")
        bootstrap = MilvusClient(**bootstrap_kwargs)
        databases = bootstrap.list_databases()
        print(f"  OK, databases: {databases}")

        print("\n[2/5] Opening target database ...")
        client = MilvusClient(uri=uri, db_name=db_name, token=token) if token else MilvusClient(uri=uri, db_name=db_name)
        collections = client.list_collections()
        print(f"  OK, collections in {db_name}: {collections}")

        print("\n[3/5] Connecting with ORM API ...")
        connections.connect(alias="default", uri=uri, db_name=db_name, token=token)
        print("  OK, ORM connection established")

        expected = {
            "entities": settings.entities_collection,
            "relations": settings.relations_collection,
            "passages": settings.passages_collection,
        }
        schema_report: dict[str, Any] = {}
        print("\n[4/5] Inspecting collection schema ...")
        for label, collection_name in expected.items():
            item: dict[str, Any] = {
                "collection": collection_name,
                "exists": collection_name in collections,
                "fields": [],
                "has_docment_id": False,
            }
            if item["exists"]:
                fields = _field_names(collection_name)
                item["fields"] = fields
                item["has_docment_id"] = "docment_id" in fields
            schema_report[label] = item
        _print_json("Schema report", schema_report)

        print("\n[5/5] Running sample scoped queries ...")
        expr = f'user_id == "{args.user_id}" and kb_id == "{args.kb_id}"'
        query_report = {
            "expr": expr,
            "entities": _sample_query(
                collection_name=settings.entities_collection,
                expr=expr,
                output_fields=["id", "name", "relation_ids"],
                limit=args.limit,
            ),
            "relations": _sample_query(
                collection_name=settings.relations_collection,
                expr=expr,
                output_fields=["id", "subject_id", "object_id", "relation", "docment_id"],
                limit=args.limit,
            ),
            "passages": _sample_query(
                collection_name=settings.passages_collection,
                expr=expr,
                output_fields=["id", "passage", "docment_id"],
                limit=args.limit,
            ),
        }
        _print_json("Query report", query_report)

        relation_docment_ok = schema_report["relations"]["has_docment_id"]
        passage_docment_ok = schema_report["passages"]["has_docment_id"]
        if not relation_docment_ok or not passage_docment_ok:
            print("\nWARNING:")
            print("  Current Milvus schema is missing `docment_id` in one or more collections.")
            print("  This matches the error you saw in the backend logs.")

        print("\nMilvus connectivity test finished.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nMilvus test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
