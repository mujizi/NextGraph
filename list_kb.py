
from pymilvus import connections, Collection
import os

def list_kb_ids():
    try:
        connections.connect(alias="default", host="10.1.80.16", port="19530", db_name="crx")
        col = Collection("Entities")
        # Use query with distinct kb_id if possible, but Milvus query doesn't support DISTINCT easily in older versions
        # We'll just fetch a bunch and see the unique ones
        res = col.query(expr="id != ''", output_fields=["kb_id"], limit=1000)
        kb_ids = set(r['kb_id'] for r in res)
        print("--- KNOWLEDGE BASE IDS ---")
        for kb_id in kb_ids:
            print(f"- {kb_id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_kb_ids()
