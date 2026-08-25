#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Any

import requests

# =========================
# 直接修改这里的参数即可
# =========================
API_BASE_URL = "http://127.0.0.1:5190"
API_PATH = "/api/search/external/query"

QUESTION = "主角和谁有冲突？"
USER_ID = "255"
KB_ID = "剧本22"

MODE = "hybrid"  # 可选: hybrid / semantic / triple
INCLUDE_ANSWER = True

TOP_K = 5
ENTITY_TOP_K = 8
RELATION_TOP_K = 8
EXPANSION_DEGREE = 2
ANSWER_TOP_K = 3
PASSAGE_TOP_K = 3

REQUEST_TIMEOUT_SECONDS = 120


def build_payload() -> dict[str, Any]:
    return {
        "question": QUESTION,
        "user_id": USER_ID,
        "kb_id": KB_ID,
        "mode": MODE,
        "include_answer": INCLUDE_ANSWER,
        "top_k": TOP_K,
        "entity_top_k": ENTITY_TOP_K,
        "relation_top_k": RELATION_TOP_K,
        "expansion_degree": EXPANSION_DEGREE,
        "answer_top_k": ANSWER_TOP_K,
        "passage_top_k": PASSAGE_TOP_K,
    }


def main() -> int:
    url = f"{API_BASE_URL.rstrip('/')}{API_PATH}"
    payload = build_payload()

    print("Request URL:")
    print(url)
    print("\nRequest Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print(f"\nRequest failed: {exc}")
        return 1

    print(f"\nHTTP Status: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        print("\nResponse is not valid JSON:")
        print(response.text)
        return 1

    print("\nResponse JSON:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if not response.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
