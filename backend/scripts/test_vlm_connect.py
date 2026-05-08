#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, api_key: str, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, api_key: str, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".bmp":
        return "image/bmp"
    if ext in {".tif", ".tiff"}:
        return "image/tiff"
    return "image/png"


def run_models_test(base_url: str, api_key: str) -> None:
    print("[1/3] Testing GET /models ...")
    data = get_json(f"{base_url.rstrip('/')}/models", api_key)
    models = data.get("data", []) if isinstance(data, dict) else []
    model_ids = [item.get("id") for item in models if isinstance(item, dict)]
    print(f"  OK, models count={len(model_ids)}")
    if model_ids:
        print("  model ids:")
        for mid in model_ids[:20]:
            print(f"    - {mid}")


def run_text_test(base_url: str, api_key: str, model: str) -> None:
    print("[2/3] Testing text chat completion ...")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "请只回复：VLM文本连通成功"},
        ],
        "temperature": 0,
    }
    data = post_json(f"{base_url.rstrip('/')}/chat/completions", payload, api_key)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        if isinstance(data, dict)
        else ""
    )
    print("  OK, response:")
    print(f"    {content}")


def run_image_test(base_url: str, api_key: str, model: str, image_path: Path) -> None:
    print("[3/3] Testing image chat completion ...")
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime = guess_mime(image_path)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文简短描述这张图的主要内容。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
    }
    data = post_json(f"{base_url.rstrip('/')}/chat/completions", payload, api_key, timeout=180)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        if isinstance(data, dict)
        else ""
    )
    print("  OK, image response:")
    print(f"    {content}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone VLM connectivity test")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default="http://10.1.80.12:8416/v1")
    parser.add_argument("--model", default="/ai/qwen3.5_9b")
    parser.add_argument("--image", default=None, help="Optional image path for multimodal test")
    args = parser.parse_args()

    print("VLM test config:")
    print(f"  base_url = {args.base_url}")
    print(f"  model    = {args.model}")
    print(f"  api_key  = {args.api_key}")

    try:
        run_models_test(args.base_url, args.api_key)
        run_text_test(args.base_url, args.api_key, args.model)
        if args.image:
            run_image_test(args.base_url, args.api_key, args.model, Path(args.image).expanduser().resolve())
        else:
            print("[3/3] Skip image test (no --image provided)")
        print("\nAll tests passed.")
        return 0
    except urllib.error.URLError as exc:
        print(f"\nConnection failed: {exc}")
        print("Hint: 确认VLM服务已启动，并监听 10.1.80.12:8416")
        return 2
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"\nHTTP error: {exc.code} {detail}")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"\nTest failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
