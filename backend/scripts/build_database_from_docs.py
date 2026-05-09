#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.parse_file.api import DEFAULT_OUTPUT_ROOT  # noqa: E402
from app.parse_file.mineru_service import SUPPORTED_EXTENSIONS, TASK_MANAGER  # noqa: E402


FINISHED_PARSE_STATUSES = {"completed", "failed", "partial_failed"}


def _read_file_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _collect_input_files(args: argparse.Namespace) -> list[str]:
    files = list(args.files or [])
    if args.file_list:
        files.extend(_read_file_list(Path(args.file_list).expanduser()))
    for raw_folder in args.folders or []:
        folder = Path(raw_folder).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"文件夹不存在: {folder}")
        files.extend(
            str(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in files:
        resolved = str(Path(raw_path).expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _find_textified_jsons(parse_result: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for item in parse_result.get("file_results", []):
        enhanced_raw = item.get("enhanced_json_path")
        if not enhanced_raw:
            continue
        enhanced_path = Path(enhanced_raw)
        if enhanced_path.name.endswith("_textified.json") and enhanced_path.exists():
            candidates.append(enhanced_path)
        textified_path = enhanced_path.with_name(enhanced_path.stem + "_textified.json")
        if textified_path.exists():
            candidates.append(textified_path)

    output_dir = parse_result.get("output_dir")
    if output_dir:
        candidates.extend(Path(output_dir).rglob("*_textified.json"))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _derive_docment_id(json_path: Path, index: int, prefix: str | None) -> str:
    stem = json_path.stem
    for suffix in ("_content_list_enhanced_textified", "_enhanced_textified", "_textified"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    if not safe_stem:
        safe_stem = f"doc_{index + 1:04d}"
    return f"{prefix}_{safe_stem}" if prefix else safe_stem


def run_parse_stage(args: argparse.Namespace) -> dict[str, Any]:
    files = _collect_input_files(args)
    if not files:
        raise SystemExit("请通过 --files 或 --file-list 提供待解析文档。")

    task = TASK_MANAGER.submit(
        file_paths=files,
        output_root=args.output_root,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        method=args.method,
        lang=args.lang,
        backend=args.backend,
        start_page=args.start_page,
        end_page=args.end_page,
        formula=not args.no_formula,
        table=not args.no_table,
        source=args.source,
        vlm_url=args.vlm_url,
    )
    print(f"[parse] task_id={task.task_id} total={task.total} output_dir={task.output_dir}")

    while True:
        current = TASK_MANAGER.get(task.task_id)
        if current is None:
            raise RuntimeError(f"解析任务丢失: {task.task_id}")
        file_progress = ", ".join(
            f"{Path(file_path).name}:{progress:.1f}%"
            for file_path, progress in sorted(current.per_file_progress.items())
        )
        print(
            "[parse] "
            f"status={current.status} completed={current.completed}/{current.total} "
            f"success={current.success} failed={current.failed} "
            f"progress={current.progress_percent}%"
            + (f" files=[{file_progress}]" if file_progress else "")
        )
        if current.status in FINISHED_PARSE_STATUSES:
            return current.to_dict()
        time.sleep(args.poll_interval)


def load_existing_parse_outputs(args: argparse.Namespace) -> dict[str, Any]:
    json_paths = [Path(path).expanduser().resolve() for path in args.parsed_json]
    missing = [str(path) for path in json_paths if not path.exists()]
    if missing:
        raise SystemExit(f"已解析 JSON 不存在: {missing}")
    return {
        "task_id": None,
        "status": "completed",
        "output_dir": None,
        "file_results": [{"enhanced_json_path": str(path)} for path in json_paths],
    }


async def run_vectorization_stage(args: argparse.Namespace, json_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        from app.vectorization_pipeline.api import (  # type: ignore
            VectorizationRequest,
            process_vectorization_task,
            tasks_db,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "找不到 app.vectorization_pipeline。请先把向量化/三元组抽取/数据库模块合入 crx，"
            "再运行 --run-vectorization。"
        ) from exc

    summaries: list[dict[str, Any]] = []
    for index, json_path in enumerate(json_paths):
        docment_id = args.docment_id or _derive_docment_id(json_path, index, args.docment_id_prefix)
        task_id = uuid.uuid4().hex
        tasks_db[task_id] = {
            "id": task_id,
            "status": "pending",
            "step": "queued",
            "msg": "脚本任务已入队",
            "progress": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        request = VectorizationRequest(
            user_id=args.user_id,
            kb_id=args.kb_id,
            docment_id=docment_id,
            json_path=str(json_path),
            max_concurrency=args.vector_concurrency,
            max_chunk_chars=args.max_chunk_chars,
            embedding_model=args.embedding_model,
        )

        print(f"[vectorize] task_id={task_id} json={json_path} docment_id={docment_id}")
        runner = asyncio.create_task(process_vectorization_task(task_id, request))
        last_seen: tuple[str | None, str | None, float | None] = (None, None, None)
        while not runner.done():
            state = tasks_db.get(task_id, {})
            current = (
                state.get("step"),
                state.get("msg"),
                state.get("progress"),
            )
            if current != last_seen:
                progress = state.get("progress", 0)
                print(
                    "[vectorize] "
                    f"step={state.get('step')} progress={progress:.0%} "
                    f"msg={state.get('msg')}"
                )
                last_seen = current
            await asyncio.sleep(args.vector_poll_interval)
        await runner
        result = dict(tasks_db.get(task_id, {}))
        print(f"[vectorize] step={result.get('step')} msg={result.get('msg')}")
        summaries.append(
            {
                "task_id": task_id,
                "json_path": str(json_path),
                "docment_id": docment_id,
                "status": result,
            }
        )
    return summaries


def write_manifest(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse documents with MinerU, then optionally vectorize and store graph data."
    )
    parser.add_argument("--files", nargs="*", default=[], help="待解析文档路径，可传多个。")
    parser.add_argument("--folders", nargs="*", default=[], help="递归处理文件夹中的支持文件类型，可传多个。")
    parser.add_argument("--file-list", help="文本文件，每行一个待解析文档路径。")
    parser.add_argument("--parsed-json", nargs="*", default=[], help="跳过解析，直接处理已有 *_textified.json。")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="MinerU 输出根目录。")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="MinerU 使用的 GPU ID。")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--lang")
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--no-formula", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--vlm-url")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--run-vectorization", action="store_true", help="调用 vectorization_pipeline 写入数据库。")
    parser.add_argument("--user-id", default="admin_user")
    parser.add_argument("--kb-id", required=True, help="目标知识库 ID。")
    parser.add_argument("--docment-id", help="单文档固定 docment_id；多文档建议使用 --docment-id-prefix。")
    parser.add_argument("--docment-id-prefix", help="自动生成 docment_id 时添加前缀。")
    parser.add_argument("--vector-concurrency", type=int, default=10)
    parser.add_argument("--vector-poll-interval", type=float, default=1.0)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--manifest",
        default=str(BACKEND_ROOT / "storage" / "database_build_manifest.json"),
        help="流水线结果清单输出路径。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.docment_id and args.docment_id_prefix:
        raise SystemExit("--docment-id 和 --docment-id-prefix 只能二选一。")
    if args.docment_id and len(args.parsed_json or args.files or []) > 1:
        raise SystemExit("多文档处理时请使用 --docment-id-prefix，不要使用单个 --docment-id。")

    parse_result = load_existing_parse_outputs(args) if args.parsed_json else run_parse_stage(args)
    textified_jsons = _find_textified_jsons(parse_result)
    if not textified_jsons:
        raise SystemExit("没有找到 *_textified.json，无法进入后续处理。")

    manifest: dict[str, Any] = {
        "parse": parse_result,
        "textified_jsons": [str(path) for path in textified_jsons],
        "vectorization": [],
    }
    print(f"[pipeline] textified_jsons={len(textified_jsons)}")

    if args.run_vectorization:
        manifest["vectorization"] = asyncio.run(run_vectorization_stage(args, textified_jsons))
    else:
        print("[pipeline] 已完成解析阶段；如需入库，合入向量化模块后追加 --run-vectorization。")

    write_manifest(args, manifest)


if __name__ == "__main__":
    main()
