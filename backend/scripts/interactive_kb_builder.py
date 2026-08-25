#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_SOURCE_DIR = BACKEND_ROOT / "storage" / "255"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "storage" / "mineru_output"
DEFAULT_RUN_ROOT = BACKEND_ROOT / "storage" / "kb_build_runs"
FINISHED_PARSE_STATUSES = {"completed", "failed", "partial_failed"}

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parse_file.mineru_service import SUPPORTED_EXTENSIONS, TASK_MANAGER  # noqa: E402
from app.vectorization_pipeline.api import (  # noqa: E402
    VectorizationRequest,
    process_vectorization_task,
    tasks_db,
)


@dataclass
class BuildConfig:
    source_dir: str
    output_root: str
    run_root: str
    kb_id: str
    user_id: str
    milvus_db: str
    gpus: list[str]
    workers_per_gpu: int
    method: str
    backend: str
    lang: str | None
    start_page: int | None
    end_page: int | None
    formula: bool
    table: bool
    source: str | None
    vlm_url: str | None
    run_vectorization: bool
    extract_concurrency: int
    embedding_concurrency: int
    vector_poll_interval: float
    parse_poll_interval: float
    max_chunk_chars: int
    chunk_overlap_chars: int
    embedding_model: str
    docment_id_prefix: str | None
    resume_run_dir: str | None
    vectorize_existing_only: bool


def _prompt_text(label: str, default: str | None = None, *, allow_empty: bool = True) -> str | None:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if allow_empty:
            return None
        print("该项不能为空，请重新输入。")


def _prompt_int(label: str, default: int | None = None, *, allow_empty: bool = True) -> int | None:
    while True:
        raw = _prompt_text(label, None if default is None else str(default), allow_empty=allow_empty)
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except ValueError:
            print("请输入整数。")


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = _prompt_text(label, str(default), allow_empty=False)
        assert raw is not None
        try:
            return float(raw)
        except ValueError:
            print("请输入数字。")


def _prompt_bool(label: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1"}:
            return True
        if raw in {"n", "no", "0"}:
            return False
        print("请输入 y 或 n。")


def _prompt_choice(label: str, choices: list[str], default: str) -> str:
    options = "/".join(choices)
    while True:
        raw = _prompt_text(f"{label} ({options})", default, allow_empty=False)
        assert raw is not None
        if raw in choices:
            return raw
        print(f"请输入以下选项之一: {options}")


def _prompt_gpus(default: list[str]) -> list[str]:
    raw = _prompt_text("GPU 列表，逗号分隔", ",".join(default), allow_empty=False)
    assert raw is not None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_match_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _collect_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"目录不存在: {source_dir}")
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _derive_textified_json(enhanced_json_path: str | None) -> Path | None:
    if not enhanced_json_path:
        return None
    enhanced_path = Path(enhanced_json_path)
    candidates = [
        enhanced_path if enhanced_path.name.endswith("_textified.json") else None,
        enhanced_path.with_name(enhanced_path.stem + "_textified.json"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


def _safe_docment_id(kb_id: str, path: Path, index: int, prefix: str | None = None) -> str:
    stem = path.stem or f"doc_{index + 1:04d}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    safe_prefix = (safe or f"doc_{index + 1:04d}")[:24].strip("_") or f"doc_{index + 1:04d}"
    digest = hashlib.md5(f"{kb_id}:{path}:{index}".encode("utf-8")).hexdigest()[:8]
    base = f"{kb_id}_{safe_prefix}_{digest}"
    return f"{prefix}_{base}" if prefix else base


def _latest_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(
        [path for path in run_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_manifest_incomplete(manifest: dict[str, Any], total_files: int) -> bool:
    files = manifest.get("files") or []
    if not manifest.get("finished_at"):
        return True
    if len(files) < total_files:
        return True
    return any((item or {}).get("overall_status") != "success" for item in files)


def _merge_items_by_file_path(*item_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for items in item_groups:
        for item in items or []:
            file_path = str(item.get("file_path") or "").strip()
            if not file_path:
                continue
            if file_path not in merged:
                ordered_paths.append(file_path)
                merged[file_path] = dict(item)
            else:
                merged[file_path].update(item)
    return [merged[path] for path in ordered_paths]


def _merge_parse_result(base: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    if incoming:
        for key, value in incoming.items():
            if key not in {"file_results", "errors"}:
                merged[key] = value
    merged["file_results"] = _merge_items_by_file_path(
        list((base or {}).get("file_results") or []),
        list((incoming or {}).get("file_results") or []),
    )
    merged["errors"] = _merge_items_by_file_path(
        list((base or {}).get("errors") or []),
        list((incoming or {}).get("errors") or []),
    )
    return merged


def _load_previous_file_records(run_dir: Path) -> list[dict[str, Any]]:
    manifest = _read_json_if_exists(run_dir / "manifest.json") or {}
    records = list(manifest.get("files") or [])
    records_by_path: dict[str, dict[str, Any]] = {}
    for item in records:
        file_path = str(item.get("file_path") or "").strip()
        if file_path:
            records_by_path[file_path] = item

    jsonl_path = run_dir / "file_results.jsonl"
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            file_path = str(item.get("file_path") or "").strip()
            if file_path:
                records_by_path[file_path] = item
    return list(records_by_path.values())


def _discover_textified_jsons(
    *,
    files: list[Path],
    output_root: Path,
    started_at: str | None,
) -> dict[str, str]:
    if not output_root.exists():
        return {}

    started_ts = None
    if started_at:
        try:
            started_ts = datetime.fromisoformat(started_at).timestamp()
        except ValueError:
            started_ts = None

    candidates = []
    for path in output_root.rglob("*_textified.json"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if started_ts is not None and stat.st_mtime + 1 < started_ts:
            continue
        candidates.append(path.resolve())

    normalized_candidates = [
        (
            path,
            _normalize_match_key(str(path)),
            path.stat().st_mtime,
        )
        for path in candidates
    ]

    discovered: dict[str, str] = {}
    for file_path in files:
        source_key = _normalize_match_key(file_path.stem)
        if not source_key:
            continue
        matches = [
            (path, mtime)
            for path, normalized, mtime in normalized_candidates
            if source_key in normalized
        ]
        if not matches:
            continue
        matches.sort(key=lambda item: item[1], reverse=True)
        discovered[str(file_path)] = str(matches[0][0])
    return discovered


def _find_resume_manifest(config: BuildConfig, total_files: int) -> Path | None:
    if config.resume_run_dir:
        manifest_path = Path(config.resume_run_dir).expanduser().resolve() / "manifest.json"
        return manifest_path if manifest_path.exists() else None

    run_root = Path(config.run_root).expanduser().resolve()
    for run_dir in _latest_run_dirs(run_root):
        manifest = _read_json_if_exists(run_dir / "manifest.json")
        if not manifest:
            continue
        previous_config = manifest.get("config") or {}
        if str(previous_config.get("source_dir") or "") != config.source_dir:
            continue
        if str(previous_config.get("kb_id") or "") != config.kb_id:
            continue
        if str(previous_config.get("user_id") or "") != config.user_id:
            continue
        if str(previous_config.get("milvus_db") or "") != config.milvus_db:
            continue
        if not _is_manifest_incomplete(manifest, total_files):
            continue
        return run_dir / "manifest.json"
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"interactive_kb_builder.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式知识库构建脚本：批量解析目录文件，并按现有逻辑完成向量化入库。"
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="待处理文档目录。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="MinerU 输出根目录。")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="日志与结果输出根目录。")
    parser.add_argument("--kb-id",default="0616", help="知识库 ID。")
    parser.add_argument("--user-id", default="admin_user")
    parser.add_argument("--milvus-db", default="bookk255")
    parser.add_argument("--gpus", default="0", help="逗号分隔，例如 0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--lang")
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--formula", dest="formula", action="store_true")
    parser.add_argument("--no-formula", dest="formula", action="store_false")
    parser.set_defaults(formula=True)
    parser.add_argument("--table", dest="table", action="store_true")
    parser.add_argument("--no-table", dest="table", action="store_false")
    parser.set_defaults(table=True)
    parser.add_argument("--source")
    parser.add_argument("--vlm-url")
    parser.add_argument("--run-vectorization", dest="run_vectorization", action="store_true")
    parser.add_argument("--skip-vectorization", dest="run_vectorization", action="store_false")
    parser.set_defaults(run_vectorization=True)
    parser.add_argument("--extract-concurrency", type=int, default=14)
    parser.add_argument("--embedding-concurrency", type=int, default=10)
    parser.add_argument("--vector-poll-interval", type=float, default=1.0)
    parser.add_argument("--parse-poll-interval", type=float, default=5.0)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--chunk-overlap-chars", type=int, default=150)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--docment-id-prefix")
    parser.add_argument("--resume-run-dir", help="指定某次 kb_build_runs 目录，从该次运行继续。")
    parser.add_argument(
        "--vectorize-existing-only",
        action="store_true",
        help="跳过仍未解析成功的文件，只处理已经存在 textified.json 的文件。",
    )
    parser.add_argument("--no-interactive", action="store_true", help="直接使用命令行参数运行，不进入交互。")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> BuildConfig:
    interactive = not args.no_interactive and sys.stdin.isatty()

    source_dir = args.source_dir
    output_root = args.output_root
    run_root = args.run_root
    kb_id = args.kb_id
    user_id = args.user_id
    milvus_db = args.milvus_db
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    workers_per_gpu = args.workers_per_gpu
    method = args.method
    backend = args.backend
    lang = args.lang
    start_page = args.start_page
    end_page = args.end_page
    formula = args.formula
    table = args.table
    source = args.source
    vlm_url = args.vlm_url
    run_vectorization = args.run_vectorization
    extract_concurrency = args.extract_concurrency
    embedding_concurrency = args.embedding_concurrency
    vector_poll_interval = args.vector_poll_interval
    parse_poll_interval = args.parse_poll_interval
    max_chunk_chars = args.max_chunk_chars
    chunk_overlap_chars = args.chunk_overlap_chars
    embedding_model = args.embedding_model
    docment_id_prefix = args.docment_id_prefix
    resume_run_dir = args.resume_run_dir
    vectorize_existing_only = bool(args.vectorize_existing_only)

    if interactive:
        print("请按提示确认运行参数，直接回车可使用默认值。")
        source_dir = _prompt_text("文档目录", source_dir, allow_empty=False) or source_dir
        output_root = _prompt_text("解析输出根目录", output_root, allow_empty=False) or output_root
        run_root = _prompt_text("日志输出根目录", run_root, allow_empty=False) or run_root
        kb_id = _prompt_text("知识库 ID", kb_id or "255", allow_empty=False)
        user_id = _prompt_text("user_id", user_id, allow_empty=False) or user_id
        milvus_db = _prompt_text("向量库 database 名称", milvus_db, allow_empty=False) or milvus_db
        gpus = _prompt_gpus(gpus or ["0"])
        workers_per_gpu = _prompt_int("每张 GPU worker 数", workers_per_gpu, allow_empty=False) or 1
        method = _prompt_choice("MinerU 解析模式", ["auto", "txt", "ocr"], method)
        backend = _prompt_text("MinerU backend", backend, allow_empty=False) or backend
        lang = _normalize_optional_text(_prompt_text("语言（留空表示自动）", lang))
        start_page = _prompt_int("起始页（留空表示从头开始）", start_page)
        end_page = _prompt_int("结束页（留空表示到末尾）", end_page)
        formula = _prompt_bool("是否启用公式识别", formula)
        table = _prompt_bool("是否启用表格识别", table)
        source = _normalize_optional_text(_prompt_text("MinerU source（留空沿用默认）", source))
        vlm_url = _normalize_optional_text(_prompt_text("VLM URL（留空沿用默认）", vlm_url))
        run_vectorization = _prompt_bool("解析完成后是否继续向量化入库", run_vectorization)
        if run_vectorization:
            extract_concurrency = _prompt_int("三元组抽取并发", extract_concurrency, allow_empty=False) or 10
            embedding_concurrency = _prompt_int("Embedding 并发", embedding_concurrency, allow_empty=False) or 10
            max_chunk_chars = _prompt_int("分块最大字符数", max_chunk_chars, allow_empty=False) or 1500
            chunk_overlap_chars = _prompt_int("分块重叠字符数", chunk_overlap_chars, allow_empty=False) or 150
            embedding_model = _prompt_text("Embedding 模型", embedding_model, allow_empty=False) or embedding_model
            docment_id_prefix = _normalize_optional_text(
                _prompt_text("docment_id 前缀（留空自动生成）", docment_id_prefix)
            )
            vector_poll_interval = _prompt_float("向量化轮询间隔（秒）", vector_poll_interval)
        parse_poll_interval = _prompt_float("解析轮询间隔（秒）", parse_poll_interval)
        resume_run_dir = _normalize_optional_text(
            _prompt_text("续跑目录（留空自动检测最近未完成 run）", resume_run_dir)
        )
        vectorize_existing_only = _prompt_bool("是否只处理已解析成功文件，跳过剩余解析失败项", vectorize_existing_only)
    else:
        kb_id = kb_id or "255"

    if not kb_id:
        raise SystemExit("必须提供 kb_id。")
    if start_page is not None and end_page is not None and start_page > end_page:
        raise SystemExit("start_page 不能大于 end_page。")
    if chunk_overlap_chars >= max_chunk_chars:
        raise SystemExit("chunk_overlap_chars 必须小于 max_chunk_chars。")

    return BuildConfig(
        source_dir=source_dir,
        output_root=output_root,
        run_root=run_root,
        kb_id=kb_id,
        user_id=user_id,
        milvus_db=milvus_db,
        gpus=gpus or ["0"],
        workers_per_gpu=workers_per_gpu,
        method=method,
        backend=backend,
        lang=_normalize_optional_text(lang),
        start_page=start_page,
        end_page=end_page,
        formula=formula,
        table=table,
        source=_normalize_optional_text(source),
        vlm_url=_normalize_optional_text(vlm_url),
        run_vectorization=run_vectorization,
        extract_concurrency=extract_concurrency,
        embedding_concurrency=embedding_concurrency,
        vector_poll_interval=vector_poll_interval,
        parse_poll_interval=parse_poll_interval,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        embedding_model=embedding_model,
        docment_id_prefix=_normalize_optional_text(docment_id_prefix),
        resume_run_dir=_normalize_optional_text(resume_run_dir),
        vectorize_existing_only=vectorize_existing_only,
    )


def _run_parse_stage(
    config: BuildConfig,
    files: list[Path],
    logger: logging.Logger,
    *,
    existing_parse: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    merged_parse = dict(existing_parse or {})
    if not files:
        if merged_parse:
            merged_parse["status"] = "completed"
        return merged_parse

    task = TASK_MANAGER.submit(
        file_paths=[str(path) for path in files],
        output_root=config.output_root,
        gpus=config.gpus,
        workers_per_gpu=config.workers_per_gpu,
        method=config.method,
        lang=config.lang,
        backend=config.backend,
        start_page=config.start_page,
        end_page=config.end_page,
        formula=config.formula,
        table=config.table,
        source=config.source,
        vlm_url=config.vlm_url,
    )
    logger.info("解析任务已提交: task_id=%s total=%s output_dir=%s", task.task_id, task.total, task.output_dir)

    last_status: tuple[str | None, int | None, int | None, int | None] = (None, None, None, None)
    while True:
        current = TASK_MANAGER.get(task.task_id)
        if current is None:
            raise RuntimeError(f"解析任务丢失: {task.task_id}")
        current_payload = current.to_dict()
        merged_parse = _merge_parse_result(merged_parse, current_payload)
        if manifest is not None and manifest_path is not None:
            manifest["parse"] = merged_parse
            _write_json(manifest_path, manifest)
        snapshot = (current.status, current.completed, current.success, current.failed)
        if snapshot != last_status:
            logger.info(
                "解析进度: status=%s completed=%s/%s success=%s failed=%s progress=%s%%",
                current.status,
                current.completed,
                current.total,
                current.success,
                current.failed,
                current.progress_percent,
            )
            last_status = snapshot
        if current.status in FINISHED_PARSE_STATUSES:
            return merged_parse
        time.sleep(config.parse_poll_interval)


async def _run_vectorization_for_file(
    *,
    config: BuildConfig,
    file_path: Path,
    json_path: Path,
    index: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    docment_id = _safe_docment_id(config.kb_id, file_path, index, config.docment_id_prefix)
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
        user_id=config.user_id,
        kb_id=config.kb_id,
        milvus_db=config.milvus_db,
        docment_id=docment_id,
        json_path=str(json_path),
        extract_concurrency=config.extract_concurrency,
        embedding_concurrency=config.embedding_concurrency,
        max_chunk_chars=config.max_chunk_chars,
        chunk_overlap_chars=config.chunk_overlap_chars,
        embedding_model=config.embedding_model,
    )

    logger.info("开始向量化: file=%s docment_id=%s task_id=%s", file_path.name, docment_id, task_id)
    started = time.time()
    runner = asyncio.create_task(process_vectorization_task(task_id, request))
    last_seen: tuple[str | None, str | None, Any] = (None, None, None)

    while not runner.done():
        state = tasks_db.get(task_id, {})
        current = (state.get("step"), state.get("msg"), state.get("progress"))
        if current != last_seen:
            logger.info(
                "向量化进度: file=%s step=%s progress=%s msg=%s",
                file_path.name,
                state.get("step"),
                state.get("progress"),
                state.get("msg"),
            )
            last_seen = current
        await asyncio.sleep(config.vector_poll_interval)

    await runner
    state = dict(tasks_db.get(task_id, {}))
    duration = round(time.time() - started, 2)
    step = state.get("step")
    success = step == "completed"
    logger.info(
        "向量化结束: file=%s success=%s duration=%.2fs msg=%s",
        file_path.name,
        success,
        duration,
        state.get("msg"),
    )
    return {
        "task_id": task_id,
        "docment_id": docment_id,
        "duration_seconds": duration,
        "step": step,
        "message": state.get("msg"),
        "progress": state.get("progress"),
        "summary": state.get("summary"),
        "raw_status": state,
        "success": success,
    }


async def _run_pipeline(config: BuildConfig) -> int:
    source_dir = Path(config.source_dir).expanduser().resolve()
    files = _collect_files(source_dir)
    if not files:
        raise SystemExit(f"目录下没有可处理文件: {source_dir}")

    resume_manifest_path = _find_resume_manifest(config, len(files))
    previous_manifest = _read_json_if_exists(resume_manifest_path) if resume_manifest_path else None
    if resume_manifest_path and previous_manifest:
        run_dir = resume_manifest_path.parent
    else:
        run_dir = Path(config.run_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(run_dir)

    manifest_path = run_dir / "manifest.json"
    file_results_path = run_dir / "file_results.jsonl"
    success_list_path = run_dir / "success_files.txt"
    failed_list_path = run_dir / "failed_files.txt"

    logger.info("开始执行知识库构建。")
    logger.info("项目目录: %s", PROJECT_ROOT)
    logger.info("源目录: %s", source_dir)
    logger.info("共发现文件: %s", len(files))
    logger.info("运行目录: %s", run_dir)

    previous_records = _load_previous_file_records(run_dir)
    previous_records_by_path = {
        str(item.get("file_path") or "").strip(): item
        for item in previous_records
        if str(item.get("file_path") or "").strip()
    }
    previous_parse = dict(previous_manifest.get("parse") or {}) if previous_manifest else {}
    previous_parse_results = {
        str(item.get("file_path") or "").strip(): item
        for item in previous_parse.get("file_results") or []
        if str(item.get("file_path") or "").strip()
    }

    if resume_manifest_path and previous_manifest:
        logger.info("检测到未完成运行，准备续跑: %s", run_dir)
    discovered_textified = _discover_textified_jsons(
        files=files,
        output_root=Path(config.output_root).expanduser().resolve(),
        started_at=(previous_manifest or {}).get("started_at"),
    )
    for file_path, json_path in discovered_textified.items():
        previous_parse_results.setdefault(
            file_path,
            {
                "file_path": file_path,
                "enhanced_json_path": json_path,
            },
        )

    resumed_count = sum(1 for item in previous_records_by_path.values() if item.get("overall_status") == "success")
    parse_reuse_count = len(previous_parse_results)
    logger.info(
        "续跑预检查: 已完成向量化 %s 个，可复用解析结果 %s 个。",
        resumed_count,
        parse_reuse_count,
    )

    manifest: dict[str, Any] = previous_manifest or {
        "started_at": datetime.now().isoformat(),
        "config": asdict(config),
        "source_dir": str(source_dir),
        "total_files": len(files),
        "parse": {},
        "files": [],
    }
    manifest["config"] = asdict(config)
    manifest["source_dir"] = str(source_dir)
    manifest["total_files"] = len(files)
    manifest.setdefault("started_at", datetime.now().isoformat())
    manifest["resumed_from_existing_run"] = bool(resume_manifest_path and previous_manifest)
    manifest.pop("finished_at", None)
    manifest.pop("summary", None)
    manifest["files"] = previous_records
    _write_json(manifest_path, manifest)

    parse_pending_files = [
        path
        for path in files
        if str(path) not in previous_parse_results
        and previous_records_by_path.get(str(path), {}).get("overall_status") != "success"
    ]
    if config.vectorize_existing_only:
        logger.info("已启用仅向量化已解析文件模式，剩余 %s 个未解析文件将被跳过。", len(parse_pending_files))
        parse_pending_files = []

    if parse_pending_files:
        logger.info("仍需解析文件 %s 个。", len(parse_pending_files))
    else:
        logger.info("所有可复用解析结果已就绪，跳过解析阶段。")

    parse_result = _run_parse_stage(
        config,
        parse_pending_files,
        logger,
        existing_parse=_merge_parse_result(
            previous_parse,
            {"file_results": list(previous_parse_results.values())},
        ),
        manifest=manifest,
        manifest_path=manifest_path,
    )
    manifest["parse"] = parse_result
    _write_json(manifest_path, manifest)

    parse_success_map = {
        str(item["file_path"]): item
        for item in parse_result.get("file_results", [])
        if item.get("file_path")
    }
    parse_error_map = {
        str(item["file_path"]): item
        for item in parse_result.get("errors", [])
        if item.get("file_path")
    }

    success_files: list[str] = []
    failed_files: list[str] = []

    for index, file_path in enumerate(files):
        file_key = str(file_path)
        previous_record = previous_records_by_path.get(file_key)
        if previous_record and previous_record.get("overall_status") == "success":
            logger.info("跳过已完成文件: %s", file_path.name)
            manifest["files"] = [
                item for item in manifest["files"] if str(item.get("file_path") or "").strip() != file_key
            ] + [previous_record]
            success_files.append(file_key)
            _write_json(manifest_path, manifest)
            continue

        parse_item = parse_success_map.get(file_key)
        error_item = parse_error_map.get(file_key)

        record: dict[str, Any] = {
            "file_path": file_key,
            "file_name": file_path.name,
            "parse_status": "pending",
            "vector_status": "skipped" if config.run_vectorization else "not_requested",
            "overall_status": "failed",
            "parse": None,
            "vector": None,
            "error": None,
        }

        if parse_item is None:
            record["parse_status"] = "failed"
            record["error"] = (error_item or {}).get("error", "解析失败，但未找到详细错误。")
            failed_files.append(file_key)
        else:
            record["parse_status"] = "success"
            textified_json = _derive_textified_json(parse_item.get("enhanced_json_path"))
            record["parse"] = {
                "duration_seconds": parse_item.get("duration_seconds"),
                "enhanced_json_path": parse_item.get("enhanced_json_path"),
                "textified_json_path": str(textified_json) if textified_json else None,
                "enhanced_count": parse_item.get("enhanced_count"),
                "enhance_failed": parse_item.get("enhance_failed"),
            }

            if not config.run_vectorization:
                record["overall_status"] = "success"
                success_files.append(file_key)
            elif textified_json is None:
                record["vector_status"] = "failed"
                record["error"] = "解析成功，但未找到 *_textified.json。"
                failed_files.append(file_key)
            else:
                vector_result = await _run_vectorization_for_file(
                    config=config,
                    file_path=file_path,
                    json_path=textified_json,
                    index=index,
                    logger=logger,
                )
                record["vector"] = vector_result
                record["vector_status"] = "success" if vector_result["success"] else "failed"
                if vector_result["success"]:
                    record["overall_status"] = "success"
                    success_files.append(file_key)
                else:
                    record["error"] = vector_result.get("message") or "向量化失败。"
                    failed_files.append(file_key)

        manifest["files"] = [
            item for item in manifest["files"] if str(item.get("file_path") or "").strip() != file_key
        ] + [record]
        _append_jsonl(file_results_path, record)
        _write_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["summary"] = {
        "success_count": len(success_files),
        "failed_count": len(failed_files),
        "success_files_path": str(success_list_path),
        "failed_files_path": str(failed_list_path),
        "file_results_path": str(file_results_path),
        "run_log_path": str(run_dir / "run.log"),
    }
    _write_json(manifest_path, manifest)
    _write_lines(success_list_path, success_files)
    _write_lines(failed_list_path, failed_files)

    logger.info("执行完成: success=%s failed=%s", len(success_files), len(failed_files))
    logger.info("结果清单: %s", manifest_path)
    logger.info("成功文件列表: %s", success_list_path)
    logger.info("失败文件列表: %s", failed_list_path)
    return 0 if not failed_files else 1


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    raise SystemExit(asyncio.run(_run_pipeline(config)))


if __name__ == "__main__":
    main()
