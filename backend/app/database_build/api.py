from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.parse_file.api import DEFAULT_OUTPUT_ROOT
from backend.app.parse_file.mineru_service import TASK_MANAGER
from backend.app.vector_database.providers.factory import get_vector_store
from backend.app.vectorization_pipeline.api import (
    VectorizationRequest,
    process_vectorization_task,
    tasks_db as vector_tasks_db,
)

router = APIRouter(prefix="/database_build", tags=["database_build"])

TASK_STORE = (
    Path(__file__).resolve().parent.parent.parent / "storage" / "database_build_tasks.json"
).resolve()
LIBRARY_FILE_STORE = (
    Path(__file__).resolve().parent.parent.parent / "storage" / "database_build_library_files.json"
).resolve()
FINISHED_PARSE_STATUSES = {"completed", "failed", "partial_failed"}
FINISHED_BUILD_STATUSES = {"completed", "failed", "partial_failed"}

_tasks_lock = threading.Lock()
_tasks: dict[str, dict[str, Any]] = {}
_library_files: dict[str, dict[str, Any]] = {}


class DatabaseBuildRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
    kb_id: str = Field(..., min_length=1)
    user_id: str = Field("admin_user", min_length=1)
    milvus_db: str = Field("crx", min_length=1)
    output_root: str = Field(DEFAULT_OUTPUT_ROOT)
    gpus: list[str] = Field(default_factory=lambda: ["0"], min_length=1)
    workers_per_gpu: int = Field(1, ge=1, le=16)
    method: str = Field("auto", pattern="^(auto|txt|ocr)$")
    lang: str | None = None
    backend: str = "pipeline"
    start_page: int | None = None
    end_page: int | None = None
    formula: bool = True
    table: bool = True
    source: str | None = None
    vlm_url: str | None = None
    vector_concurrency: int | None = Field(default=None, ge=1, le=64)
    extract_concurrency: int = Field(10, ge=1, le=64)
    embedding_concurrency: int = Field(10, ge=1, le=64)
    max_chunk_chars: int = Field(1500, ge=200, le=12000)
    chunk_overlap_chars: int = Field(150, ge=0, le=4000)
    embedding_model: str = "text-embedding-3-small"

    @field_validator("gpus")
    @classmethod
    def normalize_gpus(cls, value: list[str]) -> list[str]:
        gpu_ids = [str(item).strip() for item in value if str(item).strip()]
        if not gpu_ids:
            raise ValueError("gpus 不能为空")
        return gpu_ids


class DatabaseBuildResumeRequest(BaseModel):
    source_task_id: str = Field(..., min_length=1)
    file_paths: list[str] = Field(default_factory=list)
    user_id: str = Field("admin_user", min_length=1)
    milvus_db: str = Field("crx", min_length=1)
    output_root: str = Field(DEFAULT_OUTPUT_ROOT)
    gpus: list[str] = Field(default_factory=lambda: ["0"], min_length=1)
    workers_per_gpu: int = Field(1, ge=1, le=16)
    method: str = Field("auto", pattern="^(auto|txt|ocr)$")
    lang: str | None = None
    backend: str = "pipeline"
    start_page: int | None = None
    end_page: int | None = None
    formula: bool = True
    table: bool = True
    source: str | None = None
    vlm_url: str | None = None
    vector_concurrency: int | None = Field(default=None, ge=1, le=64)
    extract_concurrency: int = Field(10, ge=1, le=64)
    embedding_concurrency: int = Field(10, ge=1, le=64)
    max_chunk_chars: int = Field(1500, ge=200, le=12000)
    chunk_overlap_chars: int = Field(150, ge=0, le=4000)
    embedding_model: str = "text-embedding-3-small"
    reuse_existing_json: bool = True

    @field_validator("gpus")
    @classmethod
    def normalize_resume_gpus(cls, value: list[str]) -> list[str]:
        gpu_ids = [str(item).strip() for item in value if str(item).strip()]
        if not gpu_ids:
            raise ValueError("gpus 不能为空")
        return gpu_ids


def _load_tasks() -> None:
    global _tasks
    if not TASK_STORE.exists():
        return
    try:
        data = json.loads(TASK_STORE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(data, dict):
        _tasks = data


def _load_library_files() -> None:
    global _library_files
    if not LIBRARY_FILE_STORE.exists():
        return
    try:
        data = json.loads(LIBRARY_FILE_STORE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(data, dict):
        _library_files = data


def _save_tasks() -> None:
    TASK_STORE.parent.mkdir(parents=True, exist_ok=True)
    TASK_STORE.write_text(json.dumps(_tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_library_files() -> None:
    LIBRARY_FILE_STORE.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_FILE_STORE.write_text(json.dumps(_library_files, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> float:
    return time.time()


def _library_scope_key(*, kb_id: str, user_id: str, milvus_db: str) -> str:
    return f"{milvus_db}::{user_id}::{kb_id}"


def _safe_docment_id(kb_id: str, path: str, index: int) -> str:
    stem = Path(path).stem or f"doc_{index + 1:04d}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    prefix = (safe or f"doc_{index + 1:04d}")[:24].strip("_") or f"doc_{index + 1:04d}"
    digest = hashlib.md5(f"{kb_id}:{path}:{index}".encode("utf-8")).hexdigest()[:8]
    return f"{kb_id}_{prefix}_{digest}"


def _textified_json_path(file_result: dict[str, Any]) -> str | None:
    enhanced_raw = file_result.get("enhanced_json_path")
    if not enhanced_raw:
        return None
    enhanced_path = Path(enhanced_raw)
    candidates = [
        enhanced_path if enhanced_path.name.endswith("_textified.json") else None,
        enhanced_path.with_name(enhanced_path.stem + "_textified.json"),
        enhanced_path,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def _extract_plain_text(file_path: str) -> str:
    source_path = Path(file_path)
    if source_path.suffix.lower() not in {".txt", ".md"} or not source_path.exists():
        return ""
    raw_text = source_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw_text:
        return ""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
        return parsed[0].strip()
    if isinstance(parsed, dict):
        for key in ("text", "content", "markdown"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return raw_text


def _is_plain_text_source(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {".txt", ".md"}


def _json_has_text_items(json_path: str) -> bool:
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    items = data.get("items", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return False
    return any(isinstance(item, dict) and str(item.get("text", "")).strip() for item in items)


def _ensure_vector_json(json_path: str | None, file_path: str) -> str | None:
    if json_path and _json_has_text_items(json_path):
        return json_path

    text = _extract_plain_text(file_path)
    if not text:
        return json_path

    fallback_dir = Path(json_path).parent if json_path else TASK_STORE.parent / "fallback_text"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / f"{Path(file_path).stem}_source_textified.json"
    fallback_payload = {
        "total_items": 1,
        "items": [
            {
                "id": 1,
                "type": "text",
                "text": text,
                "bbox": [],
                "page_idx": -1,
            }
        ],
    }
    fallback_path.write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(fallback_path)


def _vector_json_from_plain_text(file_path: str, task_id: str) -> str | None:
    text = _extract_plain_text(file_path)
    if not text:
        return None
    fallback_dir = TASK_STORE.parent / "source_text" / task_id
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / f"{Path(file_path).stem}_textified.json"
    fallback_payload = {
        "total_items": 1,
        "items": [
            {
                "id": 1,
                "type": "text",
                "text": text,
                "bbox": [],
                "page_idx": -1,
            }
        ],
    }
    fallback_path.write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(fallback_path)


def _ensure_library_scope(*, kb_id: str, user_id: str, milvus_db: str) -> dict[str, Any]:
    scope_key = _library_scope_key(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db)
    scope = _library_files.setdefault(
        scope_key,
        {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": milvus_db,
            "updated_at": 0,
            "files": {},
        },
    )
    scope.setdefault("files", {})
    scope["updated_at"] = _now()
    return scope


def _sync_library_file_entry(task: dict[str, Any], file_item: dict[str, Any]) -> None:
    kb_id = str(task.get("kb_id") or "").strip()
    user_id = str(task.get("user_id") or "").strip()
    milvus_db = str(task.get("milvus_db") or "crx").strip() or "crx"
    file_path = str(file_item.get("path") or "").strip()
    if not kb_id or not user_id or not file_path:
        return

    scope = _ensure_library_scope(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db)
    scope["files"][file_path] = {
        "name": file_item.get("name") or Path(file_path).name,
        "path": file_path,
        "stage": file_item.get("stage"),
        "state": file_item.get("state"),
        "progress": file_item.get("progress", 0),
        "parse_progress": file_item.get("parse_progress", 0),
        "vector_progress": file_item.get("vector_progress", 0),
        "summary": file_item.get("summary") or {},
        "json_path": file_item.get("json_path"),
        "resume_action": file_item.get("resume_action"),
        "last_task_id": task.get("task_id"),
        "updated_at": _now(),
    }


def _sync_task_files_to_library(task: dict[str, Any]) -> None:
    for file_item in task.get("files") or []:
        _sync_library_file_entry(task, file_item)


def _rebuild_library_files_from_tasks() -> None:
    _library_files.clear()
    for task in sorted(_tasks.values(), key=lambda item: float(item.get("updated_at", 0)), reverse=True):
        kb_id = str(task.get("kb_id") or "").strip()
        user_id = str(task.get("user_id") or "").strip()
        milvus_db = str(task.get("milvus_db") or "crx").strip() or "crx"
        if not kb_id or not user_id:
            continue
        scope = _ensure_library_scope(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db)
        for file_item in reversed(task.get("files") or []):
            file_path = str(file_item.get("path") or "").strip()
            if not file_path or file_path in scope["files"]:
                continue
            _sync_library_file_entry(task, file_item)


def _library_summary_from_registry(*, kb_id: str, user_id: str, milvus_db: str) -> dict[str, Any]:
    scope = _library_files.get(_library_scope_key(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db), {})
    synthetic_task = {"files": list((scope.get("files") or {}).values())}
    return _task_summary(synthetic_task)


def _set_task(task_id: str, **updates: Any) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        task.update(updates)
        task["updated_at"] = _now()
        task["progress_percent"] = _task_progress(task)
        _save_tasks()
        _save_library_files()


def _update_file(task_id: str, file_path: str, **updates: Any) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        for item in task["files"]:
            if item["path"] == file_path:
                item.update(updates)
                _sync_library_file_entry(task, item)
                break
        task["updated_at"] = _now()
        task["progress_percent"] = _task_progress(task)
        _save_tasks()
        _save_library_files()


def _task_progress(task: dict[str, Any]) -> float:
    files = task.get("files") or []
    if not files:
        return 0.0
    return round(sum(float(item.get("progress", 0)) for item in files) / len(files), 2)


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(task, ensure_ascii=False))


def _get_vector_store(request: Request):
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        store = get_vector_store(request.app.state.settings)
        request.app.state.vector_store = store
    return store


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    files = task.get("files") or []
    chunk_count = 0
    entity_count = 0
    relation_count = 0
    failed_embeddings = 0
    source_documents: list[dict[str, Any]] = []
    for item in files:
        summary = item.get("summary") or {}
        chunk_count += int(summary.get("chunks") or 0)
        entity_count += int(summary.get("entities") or 0)
        relation_count += int(summary.get("relations") or 0)
        failed_embeddings += int(summary.get("failed_embeddings") or 0)
        source_documents.append(
            {
                "name": item.get("name") or Path(str(item.get("path", ""))).name,
                "path": item.get("path"),
                "stage": item.get("stage"),
                "state": item.get("state"),
                "progress": item.get("progress", 0),
                "parse_progress": item.get("parse_progress", 0),
                "vector_progress": item.get("vector_progress", 0),
                "summary": summary,
            }
        )

    completed = sum(1 for item in files if item.get("stage") == "completed")
    failed = sum(1 for item in files if item.get("stage") == "failed")
    return {
        "file_count": len(files),
        "completed": completed,
        "failed": failed,
        "chunk_count": chunk_count,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "failed_embeddings": failed_embeddings,
        "completion_ratio": _safe_ratio(completed, len(files)),
        "source_documents": source_documents,
    }


def _mark_task_interrupted(task: dict[str, Any], reason: str = "后端已重启，任务被中断，可继续构建。") -> bool:
    if str(task.get("status") or "") in FINISHED_BUILD_STATUSES:
        return False

    files = task.get("files") or []
    completed = 0
    failed = 0
    changed = False
    for item in files:
        stage = str(item.get("stage") or "")
        if stage == "completed":
            completed += 1
            continue
        if item.get("stage") != "failed" or item.get("state") != reason:
            item["stage"] = "failed"
            item["state"] = reason
            changed = True
        failed += 1

    next_status = "partial_failed" if completed > 0 else "failed"
    if (
        task.get("status") != next_status
        or task.get("stage") != next_status
        or task.get("message") != reason
        or task.get("completed") != completed
        or task.get("failed") != failed
    ):
        task["status"] = next_status
        task["stage"] = next_status
        task["message"] = reason
        task["completed"] = completed
        task["failed"] = failed
        task["updated_at"] = _now()
        changed = True
    return changed


def _reconcile_interrupted_tasks() -> bool:
    changed = False
    for task in _tasks.values():
        if _mark_task_interrupted(task):
            changed = True
    if changed:
        _rebuild_library_files_from_tasks()
        _save_tasks()
        _save_library_files()
    return changed


def _aggregate_library_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if not tasks:
        return _task_summary({"files": []})
    latest_task = max(tasks, key=lambda item: float(item.get("updated_at", 0)))
    return _library_summary_from_registry(
        kb_id=str(latest_task.get("kb_id") or "").strip(),
        user_id=str(latest_task.get("user_id") or "").strip(),
        milvus_db=str(latest_task.get("milvus_db") or "crx").strip() or "crx",
    )


def _build_task_payload_from_resume(
    source_task: dict[str, Any],
    resume_request: DatabaseBuildResumeRequest,
) -> DatabaseBuildRequest:
    source_file_paths = [
        str(item.get("path") or "").strip()
        for item in source_task.get("files") or []
        if str(item.get("path") or "").strip()
    ]
    if not source_file_paths:
        raise HTTPException(status_code=400, detail="源任务没有可续跑的文件")

    requested_paths = list(dict.fromkeys(path.strip() for path in (resume_request.file_paths or []) if path.strip()))
    if requested_paths:
        valid_paths = set(source_file_paths)
        invalid_paths = [path for path in requested_paths if path not in valid_paths]
        if invalid_paths:
            raise HTTPException(status_code=400, detail=f"续跑文件不属于源任务: {invalid_paths[0]}")
        file_paths = requested_paths
    else:
        file_paths = source_file_paths

    return DatabaseBuildRequest(
        file_paths=file_paths,
        kb_id=str(source_task.get("kb_id") or "").strip(),
        user_id=resume_request.user_id,
        milvus_db=resume_request.milvus_db,
        output_root=resume_request.output_root,
        gpus=resume_request.gpus,
        workers_per_gpu=resume_request.workers_per_gpu,
        method=resume_request.method,
        lang=resume_request.lang,
        backend=resume_request.backend,
        start_page=resume_request.start_page,
        end_page=resume_request.end_page,
        formula=resume_request.formula,
        table=resume_request.table,
        source=resume_request.source,
        vlm_url=resume_request.vlm_url,
        vector_concurrency=resume_request.vector_concurrency,
        extract_concurrency=resume_request.extract_concurrency,
        embedding_concurrency=resume_request.embedding_concurrency,
        max_chunk_chars=resume_request.max_chunk_chars,
        chunk_overlap_chars=resume_request.chunk_overlap_chars,
        embedding_model=resume_request.embedding_model,
    )


def _clone_resume_file_entry(
    source_file: dict[str, Any],
    *,
    source_task_id: str,
    reuse_existing_json: bool,
) -> dict[str, Any]:
    file_path = str(source_file.get("path") or "").strip()
    file_name = source_file.get("name") or Path(file_path).name
    stage = str(source_file.get("stage") or "queued")
    json_path = str(source_file.get("json_path") or "").strip()
    json_available = bool(json_path and Path(json_path).exists() and _json_has_text_items(json_path))
    summary = source_file.get("summary") or {}

    entry = {
        "path": file_path,
        "name": file_name,
        "stage": "queued",
        "state": "等待续跑",
        "progress": 0,
        "parse_progress": 0,
        "vector_progress": 0,
        "resume_source_task_id": source_task_id,
        "resume_source_stage": stage,
    }

    if stage == "completed":
        entry.update(
            {
                "stage": "completed",
                "state": "沿用上次成功结果",
                "progress": 100,
                "parse_progress": 100,
                "vector_progress": 100,
                "summary": summary,
                "resume_action": "reused_completed",
            }
        )
        if json_available:
            entry["json_path"] = json_path
        return entry

    if reuse_existing_json and json_available:
        entry.update(
            {
                "stage": "resume_ready",
                "state": "复用上次解析结果，等待继续向量化",
                "progress": 45,
                "parse_progress": 100,
                "json_path": json_path,
                "resume_action": "reuse_json",
            }
        )
    else:
        entry["resume_action"] = "full_retry"
    return entry


def _create_task_record(
    task_id: str,
    payload: DatabaseBuildRequest,
    *,
    source_task: dict[str, Any] | None = None,
    reuse_existing_json: bool = True,
) -> dict[str, Any]:
    now = _now()
    if source_task is None:
        files = [
            {
                "path": path,
                "name": Path(path).name,
                "stage": "queued",
                "state": "等待中",
                "progress": 0,
                "parse_progress": 0,
                "vector_progress": 0,
            }
            for path in payload.file_paths
        ]
        message = "任务已入队"
    else:
        source_task_id = str(source_task.get("task_id") or "").strip()
        selected_paths = set(payload.file_paths)
        files = [
            _clone_resume_file_entry(
                item,
                source_task_id=source_task_id,
                reuse_existing_json=reuse_existing_json,
            )
            for item in source_task.get("files") or []
            if str(item.get("path") or "").strip() in selected_paths
        ]
        message = "续跑任务已入队"

    completed = sum(1 for item in files if item.get("stage") == "completed")
    failed = sum(1 for item in files if item.get("stage") == "failed")
    return {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": message,
        "progress_percent": round(sum(float(item.get("progress", 0)) for item in files) / len(files), 2) if files else 0,
        "kb_id": payload.kb_id,
        "user_id": payload.user_id,
        "milvus_db": payload.milvus_db,
        "parse_task_id": None,
        "created_at": now,
        "updated_at": now,
        "completed": completed,
        "failed": failed,
        "files": files,
        "resume_from_task_id": source_task.get("task_id") if source_task else None,
        "resume_selected_file_count": len(files) if source_task else None,
        "resume_source_file_count": len(source_task.get("files") or []) if source_task else None,
    }


def _query_collection_preview(
    *,
    request: Request,
    kb_id: str,
    user_id: str,
    milvus_db: str,
    limit: int,
) -> dict[str, Any]:
    selected_db = milvus_db or request.app.state.settings.vector_db_database
    return _get_vector_store(request).query_collection_preview(
        kb_id=kb_id,
        user_id=user_id,
        db_name=selected_db,
        limit=limit,
    )


def _discover_kb_ids_from_milvus(
    *,
    request: Request,
    user_id: str,
    milvus_db: str,
    sample_limit: int = 20000,
) -> set[str]:
    selected_db = milvus_db or request.app.state.settings.vector_db_database
    return _get_vector_store(request).discover_kb_ids(
        user_id=user_id,
        db_name=selected_db,
        sample_limit=sample_limit,
    )


@router.get("/catalog/debug")
def debug_library_catalog(
    request: Request,
    milvus_db: str = "crx",
    user_id: str = "admin_user",
    preview_limit: int = 5,
) -> dict[str, Any]:
    discovered_kb_ids = sorted(
        _discover_kb_ids_from_milvus(
            request=request,
            user_id=user_id,
            milvus_db=milvus_db,
        )
    )

    libraries: list[dict[str, Any]] = []
    for kb_id in discovered_kb_ids:
        preview = _query_collection_preview(
            request=request,
            kb_id=kb_id,
            user_id=user_id,
            milvus_db=milvus_db,
            limit=preview_limit,
        )
        libraries.append(
            {
                "kb_id": kb_id,
                "counts": preview["counts"],
                "sample_entity_names": [item.get("name") or item.get("id") for item in preview["samples"]["entities"]],
                "sample_relation_labels": [item.get("relation") or item.get("id") for item in preview["samples"]["relations"]],
            }
        )

    return {
        "milvus_db": milvus_db,
        "user_id": user_id,
        "kb_ids": discovered_kb_ids,
        "libraries": libraries,
    }


@router.get("/catalog/debug/raw")
def debug_library_catalog_raw(
    request: Request,
    milvus_db: str = "crx",
    sample_limit: int = 10,
) -> dict[str, Any]:
    selected_db = milvus_db or request.app.state.settings.vector_db_database
    kb_ids = sorted(
        _get_vector_store(request).discover_kb_ids(
            user_id="admin_user",
            db_name=selected_db,
            sample_limit=max(sample_limit, 100),
        )
    )
    return {"milvus_db": selected_db, "kb_ids": kb_ids}


async def _vectorize_with_progress(
    *,
    build_task_id: str,
    file_path: str,
    vector_task_id: str,
    request: VectorizationRequest,
) -> None:
    vector_tasks_db[vector_task_id] = {
        "id": vector_task_id,
        "status": "pending",
        "step": "queued",
        "msg": "任务已入队",
        "progress": 0,
        "created_at": _now(),
        "updated_at": _now(),
    }
    runner = asyncio.create_task(process_vectorization_task(vector_task_id, request))
    while not runner.done():
        state = vector_tasks_db.get(vector_task_id, {})
        vector_progress = float(state.get("progress", 0) or 0)
        _update_file(
            build_task_id,
            file_path,
            stage="vectorizing",
            state=state.get("msg", "向量化中"),
            vector_task_id=vector_task_id,
            vector_progress=round(vector_progress * 100, 2),
            progress=round(45 + vector_progress * 55, 2),
        )
        await asyncio.sleep(1)
    await runner
    state = vector_tasks_db.get(vector_task_id, {})
    if state.get("step") == "completed":
        _update_file(
            build_task_id,
            file_path,
            stage="completed",
            state="已写入向量数据库",
            vector_progress=100,
            progress=100,
            summary=state.get("summary", {}),
        )
        return
    raise RuntimeError(str(state.get("msg") or "向量化失败"))


def _run_build_task(task_id: str, payload: DatabaseBuildRequest) -> None:
    try:
        _set_task(task_id, status="running", stage="parsing", message="正在解析文档...")
        text_json_by_file: dict[str, str] = {}
        task_snapshot = _tasks.get(task_id, {})
        existing_file_states = {
            str(item.get("path") or "").strip(): item
            for item in task_snapshot.get("files") or []
            if str(item.get("path") or "").strip()
        }

        for file_path, file_state in existing_file_states.items():
            if file_state.get("stage") == "completed":
                continue
            json_path = str(file_state.get("json_path") or "").strip()
            if json_path and Path(json_path).exists() and _json_has_text_items(json_path):
                text_json_by_file[file_path] = json_path
                _update_file(
                    task_id,
                    file_path,
                    stage="text_ready",
                    state="复用上次解析结果，准备抽取三元组",
                    parse_progress=100,
                    progress=max(45, float(file_state.get("progress", 0) or 0)),
                    json_path=json_path,
                )

        pending_paths = [
            path
            for path in payload.file_paths
            if existing_file_states.get(path, {}).get("stage") != "completed"
        ]
        mineru_files = [
            path
            for path in pending_paths
            if not _is_plain_text_source(path) and path not in text_json_by_file
        ]

        for file_path in pending_paths:
            if not _is_plain_text_source(file_path):
                continue
            if file_path in text_json_by_file:
                continue
            json_path = _vector_json_from_plain_text(file_path, task_id)
            if json_path:
                text_json_by_file[file_path] = json_path
                _update_file(
                    task_id,
                    file_path,
                    stage="text_ready",
                    state="文本文件已读取，准备抽取三元组",
                    parse_progress=100,
                    progress=45,
                    json_path=json_path,
                )
            else:
                _update_file(task_id, file_path, stage="failed", state="文本文件为空或无法读取", progress=0)

        result_by_file: dict[str, Any] = {}
        errors_by_file: dict[str, Any] = {}

        if mineru_files:
            parse_task = TASK_MANAGER.submit(
                file_paths=mineru_files,
                output_root=payload.output_root,
                gpus=payload.gpus,
                workers_per_gpu=payload.workers_per_gpu,
                method=payload.method,
                lang=payload.lang,
                backend=payload.backend,
                start_page=payload.start_page,
                end_page=payload.end_page,
                formula=payload.formula,
                table=payload.table,
                source=payload.source,
                vlm_url=payload.vlm_url,
            )
            _set_task(task_id, parse_task_id=parse_task.task_id)

            while True:
                current = TASK_MANAGER.get(parse_task.task_id)
                if current is None:
                    raise RuntimeError("解析任务丢失")
                parse_data = current.to_dict()
                for file_path, progress in parse_data.get("per_file_progress", {}).items():
                    _update_file(
                        task_id,
                        file_path,
                        stage="parsing",
                        state=f"解析 {round(progress)}%",
                        parse_progress=round(progress, 2),
                        progress=round(float(progress) * 0.45, 2),
                    )
                _set_task(
                    task_id,
                    message=(
                        f"解析中：{parse_data['completed']}/{parse_data['total']} 个文档完成"
                        if parse_data["status"] not in FINISHED_PARSE_STATUSES
                        else "解析完成，准备向量化入库..."
                    ),
                )
                if parse_data["status"] in FINISHED_PARSE_STATUSES:
                    break
                time.sleep(1)

            result_by_file = {
                item.get("file_path"): item for item in parse_data.get("file_results", [])
            }
            errors_by_file = {item.get("file_path"): item for item in parse_data.get("errors", [])}
        else:
            _set_task(task_id, message="文本文件已读取，准备向量化入库...")

        for index, file_path in enumerate(payload.file_paths):
            existing_state = existing_file_states.get(file_path, {})
            if existing_state.get("stage") == "completed":
                continue
            if _is_plain_text_source(file_path) and file_path not in text_json_by_file:
                continue
            if file_path in errors_by_file:
                _update_file(
                    task_id,
                    file_path,
                    stage="failed",
                    state=errors_by_file[file_path].get("error", "解析失败"),
                    progress=45,
                )
                continue
            json_path = text_json_by_file.get(file_path) or _ensure_vector_json(
                _textified_json_path(result_by_file.get(file_path, {})),
                file_path,
            )
            if not json_path:
                _update_file(task_id, file_path, stage="failed", state="未找到解析 JSON", progress=45)
                continue
            current_stage = str(existing_state.get("stage") or "")
            _update_file(
                task_id,
                file_path,
                stage="vectorizing",
                state="继续抽取三元组和向量化" if current_stage == "text_ready" else "开始抽取三元组和向量化",
                json_path=json_path,
                progress=45,
            )
            vector_request = VectorizationRequest(
                user_id=payload.user_id,
                kb_id=payload.kb_id,
                milvus_db=payload.milvus_db,
                docment_id=_safe_docment_id(payload.kb_id, file_path, index),
                json_path=json_path,
                max_concurrency=payload.vector_concurrency,
                extract_concurrency=payload.extract_concurrency,
                embedding_concurrency=payload.embedding_concurrency,
                max_chunk_chars=payload.max_chunk_chars,
                chunk_overlap_chars=payload.chunk_overlap_chars,
                embedding_model=payload.embedding_model,
            )
            vector_task_id = uuid.uuid4().hex
            try:
                asyncio.run(
                    _vectorize_with_progress(
                        build_task_id=task_id,
                        file_path=file_path,
                        vector_task_id=vector_task_id,
                        request=vector_request,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                _update_file(task_id, file_path, stage="failed", state=str(exc), progress=45)

        with _tasks_lock:
            task = _tasks[task_id]
            failed = sum(1 for item in task["files"] if item.get("stage") == "failed")
            completed = sum(1 for item in task["files"] if item.get("stage") == "completed")
        status = "completed" if failed == 0 else ("failed" if completed == 0 else "partial_failed")
        _set_task(
            task_id,
            status=status,
            stage=status,
            message="构建完成，数据已写入向量数据库。" if status == "completed" else "构建结束，但存在失败文档。",
            completed=completed,
            failed=failed,
        )
    except Exception as exc:  # noqa: BLE001
        _set_task(task_id, status="failed", stage="failed", message=f"构建任务失败: {exc}")


@router.post("/submit")
def submit_database_build(payload: DatabaseBuildRequest) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    task = _create_task_record(task_id, payload)
    with _tasks_lock:
        _tasks[task_id] = task
        _sync_task_files_to_library(task)
        _save_tasks()
        _save_library_files()

    runner = threading.Thread(target=_run_build_task, args=(task_id, payload), daemon=True)
    runner.start()
    return {"task_id": task_id, "status": task["status"], "progress_percent": 0}


@router.post("/resume")
def resume_database_build(payload: DatabaseBuildResumeRequest) -> dict[str, Any]:
    source_task = _tasks.get(payload.source_task_id)
    if source_task is None:
        raise HTTPException(status_code=404, detail=f"源任务不存在: {payload.source_task_id}")
    if source_task.get("status") not in FINISHED_BUILD_STATUSES:
        raise HTTPException(status_code=409, detail="源任务仍在执行中，暂时不能续跑")

    resume_build_request = _build_task_payload_from_resume(source_task, payload)
    task_id = uuid.uuid4().hex
    task = _create_task_record(
        task_id,
        resume_build_request,
        source_task=source_task,
        reuse_existing_json=payload.reuse_existing_json,
    )
    with _tasks_lock:
        _tasks[task_id] = task
        _sync_task_files_to_library(task)
        _save_tasks()
        _save_library_files()

    runner = threading.Thread(target=_run_build_task, args=(task_id, resume_build_request), daemon=True)
    runner.start()
    return {"task_id": task_id, "status": task["status"], "progress_percent": task["progress_percent"]}


@router.get("/progress/{task_id}")
def get_database_build_progress(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _public_task(task)


@router.get("/latest")
def get_latest_database_build(
    kb_id: str | None = None,
    user_id: str | None = None,
    milvus_db: str | None = None,
) -> dict[str, Any]:
    candidates = list(_tasks.values())
    if kb_id:
        candidates = [task for task in candidates if task.get("kb_id") == kb_id]
    if user_id:
        candidates = [task for task in candidates if task.get("user_id") == user_id]
    if milvus_db:
        candidates = [task for task in candidates if task.get("milvus_db") == milvus_db]
    if not candidates:
        raise HTTPException(status_code=404, detail="暂无构建任务")
    task = max(candidates, key=lambda item: float(item.get("updated_at", 0)))
    return _public_task(task)


@router.get("/libraries")
def list_built_libraries(kb_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    candidates = list(_tasks.values())
    if kb_id:
        candidates = [task for task in candidates if task.get("kb_id") == kb_id]
    if user_id:
        candidates = [task for task in candidates if task.get("user_id") == user_id]

    items = []
    for task in sorted(candidates, key=lambda item: float(item.get("updated_at", 0)), reverse=True):
        files = task.get("files") or []
        items.append(
            {
                "task_id": task.get("task_id"),
                "kb_id": task.get("kb_id"),
                "user_id": task.get("user_id"),
                "milvus_db": task.get("milvus_db", "crx"),
                "status": task.get("status"),
                "stage": task.get("stage"),
                "message": task.get("message"),
                "progress_percent": task.get("progress_percent", 0),
                "file_count": len(files),
                "completed": sum(1 for item in files if item.get("stage") == "completed"),
                "failed": sum(1 for item in files if item.get("stage") == "failed"),
                "updated_at": task.get("updated_at"),
            }
        )
    return {"libraries": items}


@router.get("/catalog")
def list_library_catalog(
    request: Request,
    milvus_db: str = "crx",
    user_id: str = "admin_user",
    include_live_preview: bool = True,
    preview_limit: int = 12,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in sorted(_tasks.values(), key=lambda item: float(item.get("updated_at", 0)), reverse=True):
        if task.get("milvus_db") != milvus_db or task.get("user_id") != user_id:
            continue
        kb_id = str(task.get("kb_id") or "").strip()
        if kb_id:
            grouped.setdefault(kb_id, []).append(task)

    discovered_kb_ids: set[str] = set()
    if include_live_preview:
        try:
            discovered_kb_ids = _discover_kb_ids_from_milvus(
                request=request,
                user_id=user_id,
                milvus_db=milvus_db,
            )
        except Exception:
            discovered_kb_ids = set()
        for kb_id in discovered_kb_ids:
            grouped.setdefault(kb_id, [])

    libraries: list[dict[str, Any]] = []
    candidate_kb_ids = discovered_kb_ids if include_live_preview else set(grouped.keys())
    if not candidate_kb_ids:
        candidate_kb_ids = set(grouped.keys())

    for kb_id in sorted(candidate_kb_ids):
        tasks = grouped.get(kb_id, [])
        latest_task = tasks[0] if tasks else None
        summary = _library_summary_from_registry(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db)
        library = {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": milvus_db,
            "latest_task_id": latest_task.get("task_id") if latest_task else None,
            "latest_status": latest_task.get("status") if latest_task else "discovered",
            "latest_stage": latest_task.get("stage") if latest_task else "milvus_only",
            "latest_message": latest_task.get("message") if latest_task else "从向量数据库现有数据发现",
            "updated_at": latest_task.get("updated_at") if latest_task else 0,
            "task_count": len(tasks),
            "file_count": int(summary.get("file_count", 0) or 0),
            "completed": int(summary.get("completed", 0) or 0),
            "failed": int(summary.get("failed", 0) or 0),
            "chunk_count": int(summary.get("chunk_count", 0) or 0),
            "entity_count": int(summary.get("entity_count", 0) or 0),
            "relation_count": int(summary.get("relation_count", 0) or 0),
            "failed_embeddings": int(summary.get("failed_embeddings", 0) or 0),
            "completion_ratio": float(summary.get("completion_ratio", 0.0) or 0.0),
            "source_documents": list(summary.get("source_documents") or []),
            "passage_count": 0,
        }
        if include_live_preview:
            try:
                live_preview = _query_collection_preview(
                    request=request,
                    kb_id=kb_id,
                    user_id=user_id,
                    milvus_db=milvus_db,
                    limit=preview_limit,
                )
                if not any(int(live_preview["counts"].get(key, 0)) > 0 for key in ("entities", "relations", "passages")):
                    continue
                library["live_preview"] = live_preview
                library["entity_count"] = int(live_preview["counts"]["entities"])
                library["relation_count"] = int(live_preview["counts"]["relations"])
                library["passage_count"] = int(live_preview["counts"]["passages"])
                library["completion_ratio"] = 1.0
            except Exception as exc:  # noqa: BLE001
                library["live_preview_error"] = str(exc)
                continue
        libraries.append(library)

    totals = {
        "knowledge_bases": len(libraries),
        "files": sum(int(item.get("file_count", 0)) for item in libraries),
        "entities": sum(int(item.get("entity_count", 0)) for item in libraries),
        "relations": sum(int(item.get("relation_count", 0)) for item in libraries),
        "passages": sum(int(item.get("passage_count", 0)) for item in libraries),
    }
    return {"milvus_db": milvus_db, "user_id": user_id, "totals": totals, "libraries": libraries}


_load_tasks()
_load_library_files()
_reconcile_interrupted_tasks()
if not _library_files:
    _rebuild_library_files_from_tasks()
    _save_library_files()


@router.get("/library/{kb_id}")
def inspect_built_library(
    kb_id: str,
    request: Request,
    user_id: str = "admin_user",
    milvus_db: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    try:
        selected_db = milvus_db or request.app.state.settings.vector_db_database
        return _get_vector_store(request).inspect_library(
            kb_id=kb_id,
            user_id=user_id,
            db_name=selected_db,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"读取向量数据库失败: {exc}") from exc


@router.get("/library/{kb_id}/overview")
def get_library_overview(
    kb_id: str,
    request: Request,
    user_id: str = "admin_user",
    milvus_db: str = "crx",
    limit: int = 16,
) -> dict[str, Any]:
    try:
        live_preview = _query_collection_preview(
            request=request,
            kb_id=kb_id,
            user_id=user_id,
            milvus_db=milvus_db,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}") from exc

    has_live_data = any(int(live_preview["counts"].get(key, 0)) > 0 for key in ("entities", "relations", "passages"))
    if not has_live_data:
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")

    matching_tasks = [
        task
        for task in sorted(_tasks.values(), key=lambda item: float(item.get("updated_at", 0)), reverse=True)
        if task.get("kb_id") == kb_id and task.get("user_id") == user_id and task.get("milvus_db") == milvus_db
    ]
    if not matching_tasks:
        summary = _library_summary_from_registry(kb_id=kb_id, user_id=user_id, milvus_db=milvus_db)
        return {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": milvus_db,
            "latest_task": {
                "task_id": None,
                "status": "discovered",
                "stage": "milvus_only",
                "message": "从向量数据库现有数据发现",
                "kb_id": kb_id,
                "milvus_db": milvus_db,
                "progress_percent": 100,
                "files": [],
            },
            "summary": {
                **summary,
                "entity_count": int(live_preview["counts"]["entities"]),
                "relation_count": int(live_preview["counts"]["relations"]),
            },
            "milvus_preview": live_preview,
            "milvus_error": None,
        }

    latest_task = matching_tasks[0]
    summary = _aggregate_library_summary(matching_tasks)
    overview = {
        "kb_id": kb_id,
        "user_id": user_id,
        "milvus_db": milvus_db,
        "latest_task": _public_task(latest_task),
        "summary": {
            **summary,
            "entity_count": int(live_preview["counts"]["entities"]),
            "relation_count": int(live_preview["counts"]["relations"]),
        },
        "milvus_preview": live_preview,
        "milvus_error": None,
    }
    return overview


@router.get("/library/{kb_id}/history")
def get_library_history(
    kb_id: str,
    user_id: str = "admin_user",
    milvus_db: str = "crx",
    limit: int = 20,
) -> dict[str, Any]:
    matching_tasks = [
        task
        for task in sorted(_tasks.values(), key=lambda item: float(item.get("updated_at", 0)), reverse=True)
        if task.get("kb_id") == kb_id and task.get("user_id") == user_id and task.get("milvus_db") == milvus_db
    ]
    items = [
        {
            **_public_task(task),
            "summary": _task_summary(task),
        }
        for task in matching_tasks[: max(1, min(limit, 100))]
    ]
    return {
        "kb_id": kb_id,
        "user_id": user_id,
        "milvus_db": milvus_db,
        "tasks": items,
    }
