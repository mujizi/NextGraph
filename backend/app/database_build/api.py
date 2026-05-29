from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pymilvus import Collection, connections
from pydantic import BaseModel, Field, field_validator

from backend.app.parse_file.api import DEFAULT_OUTPUT_ROOT
from backend.app.parse_file.mineru_service import TASK_MANAGER
from backend.app.vectorization_pipeline.api import (
    VectorizationRequest,
    milvus_client,
    process_vectorization_task,
    tasks_db as vector_tasks_db,
)

router = APIRouter(prefix="/database_build", tags=["database_build"])

TASK_STORE = (
    Path(__file__).resolve().parent.parent.parent / "storage" / "database_build_tasks.json"
).resolve()
FINISHED_PARSE_STATUSES = {"completed", "failed", "partial_failed"}
FINISHED_BUILD_STATUSES = {"completed", "failed", "partial_failed"}

_tasks_lock = threading.Lock()
_tasks: dict[str, dict[str, Any]] = {}


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


def _save_tasks() -> None:
    TASK_STORE.parent.mkdir(parents=True, exist_ok=True)
    TASK_STORE.write_text(json.dumps(_tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> float:
    return time.time()


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


def _set_task(task_id: str, **updates: Any) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        task.update(updates)
        task["updated_at"] = _now()
        task["progress_percent"] = _task_progress(task)
        _save_tasks()


def _update_file(task_id: str, file_path: str, **updates: Any) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        for item in task["files"]:
            if item["path"] == file_path:
                item.update(updates)
                break
        task["updated_at"] = _now()
        task["progress_percent"] = _task_progress(task)
        _save_tasks()


def _task_progress(task: dict[str, Any]) -> float:
    files = task.get("files") or []
    if not files:
        return 0.0
    return round(sum(float(item.get("progress", 0)) for item in files) / len(files), 2)


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(task, ensure_ascii=False))


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


def _query_collection_preview(
    *,
    request: Request,
    kb_id: str,
    user_id: str,
    milvus_db: str,
    limit: int,
) -> dict[str, Any]:
    selected_db = milvus_db or request.app.state.settings.milvus_db
    _connect_legacy_milvus(request, selected_db)
    collections = {
        "entities": ("Entities", ["id", "name", "relation_ids", "user_id", "kb_id"]),
        "relations": (
            "Relations",
            ["id", "subject_id", "object_id", "relation", "describe", "passage", "user_id", "kb_id"],
        ),
        "passages": ("Passage", ["id", "docment_id", "passage", "user_id", "kb_id"]),
    }
    expr = f'user_id == "{user_id}" and kb_id == "{kb_id}"'
    data: dict[str, Any] = {
        "kb_id": kb_id,
        "user_id": user_id,
        "milvus_db": selected_db,
        "counts": {"entities": 0, "relations": 0, "passages": 0},
        "samples": {"entities": [], "relations": [], "passages": []},
        "top_entities": [],
        "graph_preview": {"nodes": [], "links": []},
    }

    entity_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for key, (collection_name, output_fields) in collections.items():
        collection = Collection(collection_name)
        collection.load()
        count_rows = collection.query(expr=expr, output_fields=["count(*)"])
        matched_count = int(count_rows[0].get("count(*)", 0)) if count_rows else 0
        rows = collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=max(1, min(limit, 100)),
        )
        # Convert RepeatedScalarContainer to list for JSON serialization
        for row in rows:
            for field_name, value in row.items():
                if hasattr(value, "__class__") and "RepeatedScalarContainer" in str(value.__class__):
                    row[field_name] = list(value)
        
        data["counts"][key] = matched_count
        data["samples"][key] = rows
        if key == "entities":
            entity_rows = rows
        elif key == "relations":
            relation_rows = rows

    # Collect all entity IDs involved in the relations to fetch their names
    involved_entity_ids = set()
    for row in relation_rows:
        if row.get("subject_id"):
            involved_entity_ids.add(row["subject_id"])
        if row.get("object_id"):
            involved_entity_ids.add(row["object_id"])
    
    # Also include entities from the entity samples
    involved_entity_ids.update(row.get("id") for row in entity_rows if row.get("id"))
    
    # Fetch all these entities to get their names
    all_involved_entities = []
    if involved_entity_ids:
        entity_collection = Collection("Entities")
        entity_collection.load()
        # Fetch in batches if there are many, but here we limited relations so it's fine
        id_list_str = ", ".join([f'"{eid}"' for eid in involved_entity_ids])
        expr_entities = f'user_id == "{user_id}" and kb_id == "{kb_id}" and id in [{id_list_str}]'
        all_involved_entities = entity_collection.query(
            expr=expr_entities,
            output_fields=["id", "name"]
        )

    entity_lookup = {row.get("id"): row.get("name") or row.get("id") for row in all_involved_entities}
    
    top_entities = sorted(
        (
            {
                "id": row.get("id"),
                "name": row.get("name") or row.get("id"),
                "relation_count": len(row.get("relation_ids") or []),
            }
            for row in entity_rows
        ),
        key=lambda item: item["relation_count"],
        reverse=True,
    )
    data["top_entities"] = top_entities[: min(8, len(top_entities))]

    graph_nodes: dict[str, dict[str, Any]] = {}
    graph_links: list[dict[str, Any]] = []
    for row in relation_rows:
        subject_id = row.get("subject_id")
        object_id = row.get("object_id")
        if not subject_id or not object_id:
            continue
        graph_nodes.setdefault(
            subject_id,
            {"id": subject_id, "name": entity_lookup.get(subject_id, subject_id), "kind": "entity"},
        )
        graph_nodes.setdefault(
            object_id,
            {"id": object_id, "name": entity_lookup.get(object_id, object_id), "kind": "entity"},
        )
        graph_links.append(
            {
                "id": row.get("id"),
                "source": subject_id,
                "target": object_id,
                "label": row.get("relation") or "related_to",
                "kind": "relation",
            }
        )
    data["graph_preview"] = {"nodes": list(graph_nodes.values()), "links": graph_links}
    return data


def _discover_kb_ids_from_milvus(
    *,
    request: Request,
    user_id: str,
    milvus_db: str,
    sample_limit: int = 20000,
) -> set[str]:
    selected_db = milvus_db or request.app.state.settings.milvus_db
    _connect_legacy_milvus(request, selected_db)

    kb_ids: set[str] = set()
    batch_size = 500
    for collection_name in ("Entities", "Relations", "Passage"):
        try:
            collection = Collection(collection_name)
            collection.load()
        except Exception:
            continue

        scanned = 0
        offset = 0
        while scanned < sample_limit:
            try:
                rows = collection.query(
                    expr='id != ""',
                    output_fields=["id", "user_id", "kb_id"],
                    limit=min(batch_size, sample_limit - scanned),
                    offset=offset,
                )
            except Exception:
                break
            if not rows:
                break
            for row in rows:
                if str(row.get("user_id") or "").strip() != user_id:
                    continue
                kb_id = str(row.get("kb_id") or "").strip()
                if kb_id:
                    kb_ids.add(kb_id)
            batch_count = len(rows)
            scanned += batch_count
            offset += batch_count
            if batch_count < min(batch_size, sample_limit - scanned + batch_count):
                break
    return kb_ids


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
    selected_db = milvus_db or request.app.state.settings.milvus_db
    _connect_legacy_milvus(request, selected_db)

    payload: dict[str, Any] = {"milvus_db": selected_db, "collections": {}}
    for collection_name in ("Entities", "Relations", "Passage"):
        collection = Collection(collection_name)
        collection.load()
        rows = collection.query(
            expr='id != ""',
            output_fields=["id", "user_id", "kb_id"],
            limit=max(1, min(sample_limit, 100)),
        )
        payload["collections"][collection_name] = {
            "sample_count": len(rows),
            "samples": rows,
        }
    return payload


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
            state="已写入 Milvus",
            vector_progress=100,
            progress=100,
            summary=state.get("summary", {}),
        )
        return
    raise RuntimeError(str(state.get("msg") or "向量化失败"))


def _run_build_task(task_id: str, payload: DatabaseBuildRequest) -> None:
    try:
        _set_task(task_id, status="running", stage="parsing", message="正在解析文档...")
        milvus_client.use_database(payload.milvus_db)
        text_json_by_file: dict[str, str] = {}
        mineru_files = [path for path in payload.file_paths if not _is_plain_text_source(path)]

        for file_path in payload.file_paths:
            if not _is_plain_text_source(file_path):
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
            _update_file(
                task_id,
                file_path,
                stage="vectorizing",
                state="开始抽取三元组和向量化",
                json_path=json_path,
                progress=45,
            )
            vector_request = VectorizationRequest(
                user_id=payload.user_id,
                kb_id=payload.kb_id,
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
            message="构建完成，数据已写入 Milvus。" if status == "completed" else "构建结束，但存在失败文档。",
            completed=completed,
            failed=failed,
        )
    except Exception as exc:  # noqa: BLE001
        _set_task(task_id, status="failed", stage="failed", message=f"构建任务失败: {exc}")


@router.post("/submit")
def submit_database_build(payload: DatabaseBuildRequest) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    now = _now()
    task = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": "任务已入队",
        "progress_percent": 0,
        "kb_id": payload.kb_id,
        "user_id": payload.user_id,
        "milvus_db": payload.milvus_db,
        "parse_task_id": None,
        "created_at": now,
        "updated_at": now,
        "completed": 0,
        "failed": 0,
        "files": [
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
        ],
    }
    with _tasks_lock:
        _tasks[task_id] = task
        _save_tasks()

    runner = threading.Thread(target=_run_build_task, args=(task_id, payload), daemon=True)
    runner.start()
    return {"task_id": task_id, "status": task["status"], "progress_percent": 0}


@router.get("/progress/{task_id}")
def get_database_build_progress(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _public_task(task)


@router.get("/latest")
def get_latest_database_build(kb_id: str | None = None) -> dict[str, Any]:
    candidates = list(_tasks.values())
    if kb_id:
        candidates = [task for task in candidates if task.get("kb_id") == kb_id]
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
        library = {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": milvus_db,
            "latest_task_id": latest_task.get("task_id") if latest_task else None,
            "latest_status": latest_task.get("status") if latest_task else "discovered",
            "latest_stage": latest_task.get("stage") if latest_task else "milvus_only",
            "latest_message": latest_task.get("message") if latest_task else "从 Milvus 现有数据发现",
            "updated_at": latest_task.get("updated_at") if latest_task else 0,
            "task_count": len(tasks),
            "file_count": 0,
            "completed": 0,
            "failed": 0,
            "chunk_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "failed_embeddings": 0,
            "completion_ratio": 0.0,
            "source_documents": [],
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


def _connect_legacy_milvus(request: Request, milvus_db: str | None = None) -> None:
    settings = request.app.state.settings
    parsed = urlparse(settings.milvus_uri)
    host = parsed.hostname or settings.milvus_uri.replace("http://", "").replace("https://", "")
    port = str(parsed.port or 19530)
    connections.connect(
        alias="default",
        host=host,
        port=port,
        db_name=milvus_db or settings.milvus_db,
        token=settings.milvus_token,
    )


@router.get("/library/{kb_id}")
def inspect_built_library(
    kb_id: str,
    request: Request,
    user_id: str = "admin_user",
    milvus_db: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    try:
        selected_db = milvus_db or request.app.state.settings.milvus_db
        _connect_legacy_milvus(request, selected_db)
        collections = {
            "entities": ("Entities", ["id", "name", "user_id", "kb_id"]),
            "relations": (
            "Relations",
            ["id", "subject_id", "object_id", "relation", "describe", "passage", "user_id", "kb_id"],
        ),
            "passages": ("Passage", ["id", "docment_id", "passage", "user_id", "kb_id"]),
        }
        data: dict[str, Any] = {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": selected_db,
            "collections": {},
        }
        expr = f'user_id == "{user_id}" and kb_id == "{kb_id}"'
        for key, (collection_name, output_fields) in collections.items():
            collection = Collection(collection_name)
            collection.load()
            matched_count: int | None = None
            try:
                count_rows = collection.query(expr=expr, output_fields=["count(*)"])
                if count_rows:
                    matched_count = int(count_rows[0].get("count(*)", 0))
            except Exception:
                matched_count = None
            rows = collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=max(1, min(limit, 100)),
            )
            data["collections"][key] = {
                "collection": collection_name,
                "matched_count": matched_count,
                "sample_count": len(rows),
                "samples": rows,
            }
        return data
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"读取 Milvus 失败: {exc}") from exc


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
        return {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": milvus_db,
            "latest_task": {
                "task_id": None,
                "status": "discovered",
                "stage": "milvus_only",
                "message": "从 Milvus 现有数据发现",
                "kb_id": kb_id,
                "milvus_db": milvus_db,
                "progress_percent": 100,
                "files": [],
            },
            "summary": {
                "file_count": 0,
                "completed": 0,
                "failed": 0,
                "chunk_count": 0,
                "entity_count": int(live_preview["counts"]["entities"]),
                "relation_count": int(live_preview["counts"]["relations"]),
                "failed_embeddings": 0,
                "completion_ratio": 1.0,
                "source_documents": [],
            },
            "milvus_preview": live_preview,
            "milvus_error": None,
        }

    latest_task = matching_tasks[0]
    overview = {
        "kb_id": kb_id,
        "user_id": user_id,
        "milvus_db": milvus_db,
        "latest_task": _public_task(latest_task),
        "summary": {
            "file_count": 0,
            "completed": 0,
            "failed": 0,
            "chunk_count": 0,
            "entity_count": int(live_preview["counts"]["entities"]),
            "relation_count": int(live_preview["counts"]["relations"]),
            "failed_embeddings": 0,
            "completion_ratio": 1.0,
            "source_documents": [],
        },
        "milvus_preview": live_preview,
        "milvus_error": None,
    }
    return overview
