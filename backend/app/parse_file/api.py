from __future__ import annotations

from pathlib import Path
import shutil
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from .mineru_service import TASK_MANAGER, VLM_API_KEY, VLM_BASE_URL, VLM_MODEL_NAME

router = APIRouter(prefix="/parse_file", tags=["parse_file"])
files_router = APIRouter(prefix="/files", tags=["files"])

DEFAULT_OUTPUT_ROOT = str(
    (Path(__file__).resolve().parent.parent.parent / "storage" / "mineru_output").resolve()
)
DEFAULT_UPLOAD_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "storage" / "uploads"
).resolve()
SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".md",
    ".txt",
    ".json",
}


def _safe_upload_name(filename: str) -> str:
    raw_name = Path(filename or "untitled").name
    stem = Path(raw_name).stem or "untitled"
    suffix = Path(raw_name).suffix.lower()
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{safe_stem[:80]}_{int(time.time() * 1000)}{suffix}"


def _file_payload(path: Path) -> dict:
    stat = path.stat()
    return {
        "id": str(path),
        "name": path.name,
        "path": str(path),
        "size": stat.st_size,
        "uploaded_at": int(stat.st_mtime),
        "extension": path.suffix.lower().lstrip(".") or "file",
    }


@files_router.get("")
def list_uploaded_files() -> dict:
    DEFAULT_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    files = [
        _file_payload(path)
        for path in sorted(DEFAULT_UPLOAD_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
        if path.is_file()
    ]
    return {"files": files}


@files_router.post("/upload")
def upload_files(files: list[UploadFile] = File(...)) -> dict:
    DEFAULT_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    uploaded: list[dict] = []

    for item in files:
        suffix = Path(item.filename or "").suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {item.filename}")

        target = DEFAULT_UPLOAD_ROOT / _safe_upload_name(item.filename or "untitled")
        with target.open("wb") as buffer:
            shutil.copyfileobj(item.file, buffer)
        uploaded.append(_file_payload(target))

    return {"files": uploaded}


class MineruParseRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1, description="前端选中的本地文件绝对路径列表")
    output_root: str = Field(DEFAULT_OUTPUT_ROOT, description="解析结果根目录")
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

    @field_validator("gpus")
    @classmethod
    def normalize_gpus(cls, value: list[str]) -> list[str]:
        gpu_ids = [str(item).strip() for item in value if str(item).strip()]
        if not gpu_ids:
            raise ValueError("gpus 不能为空")
        return gpu_ids


@router.post("/mineru/submit")
def submit_mineru_parse_task(payload: MineruParseRequest) -> dict:
    try:
        task = TASK_MANAGER.submit(
            file_paths=payload.file_paths,
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
        return {
            "task_id": task.task_id,
            "status": task.status,
            "output_dir": task.output_dir,
            "total": task.total,
            "progress_percent": task.progress_percent,
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"提交任务失败: {exc}") from exc


@router.get("/mineru/progress/{task_id}")
def get_mineru_task_progress(task_id: str) -> dict:
    task = TASK_MANAGER.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress_percent": task.progress_percent,
        "completed": task.completed,
        "total": task.total,
        "success": task.success,
        "failed": task.failed,
        "per_file_progress": task.per_file_progress,
    }


@router.get("/mineru/result/{task_id}")
def get_mineru_task_result(task_id: str) -> dict:
    task = TASK_MANAGER.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    data = task.to_dict()
  
    return data
