from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .mineru_service import TASK_MANAGER, VLM_API_KEY, VLM_BASE_URL, VLM_MODEL_NAME

router = APIRouter(prefix="/parse_file", tags=["parse_file"])

DEFAULT_OUTPUT_ROOT = str(
    (Path(__file__).resolve().parent.parent.parent / "storage" / "mineru_output").resolve()
)


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
