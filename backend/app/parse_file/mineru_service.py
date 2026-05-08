from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpeg",
    ".jpg",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
    ".txt",
    ".md",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}

VLM_API_KEY = "EMPTY"
VLM_BASE_URL = "http://10.1.80.12:8416/v1"
VLM_MODEL_NAME = "/ai/qwen3.5_9b"

MINERU_LOCAL_CONFIG_JSON = "/opt/mengsen/NextGraph/backend/storage/mineru.local.json"


def _configure_local_mineru_runtime(worker_env: dict[str, str]) -> bool:
    config_path = Path(MINERU_LOCAL_CONFIG_JSON).expanduser().resolve()
    if not config_path.exists():
        return False

    config_data: dict[str, Any] = {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except Exception:
        return False

    models_dir = config_data.get("models-dir")
    if not isinstance(models_dir, dict):
        return False
    model_dir = models_dir.get("pipeline")
    if not isinstance(model_dir, str) or not model_dir.strip():
        return False

    model_root = Path(model_dir).expanduser().resolve()
    if not model_root.exists():
        return False

    worker_env["MINERU_MODEL_SOURCE"] = "local"
    worker_env["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)
    return True


def _resolve_bound_gpu_id(gpu_id: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return gpu_id

    visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if not visible_ids:
        return gpu_id

    if gpu_id.isdigit():
        idx = int(gpu_id)
        if 0 <= idx < len(visible_ids):
            return visible_ids[idx]

    return gpu_id


def _build_file_output_dir(base_output_dir: Path, file_path: Path) -> Path:
    short_hash = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return base_output_dir / f"{file_path.stem}_{short_hash}"


def _convert_office_to_pdf(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for cmd in ("libreoffice", "soffice"):
            try:
                result = subprocess.run(
                    [
                        cmd,
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(temp_path),
                        str(input_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                if result.returncode == 0:
                    pdf_files = list(temp_path.glob("*.pdf"))
                    if pdf_files:
                        out_pdf = output_dir / f"{input_path.stem}.pdf"
                        out_pdf.write_bytes(pdf_files[0].read_bytes())
                        return out_pdf
            except FileNotFoundError:
                continue
    raise RuntimeError(f"Office转PDF失败: {input_path}")


def _convert_text_to_pdf(input_path: Path, output_dir: Path) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("txt/md 转 PDF 需要 reportlab") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f"{input_path.stem}.pdf"
    text = input_path.read_text(encoding="utf-8", errors="ignore")

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    _, height = A4
    y = height - 40
    for line in text.splitlines():
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:1000])
        y -= 16
    c.save()
    return out_pdf


def _convert_image_to_png_if_needed(input_path: Path, output_dir: Path) -> Path:
    if input_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return input_path
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("图片格式转换需要 Pillow") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{input_path.stem}_converted.png"
    with Image.open(input_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out_png, "PNG", optimize=True)
    return out_png


def _prepare_input_for_mineru(file_path: Path, file_output_dir: Path) -> Path:
    ext = file_path.suffix.lower()
    if ext in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        return _convert_office_to_pdf(file_path, file_output_dir / "_prepared")
    if ext in {".txt", ".md"}:
        return _convert_text_to_pdf(file_path, file_output_dir / "_prepared")
    if ext in {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}:
        return _convert_image_to_png_if_needed(file_path, file_output_dir / "_prepared")
    return file_path


def _run_mineru_for_file(
    file_path: Path,
    output_dir: Path,
    method: str,
    lang: str | None,
    backend: str | None,
    start_page: int | None,
    end_page: int | None,
    formula: bool,
    table: bool,
    source: str | None,
    vlm_url: str | None,
    env: dict[str, str] | None,
    progress_callback=None,
) -> None:
    cmd = [
        "mineru",
        "-p",
        str(file_path),
        "-o",
        str(output_dir),
        "-m",
        method,
    ]

    if backend:
        cmd.extend(["-b", backend])
    if source:
        cmd.extend(["--source", source])
    if lang:
        cmd.extend(["-l", lang])
    if start_page is not None:
        cmd.extend(["-s", str(start_page)])
    if end_page is not None:
        cmd.extend(["-e", str(end_page)])
    if not formula:
        cmd.extend(["-f", "false"])
    if not table:
        cmd.extend(["-t", "false"])

    # Each worker only sees one physical GPU and should use cuda:0.
    cmd.extend(["-d", "cuda:0"])

    if vlm_url:
        cmd.extend(["-u", vlm_url])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env,
    )
    collected: list[str] = []
    batch_pattern = re.compile(r"batch\\s+(\\d+)/(\\d+)", re.IGNORECASE)
    assert process.stdout is not None
    for line in process.stdout:
        collected.append(line.rstrip("\n"))
        if progress_callback is not None:
            matched = batch_pattern.search(line)
            if matched:
                done = int(matched.group(1))
                total = int(matched.group(2))
                if total > 0:
                    ratio = min(1.0, max(0.0, done / total))
                    # Keep room for post-enhance stage, max 95% here.
                    progress_callback(round(ratio * 95.0, 2))
    ret = process.wait()
    if ret != 0:
        error_text = "\n".join(collected[-30:]) if collected else "MinerU failed"
        raise RuntimeError(error_text)


def _discover_content_json(file_output_dir: Path) -> Path | None:
    candidates = sorted(file_output_dir.rglob("*_content_list.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    return candidates[0]


def _encode_image_file(image_path: Path) -> str | None:
    try:
        raw = image_path.read_bytes()
        return base64.b64encode(raw).decode("utf-8")
    except Exception:
        return None


def _guess_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext in {".bmp"}:
        return "image/bmp"
    if ext in {".tif", ".tiff"}:
        return "image/tiff"
    return "image/png"


def _call_vlm_for_enhanced_description(image_path: Path, context_text: str = "") -> str:
    image_b64 = _encode_image_file(image_path)
    if not image_b64:
        raise RuntimeError(f"无法读取图像: {image_path}")

    mime = _guess_mime_type(image_path)
    prompt = (
        "请对这张图进行详细描述增强。输出中文，要求结构化包含："
        "1) 画面主体；2) 关键细节；3) 文字信息（若有）；4) 语义总结；"
        "5) 可用于检索的关键词（逗号分隔）。"
    )
    if context_text.strip():
        prompt += f"\n补充上下文：{context_text.strip()}"

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        url=f"{VLM_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLM_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"VLM HTTP错误: {exc.code} {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM调用失败: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("VLM返回为空")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in text_parts if part).strip()
    if isinstance(content, str):
        return content.strip()
    raise RuntimeError("VLM返回格式异常")


def _resolve_image_path(raw_path: str, base_dir: Path) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists() and candidate.is_file():
        return candidate
    resolved = (base_dir / raw_path).resolve()
    if resolved.exists() and resolved.is_file():
        return resolved
    return None


def _enhance_descriptions_after_mineru(file_output_dir: Path) -> tuple[str | None, int, int]:
    content_json_path = _discover_content_json(file_output_dir)
    if content_json_path is None:
        return None, 0, 0

    with content_json_path.open("r", encoding="utf-8") as handle:
        content_list = json.load(handle)

    if not isinstance(content_list, list):
        return None, 0, 0

    enhanced_count = 0
    failed_count = 0
    base_dir = content_json_path.parent

    for item in content_list:
        if not isinstance(item, dict):
            continue

        img_path = item.get("img_path")
        if not img_path or not isinstance(img_path, str):
            continue

        resolved_img_path = _resolve_image_path(img_path, base_dir)
        if resolved_img_path is None:
            continue

        context_text = ""
        if item.get("type") == "image":
            captions = item.get("image_caption", item.get("img_caption", []))
            footnotes = item.get("image_footnote", item.get("img_footnote", []))
            context_text = f"captions={captions}; footnotes={footnotes}"
        elif item.get("type") == "table":
            context_text = f"table_caption={item.get('table_caption', [])}"
        elif item.get("type") == "equation":
            context_text = f"equation={item.get('text', '')}"

        try:
            enhanced_text = _call_vlm_for_enhanced_description(
                image_path=resolved_img_path,
                context_text=context_text,
            )
            item["enhanced_caption"] = enhanced_text
            enhanced_count += 1
        except Exception as exc:  # noqa: BLE001
            item["enhance_error"] = str(exc)
            failed_count += 1

    enhanced_json_path = content_json_path.with_name(
        content_json_path.stem + "_enhanced.json"
    )
    with enhanced_json_path.open("w", encoding="utf-8") as handle:
        json.dump(content_list, handle, ensure_ascii=False, indent=2)

    _postprocess_enhanced_content(enhanced_json_path)

    return str(enhanced_json_path), enhanced_count, failed_count


def _to_text_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    page_idx = item.get("page_idx", -1)
    bbox = item.get("bbox", [])
    if not isinstance(bbox, list):
        bbox = []

    if item.get("type") == "text":
        text = str(item.get("text", "")).strip()
    elif item.get("type") == "image":
        text = str(
            item.get("enhanced_caption")
            or item.get("image_caption")
            or item.get("img_caption")
            or ""
        ).strip()
    elif item.get("type") == "table":
        text = str(
            item.get("enhanced_caption")
            or item.get("table_caption")
            or item.get("table_body")
            or ""
        ).strip()
    elif item.get("type") == "equation":
        text = str(item.get("enhanced_caption") or item.get("text") or "").strip()
    else:
        text = str(item.get("enhanced_caption") or item.get("text") or "").strip()

    return {
        "id": index,
        "type": "text",
        "text": text,
        "bbox": bbox,
        "page_idx": page_idx,
    }


def _postprocess_enhanced_content(enhanced_json_path: Path) -> str | None:
    try:
        with enhanced_json_path.open("r", encoding="utf-8") as handle:
            content_list = json.load(handle)
    except Exception:
        return None

    if not isinstance(content_list, list):
        return None

    normalized_items: list[dict[str, Any]] = []
    for idx, item in enumerate(content_list, start=1):
        if not isinstance(item, dict):
            continue
        normalized_items.append(_to_text_item(item, idx))

    output = {
        "total_items": len(normalized_items),
        "items": normalized_items,
    }
    normalized_path = enhanced_json_path.with_name(
        enhanced_json_path.stem + "_textified.json"
    )
    with normalized_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    return str(normalized_path)


@dataclass
class WorkerResult:
    gpu: str
    file_path: str
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    enhanced_json_path: str | None = None
    enhanced_count: int = 0
    enhance_failed: int = 0


def _worker_main(
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    method: str,
    lang: str | None,
    backend: str | None,
    start_page: int | None,
    end_page: int | None,
    formula: bool,
    table: bool,
    source: str | None,
    vlm_url: str | None,
    progress_queue: mp.Queue,
) -> None:
    bound_gpu_id = _resolve_bound_gpu_id(gpu_id)
    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = bound_gpu_id
    _configure_local_mineru_runtime(worker_env)

    base_output_dir = Path(output_dir)

    while True:
        try:
            file_raw = task_queue.get_nowait()
        except Empty:
            break

        file_path = Path(file_raw)
        started = time.time()
        try:
            file_output_dir = _build_file_output_dir(base_output_dir, file_path)
            file_output_dir.mkdir(parents=True, exist_ok=True)
            parse_input_path = _prepare_input_for_mineru(file_path, file_output_dir)
            progress_queue.put({"file_path": str(file_path), "progress": 1.0})
            _run_mineru_for_file(
                file_path=parse_input_path,
                output_dir=file_output_dir,
                method=method,
                lang=lang,
                backend=backend,
                start_page=start_page,
                end_page=end_page,
                formula=formula,
                table=table,
                source="local",
                vlm_url=vlm_url,
                env=worker_env,
                progress_callback=lambda p: progress_queue.put(
                    {"file_path": str(file_path), "progress": p}
                ),
            )
            progress_queue.put({"file_path": str(file_path), "progress": 96.0})

            enhanced_json_path, enhanced_count, enhance_failed = _enhance_descriptions_after_mineru(
                file_output_dir=file_output_dir
            )
            progress_queue.put({"file_path": str(file_path), "progress": 100.0})

            result_queue.put(
                WorkerResult(
                    gpu=gpu_id,
                    file_path=str(file_path),
                    success=True,
                    duration_seconds=time.time() - started,
                    enhanced_json_path=enhanced_json_path,
                    enhanced_count=enhanced_count,
                    enhance_failed=enhance_failed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            result_queue.put(
                WorkerResult(
                    gpu=gpu_id,
                    file_path=str(file_path),
                    success=False,
                    error=str(exc),
                    duration_seconds=time.time() - started,
                )
            )


@dataclass
class ParseTask:
    task_id: str
    files: list[str]
    output_dir: str
    gpus: list[str]
    workers_per_gpu: int
    status: str = "queued"
    total: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    errors: list[dict[str, Any]] = field(default_factory=list)
    file_results: list[dict[str, Any]] = field(default_factory=list)
    per_file_progress: dict[str, float] = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return round((self.completed / self.total) * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "files": self.files,
            "output_dir": self.output_dir,
            "gpus": self.gpus,
            "workers_per_gpu": self.workers_per_gpu,
            "total": self.total,
            "completed": self.completed,
            "success": self.success,
            "failed": self.failed,
            "progress_percent": self.progress_percent,
            "errors": self.errors,
            "file_results": self.file_results,
            "per_file_progress": self.per_file_progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MineruTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, ParseTask] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        file_paths: list[str],
        output_root: str,
        gpus: list[str],
        workers_per_gpu: int,
        method: str,
        lang: str | None,
        backend: str | None,
        start_page: int | None,
        end_page: int | None,
        formula: bool,
        table: bool,
        source: str | None,
        vlm_url: str | None,
    ) -> ParseTask:
        files = self._validate_files(file_paths)
        task_id = uuid.uuid4().hex
        task_output_dir = Path(output_root).expanduser().resolve() / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        task = ParseTask(
            task_id=task_id,
            files=[str(path) for path in files],
            output_dir=str(task_output_dir),
            gpus=gpus,
            workers_per_gpu=workers_per_gpu,
            status="running",
            total=len(files),
        )

        with self._lock:
            self._tasks[task_id] = task

        runner = threading.Thread(
            target=self._run_task,
            kwargs={
                "task_id": task_id,
                "files": files,
                "output_dir": str(task_output_dir),
                "gpus": gpus,
                "workers_per_gpu": workers_per_gpu,
                "method": method,
                "lang": lang,
                "backend": backend,
                "start_page": start_page,
                "end_page": end_page,
                "formula": formula,
                "table": table,
                "source": source,
                "vlm_url": vlm_url,
            },
            daemon=True,
        )
        runner.start()
        return task

    def get(self, task_id: str) -> ParseTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _update_task(self, task_id: str, **kwargs: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in kwargs.items():
                setattr(task, key, value)
            task.updated_at = time.time()

    def _append_error(self, task_id: str, error_item: dict[str, Any]) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.errors.append(error_item)
            task.updated_at = time.time()

    def _append_file_result(self, task_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.file_results.append(result)
            task.updated_at = time.time()

    def _run_task(
        self,
        task_id: str,
        files: list[Path],
        output_dir: str,
        gpus: list[str],
        workers_per_gpu: int,
        method: str,
        lang: str | None,
        backend: str | None,
        start_page: int | None,
        end_page: int | None,
        formula: bool,
        table: bool,
        source: str | None,
        vlm_url: str | None,
    ) -> None:
        task_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()
        progress_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        for file_path in files:
            task_queue.put(str(file_path))

        for gpu_id in gpus:
            for index in range(workers_per_gpu):
                process = mp.Process(
                    target=_worker_main,
                    args=(
                        gpu_id,
                        task_queue,
                        result_queue,
                        output_dir,
                        method,
                        lang,
                        backend,
                        start_page,
                        end_page,
                        formula,
                        table,
                        source,
                        vlm_url,
                        progress_queue,
                    ),
                    name=f"mineru-{task_id[:6]}-gpu{gpu_id}-w{index + 1}",
                )
                process.start()
                processes.append(process)

        completed = 0
        success = 0
        failed = 0
        total = len(files)

        try:
            while completed < total:
                try:
                    while True:
                        event = progress_queue.get_nowait()
                        if isinstance(event, dict):
                            file_path = str(event.get("file_path", ""))
                            progress = float(event.get("progress", 0.0))
                            with self._lock:
                                task = self._tasks.get(task_id)
                                if task is not None and file_path:
                                    task.per_file_progress[file_path] = max(
                                        task.per_file_progress.get(file_path, 0.0), progress
                                    )
                                    task.updated_at = time.time()
                except Empty:
                    pass

                try:
                    result: WorkerResult = result_queue.get(timeout=1.0)
                except Empty:
                    if not any(process.is_alive() for process in processes):
                        break
                    continue

                completed += 1
                if result.success:
                    success += 1
                    self._append_file_result(
                        task_id,
                        {
                            "file_path": result.file_path,
                            "enhanced_json_path": result.enhanced_json_path,
                            "enhanced_count": result.enhanced_count,
                            "enhance_failed": result.enhance_failed,
                            "duration_seconds": round(result.duration_seconds, 2),
                        },
                    )
                else:
                    failed += 1
                    self._append_error(
                        task_id,
                        {
                            "gpu": result.gpu,
                            "file_path": result.file_path,
                            "error": result.error,
                            "duration_seconds": round(result.duration_seconds, 2),
                        },
                    )

                self._update_task(
                    task_id,
                    completed=completed,
                    success=success,
                    failed=failed,
                    status="running",
                )

            for process in processes:
                process.join()

            if completed < total:
                missing = total - completed
                failed += missing
                completed = total
                self._append_error(
                    task_id,
                    {
                        "gpu": "system",
                        "file_path": "*",
                        "error": f"{missing} 个文件未返回结果，可能是worker异常退出",
                        "duration_seconds": 0,
                    },
                )

            status = "completed" if failed == 0 else ("failed" if success == 0 else "partial_failed")
            self._update_task(
                task_id,
                completed=completed,
                success=success,
                failed=failed,
                status=status,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_error(
                task_id,
                {
                    "gpu": "system",
                    "file_path": "*",
                    "error": f"Task runner error: {exc}",
                    "duration_seconds": 0,
                },
            )
            self._update_task(task_id, status="failed")

    @staticmethod
    def _validate_files(file_paths: list[str]) -> list[Path]:
        if not file_paths:
            raise ValueError("file_paths 不能为空")

        valid_paths: list[Path] = []
        for raw_path in file_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"文件不存在: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"不支持的文件类型: {path.suffix} ({path})")
            valid_paths.append(path)

        return valid_paths


TASK_MANAGER = MineruTaskManager()
