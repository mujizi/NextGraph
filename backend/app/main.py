from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

ROOT_DIR = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT_DIR / "tmp"
PARSE_DIR = TMP_DIR / "parse"
GRAPH_DIR = TMP_DIR / "graph"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".xlsx", ".docx"}
MINERU_API_URL = os.environ.get("MINERU_API_URL", "http://10.1.80.16:8004")
# MINERU_API_URL = os.environ.get("MINERU_API_URL", "http://10.1.15.222:8004")
MINERU_BACKEND = os.environ.get("MINERU_BACKEND", "pipeline")
MINERU_API_TIMEOUT = float(os.environ.get("MINERU_API_TIMEOUT", "3600"))
MAX_CONCURRENT_PARSE_FILES = int(os.environ.get("MAX_CONCURRENT_PARSE_FILES", "4"))
MAX_CONCURRENT_GRAPH_FILES = int(os.environ.get("MAX_CONCURRENT_GRAPH_FILES", "2"))
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "text-embedding-small",
)
AZURE_OPENAI_TIMEOUT = float(os.environ.get("AZURE_OPENAI_TIMEOUT", "180"))
AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS = int(
    os.environ.get("AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS", "16384")
)
CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "azure").lower()
CHAT_HISTORY_ROUNDS = int(os.environ.get("CHAT_HISTORY_ROUNDS", "10"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-5.4-mini")
DEFAULT_GRAPH_LLM_CONCURRENCY = int(os.environ.get("GRAPH_LLM_CONCURRENCY", "50"))
DEFAULT_GRAPH_EMBEDDING_CONCURRENCY = int(os.environ.get("GRAPH_EMBEDDING_CONCURRENCY", "50"))
GRAPH_EMBEDDING_BATCH_SIZE = int(os.environ.get("GRAPH_EMBEDDING_BATCH_SIZE", "64"))
DEFAULT_GRAPH_MAX_CHUNK_TOKENS = int(os.environ.get("GRAPH_MAX_CHUNK_TOKENS", "5000"))
GRAPH_LLM_MAX_RETRIES = int(os.environ.get("GRAPH_LLM_MAX_RETRIES", "3"))
GRAPH_LLM_RETRY_BACKOFF = float(os.environ.get("GRAPH_LLM_RETRY_BACKOFF", "1.5"))
MILVUS_URI = os.environ.get("MILVUS_URI", "")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
MILVUS_DB_NAME = os.environ.get("MILVUS_DB_NAME", "benqi")
MILVUS_QUESTION_COLLECTION = os.environ.get("MILVUS_QUESTION_COLLECTION", "potential_questions")
MILVUS_UPSERT_BATCH_SIZE = int(os.environ.get("MILVUS_UPSERT_BATCH_SIZE", "64"))

app = FastAPI(title="NextGraph API")
logger = logging.getLogger("nextgraph.graph_extract")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_segment(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._- " else "_" for char in value)
    return cleaned.strip(" .") or "default"


def safe_relative_path(value: str) -> Path:
    parts = [safe_segment(part) for part in Path(value).parts if part not in ("", ".", "..")]
    if not parts:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return Path(*parts)


def safe_tmp_path(value: str) -> Path:
    path = (ROOT_DIR / value).resolve()
    if TMP_DIR.resolve() not in path.parents and path != TMP_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid stored path")
    return path


class DeleteFilesRequest(BaseModel):
    storedPaths: List[str]


class ParseFileRequest(BaseModel):
    storedPath: str
    name: Optional[str] = None


class ParseFilesRequest(BaseModel):
    knowledgeName: str
    files: List[ParseFileRequest]


class GraphExtractOptions(BaseModel):
    llmConcurrency: int = DEFAULT_GRAPH_LLM_CONCURRENCY
    embeddingConcurrency: int = DEFAULT_GRAPH_EMBEDDING_CONCURRENCY
    maxChunkTokens: int = DEFAULT_GRAPH_MAX_CHUNK_TOKENS
    llmMaxRetries: int = GRAPH_LLM_MAX_RETRIES
    catalogMergeMode: str = "rule"
    forceRebuild: bool = False


class GraphExtractFilesRequest(BaseModel):
    knowledgeName: str
    files: List[ParseFileRequest]
    options: GraphExtractOptions = Field(default_factory=GraphExtractOptions)


class ParseStatusRequest(BaseModel):
    knowledgeName: str
    files: List[ParseFileRequest]


class GraphStatusRequest(BaseModel):
    knowledgeName: str
    files: List[ParseFileRequest]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatStreamRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    knowledgeId: Optional[str] = None
    knowledgeName: Optional[str] = None
    historyRounds: Optional[int] = None


class SearchRequest(BaseModel):
    query: str
    knowledgeName: str
    mode: str = "hybrid"
    localTopK: int = 5
    globalTopK: int = 5


FILM_KNOWLEDGE_TOOL_LABEL = "影视知识库"


parse_batches: Dict[str, Dict[str, Any]] = {}
parse_file_states: Dict[str, Dict[str, Any]] = {}
parse_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARSE_FILES)
graph_batches: Dict[str, Dict[str, Any]] = {}
graph_file_states: Dict[str, Dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_dir_for_file(knowledge_name: str, stored_path: str) -> Path:
    safe_key = uuid.uuid5(uuid.NAMESPACE_URL, stored_path).hex[:12]
    file_stem = safe_segment(Path(stored_path).stem)
    return PARSE_DIR / safe_segment(knowledge_name) / f"{file_stem}-{safe_key}"


def graph_dir_for_file(knowledge_name: str, stored_path: str) -> Path:
    safe_key = uuid.uuid5(uuid.NAMESPACE_URL, stored_path).hex[:12]
    file_stem = safe_segment(Path(stored_path).stem)
    return GRAPH_DIR / safe_segment(knowledge_name) / f"{file_stem}-{safe_key}"


def summarize_batch(batch: Dict[str, Any]) -> None:
    files = batch["files"]
    total = len(files)
    completed = sum(1 for item in files if item["status"] == "parsed")
    failed = sum(1 for item in files if item["status"] == "failed")
    finished = completed + failed
    running = sum(1 for item in files if item["status"] == "parsing")
    queued = sum(1 for item in files if item["status"] == "queued")

    batch["total"] = total
    batch["completed"] = completed
    batch["failed"] = failed
    batch["finished"] = finished
    batch["running"] = running
    batch["queued"] = queued
    batch["progress"] = round((finished / total) * 100) if total else 0

    if total and finished == total:
        batch["status"] = "partial_failed" if failed else "done"
        batch["finishedAt"] = batch.get("finishedAt") or now_iso()
    elif running:
        batch["status"] = "running"
    elif queued:
        batch["status"] = "queued"


def summarize_graph_batch(batch: Dict[str, Any]) -> None:
    files = batch["files"]
    total = len(files)
    completed = sum(1 for item in files if item["status"] == "done")
    failed = sum(1 for item in files if item["status"] in ("failed", "skipped"))
    finished = completed + failed
    running = sum(
        1
        for item in files
        if item["status"]
        in ("validating", "chunking", "llm_extracting", "catalog_merging", "embedding", "writing")
    )
    queued = sum(1 for item in files if item["status"] == "queued")

    batch["total"] = total
    batch["completed"] = completed
    batch["failed"] = failed
    batch["finished"] = finished
    batch["running"] = running
    batch["queued"] = queued
    batch["progress"] = round(sum(item.get("progress", 0) for item in files) / total) if total else 0

    if total and finished == total:
        batch["status"] = "partial_failed" if failed else "done"
        batch["finishedAt"] = batch.get("finishedAt") or now_iso()
    elif running:
        batch["status"] = "running"
    elif queued:
        batch["status"] = "queued"


def write_parse_meta(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_zip_safely(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = (output_dir / member.filename).resolve()
            if target_path != output_root and output_root not in target_path.parents:
                raise ValueError(f"Unsafe zip path from MinerU response: {member.filename}")
        archive.extractall(output_dir)


def save_mineru_response(output_dir: Path, file_name: str, response: httpx.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    if "application/zip" in content_type or response.content.startswith(b"PK"):
        zip_path = output_dir / "mineru_result.zip"
        zip_path.write_bytes(response.content)
        extract_zip_safely(zip_path, output_dir)
        return {
            "responseType": "zip",
            "resultPath": zip_path.relative_to(ROOT_DIR).as_posix(),
        }

    try:
        payload = response.json()
    except ValueError:
        result_path = output_dir / f"{Path(file_name).stem}.txt"
        result_path.write_text(response.text, encoding="utf-8")
        return {
            "responseType": "text",
            "resultPath": result_path.relative_to(ROOT_DIR).as_posix(),
        }

    result_path = output_dir / "mineru_result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "responseType": "json",
        "resultPath": result_path.relative_to(ROOT_DIR).as_posix(),
    }


async def run_mineru_parse(file_item: Dict[str, Any], knowledge_name: str, batch_id: str) -> None:
    batch = parse_batches[batch_id]
    stored_path = file_item["storedPath"]
    source_path = safe_tmp_path(stored_path)
    output_dir = output_dir_for_file(knowledge_name, stored_path)

    async with parse_semaphore:
        file_item["status"] = "parsing"
        file_item["startedAt"] = now_iso()
        parse_file_states[stored_path] = file_item.copy()
        summarize_batch(batch)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        endpoint = f"{MINERU_API_URL.rstrip('/')}/file_parse"

        file_item["outputPath"] = output_dir.relative_to(ROOT_DIR).as_posix()
        file_item["mineruEndpoint"] = endpoint

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(MINERU_API_TIMEOUT),
                trust_env=False,
            ) as client:
                with source_path.open("rb") as source_file:
                    response = await client.post(
                        endpoint,
                        files={"files": (source_path.name, source_file, "application/octet-stream")},
                        data={
                            "backend": MINERU_BACKEND,
                            "return_md": "true",
                            "return_middle_json": "true",
                            "return_content_list": "true",
                            "return_model_output": "true",
                            "return_images": "true",
                            "return_original_file": "true",
                            "response_format_zip": "true",
                        },
                    )
            response.raise_for_status()
            result_meta = save_mineru_response(output_dir, source_path.name, response)
            returncode = 0
            error_text = ""
        except httpx.HTTPStatusError as exc:
            returncode = 1
            error_text = exc.response.text.strip() or str(exc)
            result_meta = {}
        except Exception as exc:
            returncode = 1
            error_text = str(exc)
            result_meta = {}

        file_item["finishedAt"] = now_iso()

        if returncode == 0:
            file_item["status"] = "parsed"
            file_item["progress"] = 100
            file_item["error"] = None
            file_item.update(result_meta)
        else:
            file_item["status"] = "failed"
            file_item["progress"] = 0
            file_item["error"] = error_text[-1000:] or "MinerU API request failed"

        write_parse_meta(
            output_dir,
            {
                "batchId": batch_id,
                "knowledgeName": knowledge_name,
                "storedPath": stored_path,
                "sourcePath": str(source_path),
                "mineruEndpoint": endpoint,
                "status": file_item["status"],
                "error": file_item.get("error"),
                "startedAt": file_item.get("startedAt"),
                "finishedAt": file_item.get("finishedAt"),
                "mineruApiUrl": MINERU_API_URL,
                "mineruBackend": MINERU_BACKEND,
                **result_meta,
            },
        )

        parse_file_states[stored_path] = file_item.copy()
        summarize_batch(batch)


async def process_parse_batch(batch_id: str) -> None:
    batch = parse_batches[batch_id]
    batch["status"] = "running"
    batch["startedAt"] = now_iso()
    summarize_batch(batch)

    await asyncio.gather(
        *[
            run_mineru_parse(file_item, batch["knowledgeName"], batch_id)
            for file_item in batch["files"]
        ],
    )
    summarize_batch(batch)


def find_content_list_path(knowledge_name: str, stored_path: str) -> Path:
    parse_dir = output_dir_for_file(knowledge_name, stored_path)
    if not parse_dir.exists():
        raise FileNotFoundError(f"Parse output not found: {parse_dir.relative_to(ROOT_DIR).as_posix()}")

    matches = sorted(
        path
        for path in parse_dir.rglob("*.json")
        if path.is_file() and path.name.endswith("content_list.json")
    )
    if not matches:
        raise FileNotFoundError("No parsed *_content_list.json found")
    return matches[0]


def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return max(1, round(chinese_chars * 0.75 + other_chars / 4))


def load_paragraphs(content_list_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("content_list.json must be a list")

    paragraphs: List[Dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or item.get("type") != "text":
            continue

        text = str(item.get("text") or "").strip()
        if not text:
            continue

        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            page_idx = 0

        paragraphs.append(
            {
                "paragraphId": f"p_{page_idx:06d}_{len(paragraphs) + 1:04d}",
                "text": text,
                "pageIdx": page_idx,
                "bbox": item.get("bbox"),
                "order": index,
            }
        )

    if not paragraphs:
        raise ValueError("No text paragraphs found in content_list.json")
    return paragraphs


def build_text_chunks(paragraphs: List[Dict[str, Any]], max_chunk_tokens: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_ids: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        block = f"[{paragraph['paragraphId']}]\n{paragraph['text']}"
        block_tokens = estimate_tokens(block)

        if current_lines and current_tokens + block_tokens > max_chunk_tokens:
            chunks.append(
                {
                    "chunkId": f"chunk_{len(chunks) + 1:06d}",
                    "text": "\n\n".join(current_lines),
                    "paragraphIds": current_ids,
                }
            )
            current_lines = []
            current_ids = []
            current_tokens = 0

        current_lines.append(block)
        current_ids.append(paragraph["paragraphId"])
        current_tokens += block_tokens

    if current_lines:
        chunks.append(
            {
                "chunkId": f"chunk_{len(chunks) + 1:06d}",
                "text": "\n\n".join(current_lines),
                "paragraphIds": current_ids,
            }
        )

    return chunks


def normalize_catalog_node(node: Any, valid_paragraph_ids: set[str], depth: int = 1) -> Optional[Dict[str, Any]]:
    if not isinstance(node, dict):
        return None

    title = str(node.get("title") or "").strip()
    summary = str(node.get("summary") or "").strip()
    paragraph_ids = [
        paragraph_id
        for paragraph_id in node.get("paragraphIds", [])
        if isinstance(paragraph_id, str) and paragraph_id in valid_paragraph_ids
    ]
    children = []
    if depth < 3:
        children = [
            child
            for child in (
                normalize_catalog_node(child, valid_paragraph_ids, depth + 1)
                for child in node.get("children", [])
            )
            if child
        ]

    if not paragraph_ids and children:
        merged_ids: List[str] = []
        seen: set[str] = set()
        for child in children:
            for paragraph_id in child.get("paragraphIds", []):
                if paragraph_id not in seen:
                    merged_ids.append(paragraph_id)
                    seen.add(paragraph_id)
        paragraph_ids = merged_ids

    if not title or not summary or not paragraph_ids:
        return None

    normalized = {
        "title": title,
        "summary": summary,
        "paragraphIds": paragraph_ids,
    }
    if children:
        normalized["children"] = children
    return normalized


def normalize_graph_llm_result(payload: Any, valid_paragraph_ids: set[str]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("LLM result must be a JSON object")

    catalog = [
        item
        for item in (
            normalize_catalog_node(item, valid_paragraph_ids)
            for item in payload.get("catalog", [])
        )
        if item
    ]
    questions = []
    for item in payload.get("questions", []):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer_hint = str(item.get("answerHint") or "").strip()
        paragraph_ids = [
            paragraph_id
            for paragraph_id in item.get("paragraphIds", [])
            if isinstance(paragraph_id, str) and paragraph_id in valid_paragraph_ids
        ]
        if question and answer_hint and paragraph_ids:
            questions.append(
                {
                    "question": question,
                    "answerHint": answer_hint,
                    "paragraphIds": paragraph_ids,
                }
            )

    return {"catalog": catalog, "questions": questions}


def append_graph_log(output_dir: Path, event: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event = {"time": now_iso(), **event}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_event, ensure_ascii=False) + "\n")


def graph_timer_start(output_dir: Path, event: str, **payload: Any) -> float:
    started_at = time.perf_counter()
    append_graph_log(output_dir, {"event": f"{event}_started", **payload})
    return started_at


def append_graph_timing(
    output_dir: Path,
    event: str,
    started_at: float,
    **payload: Any,
) -> None:
    append_graph_log(
        output_dir,
        {
            "event": f"{event}_finished",
            "elapsedSeconds": round(time.perf_counter() - started_at, 3),
            **payload,
        },
    )


def assign_catalog_ids(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    paragraph_to_catalog_ids: Dict[str, List[str]] = {}
    counter = 0

    def visit(nodes: List[Dict[str, Any]]) -> None:
        nonlocal counter
        for node in nodes:
            counter += 1
            catalog_id = f"cat_{counter:06d}"
            node["catalogId"] = catalog_id
            for paragraph_id in node.get("paragraphIds", []):
                paragraph_to_catalog_ids.setdefault(paragraph_id, []).append(catalog_id)
            visit(node.get("children", []))

    visit(items)
    return paragraph_to_catalog_ids


async def call_azure_chat_json(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
) -> Dict[str, Any]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

    endpoint = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT), trust_env=False) as client:
        response = await client.post(
            endpoint,
            params={"api-version": AZURE_OPENAI_API_VERSION},
            headers={"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"},
            json={
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_completion_tokens": max_completion_tokens,
            },
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    if not content or not content.strip():
        raise ValueError(
            "Azure OpenAI returned empty chat content"
            f"; finish_reason={choice.get('finish_reason')}"
            f"; usage={payload.get('usage')}"
            f"; content_filter_results={choice.get('content_filter_results')}"
        )
    return json.loads(content)


async def call_openrouter_chat_json(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = 2048,
) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter is not configured: set OPENROUTER_API_KEY")

    endpoint = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT), trust_env=False) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "NextGraph",
            },
            json={
                "model": OPENROUTER_CHAT_MODEL,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": max_completion_tokens,
            },
        )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError("OpenRouter returned empty chat content")
    return json.loads(content)


async def call_chat_json(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = 2048,
) -> Dict[str, Any]:
    if CHAT_PROVIDER == "openrouter":
        return await call_openrouter_chat_json(messages, max_completion_tokens)
    return await call_azure_chat_json(messages, max_completion_tokens)


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def recent_chat_history(messages: List[ChatMessage], rounds: int) -> List[Dict[str, str]]:
    max_messages = max(0, min(50, rounds * 2))
    trimmed = messages[-max_messages:] if max_messages else []
    allowed_roles = {"user", "assistant", "system"}
    return [
        {"role": message.role, "content": message.content.strip()}
        for message in trimmed
        if message.role in allowed_roles and message.content.strip()
    ]


def build_chat_messages(payload: ChatStreamRequest, tool_context: Optional[str] = None) -> List[Dict[str, str]]:
    rounds = payload.historyRounds if payload.historyRounds is not None else CHAT_HISTORY_ROUNDS
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是 NextGraph 的聊天助手，擅长影视、剧本、故事结构和角色分析。"
                "请用中文自然、清晰地回答。若系统提供了影视知识库检索资料，"
                "只在资料确实有帮助时参考；使用资料中的观点或段落时，必须写明出处，"
                "格式如（来源：《书名》/p.页码/段落ID）。不要展示工具调用过程，也不要编造出处。"
            ),
        }
    ]
    if payload.knowledgeName:
        messages.append(
            {
                "role": "system",
                "content": f"当前用户打开的知识库是：{payload.knowledgeName}。",
            }
        )
    if tool_context:
        messages.append({"role": "system", "content": tool_context})
    messages.extend(recent_chat_history(payload.history, rounds))
    messages.append({"role": "user", "content": payload.message.strip()})
    return messages


def build_chat_prompt(knowledge_name: Optional[str]) -> str:
    prompt = (
        "你是 NextGraph 的聊天助手，擅长影视、剧本、故事结构和角色分析。"
        f"你可以调用工具 `{FILM_KNOWLEDGE_TOOL_LABEL}` 检索当前电影/剧本知识库。"
        "当用户问题涉及电影、影视、剧本、编剧、故事结构、叙事、角色、人物、情节、场景、台词、冲突、导演表达，"
        "或需要查找当前知识库资料时，直接调用工具。"
        "工具会返回整理后的 local 相似问题资料和 global 目录段落资料。"
        "不要把检索资料原样堆砌给用户；请综合后自然回答。"
        "如果参考了检索资料中的观点或原文，请在对应句子后写出处，格式如（来源：《书名》/p.页码/段落ID）。"
        "如果资料不相关，可以不用。不要编造出处。"
    )
    if knowledge_name:
        prompt += f"\n当前知识库：{knowledge_name}"
    return prompt


def build_langchain_messages(payload: ChatStreamRequest) -> List[BaseMessage]:
    rounds = payload.historyRounds if payload.historyRounds is not None else CHAT_HISTORY_ROUNDS
    messages: List[BaseMessage] = []
    for item in recent_chat_history(payload.history, rounds):
        if item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
        elif item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
    messages.append(HumanMessage(content=payload.message.strip()))
    return messages


def build_langchain_chat_model():
    if CHAT_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter is not configured: set OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL.rstrip("/"),
            default_headers={
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "NextGraph",
            },
            max_completion_tokens=AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
            timeout=AZURE_OPENAI_TIMEOUT,
            streaming=True,
        )

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT.rstrip("/"),
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        max_completion_tokens=AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
        timeout=AZURE_OPENAI_TIMEOUT,
        streaming=True,
    )


async def film_knowledge_search(knowledge_name: str, query: str) -> Dict[str, Any]:
    return await search_knowledge(
        SearchRequest(
            query=query,
            knowledgeName=knowledge_name,
            mode="hybrid",
            localTopK=5,
            globalTopK=5,
        )
    )


def build_film_knowledge_tool(knowledge_name: Optional[str], search_result_sink: List[Dict[str, Any]]):
    @tool("film_knowledge_base", response_format="content_and_artifact")
    async def film_knowledge_base(query: str) -> tuple[str, Dict[str, Any]]:
        """检索当前电影/剧本知识库。适用于影视、剧本、编剧、故事结构、叙事、角色、情节、场景、台词、冲突等问题。"""

        if not knowledge_name:
            empty_result = {
                "knowledgeName": "",
                "query": query,
                "local": {"results": []},
                "global": {"books": []},
                "error": "当前未选择知识库，无法检索影视知识库。",
            }
            search_result_sink.append(empty_result)
            return "当前未选择知识库，无法检索影视知识库。", empty_result
        search_result = await film_knowledge_search(knowledge_name, query)
        search_result_sink.append(search_result)
        return format_film_knowledge_context(search_result), search_result

    return film_knowledge_base


def build_chat_react_agent(payload: ChatStreamRequest):
    search_result_sink: List[Dict[str, Any]] = []
    agent = create_react_agent(
        build_langchain_chat_model(),
        tools=[build_film_knowledge_tool(payload.knowledgeName, search_result_sink)],
        prompt=build_chat_prompt(payload.knowledgeName),
    )
    return agent, search_result_sink


async def stream_azure_chat(messages: List[Dict[str, str]]):
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

    endpoint = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT), trust_env=False) as client:
        async with client.stream(
            "POST",
            endpoint,
            params={"api-version": AZURE_OPENAI_API_VERSION},
            headers={"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"},
            json={
                "messages": messages,
                "max_completion_tokens": AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                if payload.get("error"):
                    raise RuntimeError(payload["error"].get("message", "Azure OpenAI stream failed"))
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


async def stream_openrouter_chat(messages: List[Dict[str, str]]):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter is not configured: set OPENROUTER_API_KEY")

    endpoint = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT), trust_env=False) as client:
        async with client.stream(
            "POST",
            endpoint,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "NextGraph",
            },
            json={
                "model": OPENROUTER_CHAT_MODEL,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                if payload.get("error"):
                    raise RuntimeError(payload["error"].get("message", "OpenRouter stream failed"))
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


async def call_azure_embedding(input_texts: List[str]) -> List[List[float]]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

    endpoint = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_EMBEDDING_DEPLOYMENT}/embeddings"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT), trust_env=False) as client:
        response = await client.post(
            endpoint,
            params={"api-version": AZURE_OPENAI_API_VERSION},
            headers={"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"},
            json={"input": input_texts},
        )
    response.raise_for_status()
    payload = response.json()
    return [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]


async def extract_chunk_with_llm(chunk: Dict[str, Any], valid_paragraph_ids: set[str]) -> Dict[str, Any]:
    system_prompt = (
        "你是书籍知识库索引构建器。只根据用户给出的段落文本抽取目录树和潜在问题。"
        "必须返回 JSON object。目录最多三级。每个目录节点必须有 title、summary、paragraphIds，"
        "子目录放 children。每个问题必须有 question、answerHint、paragraphIds。"
        "只能引用输入中出现的 paragraphId。不要输出 markdown。"
    )
    user_prompt = (
        "请为下面这段书籍内容生成局部多级目录和潜在检索问题。\n"
        "要求：\n"
        "1. catalog 是目录树数组，最多三级。\n"
        "2. 父目录 paragraphIds 可以是子目录 paragraphIds 的并集，也可以包含总论段落。\n"
        "3. questions 用于后续 embedding 检索，要像用户可能会问的问题。\n"
        "4. 忽略疑似原书目录页，不要复刻书籍原目录。\n\n"
        f"{chunk['text']}"
    )
    payload = await call_azure_chat_json(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_completion_tokens=AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
    )
    result = normalize_graph_llm_result(payload, valid_paragraph_ids)
    result["chunkId"] = chunk["chunkId"]
    return result


async def extract_chunk_with_retries(
    chunk: Dict[str, Any],
    valid_paragraph_ids: set[str],
    output_dir: Path,
    max_retries: int,
) -> Dict[str, Any]:
    attempts = max(1, max_retries)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            result = await extract_chunk_with_llm(chunk, valid_paragraph_ids)
            if attempt > 1:
                append_graph_log(
                    output_dir,
                    {
                        "event": "chunk_retry_succeeded",
                        "chunkId": chunk["chunkId"],
                        "attempt": attempt,
                    },
                )
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "graph_extract chunk failed chunk=%s attempt=%s/%s error=%s",
                chunk["chunkId"],
                attempt,
                attempts,
                last_error,
            )
            append_graph_log(
                output_dir,
                {
                    "event": "chunk_attempt_failed",
                    "chunkId": chunk["chunkId"],
                    "attempt": attempt,
                    "maxAttempts": attempts,
                    "error": last_error[-1000:],
                },
            )
            if attempt < attempts:
                await asyncio.sleep(GRAPH_LLM_RETRY_BACKOFF * attempt)

    append_graph_log(
        output_dir,
        {
            "event": "chunk_skipped",
            "chunkId": chunk["chunkId"],
            "attempts": attempts,
            "error": last_error[-1000:],
        },
    )
    return {
        "chunkId": chunk["chunkId"],
        "catalog": [],
        "questions": [],
        "status": "skipped",
        "error": last_error[-1000:],
    }


async def merge_catalog_once(
    catalog_inputs: List[Dict[str, Any]],
    valid_paragraph_ids: set[str],
) -> List[Dict[str, Any]]:
    system_prompt = (
        "你是书籍知识库目录合并器。你只合并用户提供的局部目录候选。"
        "必须返回 JSON object，格式为 {\"items\": [...]}。目录最多三级。"
        "每个节点必须有 title、summary、paragraphIds，可有 children。"
        "合并语义重复节点，不要生成 catalogId，不要输出 markdown。"
    )
    user_prompt = (
        "请把以下同一本书的局部目录候选合并成全书统一目录树。\n"
        "要求：\n"
        "1. 最多三级。\n"
        "2. 合并重复或近义目录。\n"
        "3. 每个节点保留自己的 summary 和 paragraphIds。\n"
        "4. 父节点 paragraphIds 通常是子节点并集，也可以包含总论段落。\n"
        "5. 不要复刻原书目录页。\n\n"
        f"{json.dumps(catalog_inputs, ensure_ascii=False)}"
    )
    payload = await call_azure_chat_json(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_completion_tokens=AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS,
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [
        item
        for item in (normalize_catalog_node(item, valid_paragraph_ids) for item in items)
        if item
    ]


async def merge_catalog_with_llm(
    chunk_results: List[Dict[str, Any]],
    valid_paragraph_ids: set[str],
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    catalog_inputs = [
        {"chunkId": item["chunkId"], "catalog": item.get("catalog", [])}
        for item in chunk_results
        if item.get("catalog")
    ]
    if not catalog_inputs:
        return []

    round_index = 0
    while len(catalog_inputs) > 1:
        round_index += 1
        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_tokens = 0

        for item in catalog_inputs:
            item_tokens = estimate_tokens(json.dumps(item, ensure_ascii=False))
            if current_group and current_tokens + item_tokens > 12000:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            current_group.append(item)
            current_tokens += item_tokens

        if current_group:
            groups.append(current_group)

        if len(groups) == 1:
            try:
                return await merge_catalog_once(groups[0], valid_paragraph_ids)
            except Exception as exc:
                if output_dir:
                    append_graph_log(
                        output_dir,
                        {"event": "catalog_merge_failed", "round": round_index, "error": str(exc)[-1000:]},
                    )
                raise

        merge_results = await asyncio.gather(
            *(merge_catalog_once(group, valid_paragraph_ids) for group in groups),
            return_exceptions=True,
        )
        merged_groups = []
        for index, result in enumerate(merge_results):
            if isinstance(result, Exception):
                if output_dir:
                    append_graph_log(
                        output_dir,
                        {
                            "event": "catalog_merge_group_skipped",
                            "round": round_index,
                            "groupIndex": index,
                            "error": str(result)[-1000:],
                        },
                    )
                continue
            merged_groups.append(result)

        if not merged_groups:
            return []
        catalog_inputs = [
            {"chunkId": f"merge_{round_index:02d}_{index + 1:04d}", "catalog": items}
            for index, items in enumerate(merged_groups)
            if items
        ]

    return catalog_inputs[0].get("catalog", [])


def normalize_catalog_title(title: str) -> str:
    normalized = title.strip().lower()
    normalized = re.sub(r"^[\s第]*(\d+|[一二三四五六七八九十百千万]+)[章节部篇讲节、.\-：:\s]*", "", normalized)
    normalized = re.sub(r"^\d+(\.\d+)*[\s、.\-：:]+", "", normalized)
    normalized = re.sub(r"[\s　\"'“”‘’《》〈〉（）()【】\[\]、，,。.!！?？:：;；\-—_]+", "", normalized)
    return normalized or title.strip().lower()


def unique_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def merge_summary(existing: str, incoming: str, max_chars: int = 500) -> str:
    if not existing:
        return incoming[:max_chars]
    if not incoming or incoming in existing:
        return existing[:max_chars]
    if existing in incoming:
        return incoming[:max_chars]
    return f"{existing}；{incoming}"[:max_chars]


def find_similar_catalog_node(nodes: List[Dict[str, Any]], title: str) -> Optional[Dict[str, Any]]:
    key = normalize_catalog_title(title)
    for node in nodes:
        node_key = normalize_catalog_title(str(node.get("title") or ""))
        if node_key == key:
            return node
        if key and node_key and (key in node_key or node_key in key):
            shorter = min(len(key), len(node_key))
            longer = max(len(key), len(node_key))
            if shorter >= 4 and shorter / longer >= 0.6:
                return node
        if difflib.SequenceMatcher(None, node_key, key).ratio() >= 0.88:
            return node
    return None


def merge_catalog_node_by_rule(
    target_nodes: List[Dict[str, Any]],
    incoming_node: Dict[str, Any],
    valid_paragraph_ids: set[str],
    depth: int = 1,
) -> None:
    normalized = normalize_catalog_node(incoming_node, valid_paragraph_ids, depth)
    if not normalized:
        return

    target = find_similar_catalog_node(target_nodes, normalized["title"])
    if not target:
        target = {
            "title": normalized["title"],
            "summary": normalized["summary"],
            "paragraphIds": unique_preserve_order(normalized["paragraphIds"]),
        }
        if normalized.get("children"):
            target["children"] = []
        target_nodes.append(target)
    else:
        target["summary"] = merge_summary(target.get("summary", ""), normalized["summary"])
        target["paragraphIds"] = unique_preserve_order(
            [*target.get("paragraphIds", []), *normalized.get("paragraphIds", [])]
        )

    if depth >= 3:
        return

    incoming_children = normalized.get("children", [])
    if incoming_children:
        target_children = target.setdefault("children", [])
        for child in incoming_children:
            merge_catalog_node_by_rule(target_children, child, valid_paragraph_ids, depth + 1)


def merge_catalog_with_rules(
    chunk_results: List[Dict[str, Any]],
    valid_paragraph_ids: set[str],
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    source_node_count = 0

    for result in chunk_results:
        for node in result.get("catalog", []):
            source_node_count += 1
            merge_catalog_node_by_rule(merged, node, valid_paragraph_ids)

    if output_dir:
        append_graph_log(
            output_dir,
            {
                "event": "catalog_rule_merged",
                "sourceNodeCount": source_node_count,
                "mergedTopLevelCount": len(merged),
            },
        )
    return merged


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def flatten_catalog(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []

    def visit(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            flat.append(node)
            visit(node.get("children", []))

    visit(items)
    return flat


async def embed_questions(questions: List[Dict[str, Any]], concurrency: int) -> List[Dict[str, Any]]:
    if not questions:
        return []

    batch_size = max(1, GRAPH_EMBEDDING_BATCH_SIZE)
    batches = [
        (start_index, questions[start_index : start_index + batch_size])
        for start_index in range(0, len(questions), batch_size)
    ]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    rows: List[Optional[Dict[str, Any]]] = [None] * len(questions)

    async def embed_batch(start_index: int, batch_questions: List[Dict[str, Any]]) -> None:
        async with semaphore:
            embeddings = await call_azure_embedding(
                [question["question"] for question in batch_questions]
            )
            if len(embeddings) != len(batch_questions):
                raise RuntimeError(
                    "Azure embedding returned unexpected vector count: "
                    f"expected {len(batch_questions)}, got {len(embeddings)}"
                )
            for offset, (question, embedding) in enumerate(zip(batch_questions, embeddings)):
                rows[start_index + offset] = {**question, "embedding": embedding}

    await asyncio.gather(*(embed_batch(start_index, batch) for start_index, batch in batches))
    return [row for row in rows if row is not None]


def write_questions_to_milvus(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"enabled": False, "inserted": 0, "reason": "no questions"}
    if not MILVUS_URI:
        return {"enabled": False, "inserted": 0, "reason": "MILVUS_URI is not configured"}

    try:
        from pymilvus import MilvusClient  # type: ignore
    except Exception as exc:
        return {"enabled": False, "inserted": 0, "reason": f"pymilvus unavailable: {exc}"}

    try:
        from pymilvus import DataType  # type: ignore

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or None, db_name=MILVUS_DB_NAME)
        if not client.has_collection(MILVUS_QUESTION_COLLECTION):
            dimension = len(rows[0]["embedding"])
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,
            )
            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            client.create_collection(
                collection_name=MILVUS_QUESTION_COLLECTION,
                schema=schema,
                index_params=index_params,
            )
        batch_size = max(1, MILVUS_UPSERT_BATCH_SIZE)
        inserted = 0
        batch_count = 0
        for start_index in range(0, len(rows), batch_size):
            batch_rows = rows[start_index : start_index + batch_size]
            client.upsert(collection_name=MILVUS_QUESTION_COLLECTION, data=batch_rows)
            inserted += len(batch_rows)
            batch_count += 1
        return {
            "enabled": True,
            "inserted": inserted,
            "batchSize": batch_size,
            "batchCount": batch_count,
            "collection": MILVUS_QUESTION_COLLECTION,
        }
    except Exception as exc:
        return {"enabled": False, "inserted": 0, "reason": f"Milvus write failed: {exc}"}


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_graph_books(knowledge_name: str) -> List[Dict[str, Any]]:
    knowledge_dir = GRAPH_DIR / safe_segment(knowledge_name)
    if not knowledge_dir.exists():
        return []

    books: List[Dict[str, Any]] = []
    for book_dir in sorted(path for path in knowledge_dir.iterdir() if path.is_dir()):
        meta_path = book_dir / "meta.json"
        catalog_path = book_dir / "catalog.json"
        if not meta_path.exists() or not catalog_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        paragraphs = read_jsonl(book_dir / "paragraphs.jsonl")
        questions = read_jsonl(book_dir / "questions.jsonl")
        books.append(
            {
                "fileId": meta.get("fileId") or book_dir.name,
                "fileName": meta.get("fileName") or book_dir.name,
                "knowledgeName": meta.get("knowledgeName") or knowledge_name,
                "storedPath": meta.get("storedPath"),
                "contentListPath": meta.get("contentListPath"),
                "catalog": catalog_payload.get("items", []) if isinstance(catalog_payload, dict) else [],
                "paragraphs": paragraphs,
                "questions": questions,
                "graphDir": book_dir.relative_to(ROOT_DIR).as_posix(),
            }
        )
    return books


def paragraph_map(book: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(paragraph.get("paragraphId")): paragraph
        for paragraph in book.get("paragraphs", [])
        if paragraph.get("paragraphId")
    }


def paragraphs_for_ids(book: Dict[str, Any], paragraph_ids: List[str], limit: int = 12) -> List[Dict[str, Any]]:
    by_id = paragraph_map(book)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for paragraph_id in paragraph_ids:
        if paragraph_id in seen:
            continue
        seen.add(paragraph_id)
        paragraph = by_id.get(paragraph_id)
        if not paragraph:
            continue
        rows.append(
            {
                "paragraphId": paragraph_id,
                "text": paragraph.get("text", ""),
                "pageIdx": paragraph.get("pageIdx"),
                "bbox": paragraph.get("bbox"),
                "order": paragraph.get("order"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def flatten_catalog_with_path(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def visit(nodes: List[Dict[str, Any]], path: List[str], depth: int) -> None:
        for node in nodes:
            title = str(node.get("title") or "").strip()
            catalog_id = str(node.get("catalogId") or "")
            next_path = [*path, title] if title else path
            rows.append(
                {
                    "catalogId": catalog_id,
                    "title": title,
                    "summary": str(node.get("summary") or ""),
                    "paragraphIds": [
                        paragraph_id
                        for paragraph_id in node.get("paragraphIds", [])
                        if isinstance(paragraph_id, str)
                    ],
                    "path": next_path,
                    "depth": depth,
                }
            )
            visit(node.get("children", []), next_path, depth + 1)

    visit(items, [], 1)
    return rows


def search_terms(query: str) -> List[str]:
    tokens = [token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", query) if token.strip()]
    chars = [char for char in query.lower() if "\u4e00" <= char <= "\u9fff"]
    return unique_preserve_order([*tokens, *chars])


def text_score(query: str, text: str) -> float:
    text_lower = text.lower()
    terms = search_terms(query)
    if not terms or not text_lower:
        return 0
    score = 0.0
    for term in terms:
        if not term:
            continue
        count = text_lower.count(term)
        if count:
            score += 2.0 if len(term) > 1 else 0.4
            score += min(count, 6) * (0.8 if len(term) > 1 else 0.15)
    if query.strip().lower() in text_lower:
        score += 6.0
    return score


def question_to_local_result(book: Dict[str, Any], question: Dict[str, Any], score: float) -> Dict[str, Any]:
    paragraph_ids = [
        paragraph_id
        for paragraph_id in question.get("paragraphIds", question.get("paragraph_ids", []))
        if isinstance(paragraph_id, str)
    ]
    return {
        "book": {
            "fileId": book["fileId"],
            "fileName": book["fileName"],
        },
        "question": {
            "questionId": question.get("questionId") or question.get("question_id"),
            "question": question.get("question", ""),
            "answerHint": question.get("answerHint") or question.get("answer_hint", ""),
            "score": round(float(score), 4),
            "catalogIds": question.get("catalogIds") or question.get("catalog_ids", []),
        },
        "paragraphs": paragraphs_for_ids(book, paragraph_ids),
    }


async def search_local_with_milvus(
    knowledge_name: str,
    query: str,
    top_k: int,
    books: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not MILVUS_URI:
        return []
    try:
        from pymilvus import MilvusClient  # type: ignore
    except Exception:
        return []

    try:
        embedding = (await call_azure_embedding([query]))[0]
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or None, db_name=MILVUS_DB_NAME)
        if not client.has_collection(MILVUS_QUESTION_COLLECTION):
            return []
        escaped_name = knowledge_name.replace("\\", "\\\\").replace('"', '\\"')
        results = client.search(
            collection_name=MILVUS_QUESTION_COLLECTION,
            data=[embedding],
            limit=top_k,
            filter=f'knowledge_name == "{escaped_name}"',
            output_fields=[
                "file_id",
                "file_name",
                "question_id",
                "question",
                "answer_hint",
                "paragraph_ids",
                "catalog_ids",
            ],
        )
    except Exception as exc:
        logger.warning("local milvus search failed: %s", exc)
        return []

    books_by_id = {book["fileId"]: book for book in books}
    rows: List[Dict[str, Any]] = []
    for hit in (results[0] if results else []):
        if isinstance(hit, dict):
            entity = hit.get("entity", {})
        else:
            entity = getattr(hit, "entity", {}) or {}
        if hasattr(entity, "to_dict"):
            entity = entity.to_dict()
        score = getattr(hit, "score", None)
        if score is None and isinstance(hit, dict):
            score = hit.get("distance", hit.get("score", 0))
        if not isinstance(entity, dict):
            continue
        file_id = entity.get("file_id")
        book = books_by_id.get(file_id)
        if not book:
            continue
        rows.append(question_to_local_result(book, entity, float(score or 0)))
    return rows


def search_local_fallback(books: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    scored: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for book in books:
        by_id = paragraph_map(book)
        for question in book.get("questions", []):
            paragraph_text = "\n".join(
                str(by_id.get(paragraph_id, {}).get("text", ""))
                for paragraph_id in question.get("paragraphIds", [])
            )
            haystack = f"{question.get('question', '')}\n{question.get('answerHint', '')}\n{paragraph_text}"
            score = text_score(query, haystack)
            if score > 0:
                scored.append((score, book, question))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [question_to_local_result(book, question, score) for score, book, question in scored[:top_k]]


async def search_local_results(knowledge_name: str, query: str, top_k: int, books: List[Dict[str, Any]]) -> Dict[str, Any]:
    milvus_results = await search_local_with_milvus(knowledge_name, query, top_k, books)
    fallback_results = [] if milvus_results else search_local_fallback(books, query, top_k)
    return {
        "topK": top_k,
        "strategy": "milvus" if milvus_results else "keyword",
        "results": milvus_results or fallback_results,
    }


def compact_catalog_for_llm(book: Dict[str, Any]) -> Dict[str, Any]:
    def compact(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for node in nodes:
            row = {
                "catalogId": node.get("catalogId"),
                "title": node.get("title"),
                "summary": node.get("summary"),
            }
            children = compact(node.get("children", []))
            if children:
                row["children"] = children
            rows.append(row)
        return rows

    return {
        "fileId": book["fileId"],
        "fileName": book["fileName"],
        "catalog": compact(book.get("catalog", [])),
    }


async def select_global_catalogs_with_llm(
    query: str,
    books: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        return []
    compact_books = [compact_catalog_for_llm(book) for book in books]
    try:
        payload = await call_azure_chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是多书知识库目录检索器。根据用户问题，从多本书的目录树中选择相关子目录。"
                        "只返回 JSON object，格式 {\"selections\":[{\"fileId\":\"...\",\"catalogIds\":[\"...\"],\"reason\":\"...\"}]}。"
                        "catalogIds 必须来自输入，不要输出 markdown。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{query}\n"
                        f"最多选择 {top_k} 个子目录，优先选择最具体的目录。\n\n"
                        f"{json.dumps(compact_books, ensure_ascii=False)}"
                    ),
                },
            ],
            max_completion_tokens=4096,
        )
    except Exception as exc:
        logger.warning("global catalog llm selection failed: %s", exc)
        return []

    selections = payload.get("selections", []) if isinstance(payload, dict) else []
    if not isinstance(selections, list):
        return []
    normalized = []
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        file_id = str(selection.get("fileId") or "")
        catalog_ids = [item for item in selection.get("catalogIds", []) if isinstance(item, str)]
        if file_id and catalog_ids:
            normalized.append(
                {
                    "fileId": file_id,
                    "catalogIds": catalog_ids[:top_k],
                    "reason": str(selection.get("reason") or ""),
                }
            )
    return normalized


def select_global_catalogs_fallback(query: str, books: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    scored: List[tuple[float, str, str]] = []
    for book in books:
        for node in flatten_catalog_with_path(book.get("catalog", [])):
            haystack = " / ".join(node["path"]) + "\n" + node.get("summary", "")
            score = text_score(query, haystack)
            if score > 0 and node.get("catalogId"):
                scored.append((score, book["fileId"], node["catalogId"]))
    scored.sort(key=lambda item: item[0], reverse=True)

    selections: Dict[str, List[str]] = {}
    for _, file_id, catalog_id in scored[:top_k]:
        selections.setdefault(file_id, []).append(catalog_id)
    return [
        {"fileId": file_id, "catalogIds": catalog_ids, "reason": "keyword"}
        for file_id, catalog_ids in selections.items()
    ]


def build_global_books(books: List[Dict[str, Any]], selections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    books_by_id = {book["fileId"]: book for book in books}
    result_books: List[Dict[str, Any]] = []
    for selection in selections:
        book = books_by_id.get(selection["fileId"])
        if not book:
            continue
        flat_nodes = flatten_catalog_with_path(book.get("catalog", []))
        selected_ids = [
            catalog_id
            for catalog_id in selection.get("catalogIds", [])
            if any(node.get("catalogId") == catalog_id for node in flat_nodes)
        ]
        paragraph_ids: List[str] = []
        selected_catalogs = []
        for node in flat_nodes:
            if node.get("catalogId") not in selected_ids:
                continue
            selected_catalogs.append(node)
            paragraph_ids.extend(node.get("paragraphIds", []))
        result_books.append(
            {
                "fileId": book["fileId"],
                "fileName": book["fileName"],
                "graphDir": book["graphDir"],
                "catalog": book.get("catalog", []),
                "selectedCatalogIds": selected_ids,
                "selectedCatalogs": selected_catalogs,
                "reason": selection.get("reason", ""),
                "paragraphs": paragraphs_for_ids(book, paragraph_ids, limit=20),
            }
        )
    return result_books


def material_image_url(knowledge_name: str, file_id: str, paragraph_id: str) -> str:
    return "/api/materials/snippet?" + urlencode(
        {
            "knowledgeName": knowledge_name,
            "fileId": file_id,
            "paragraphId": paragraph_id,
        }
    )


def build_search_materials(search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    knowledge_name = str(search_result.get("knowledgeName") or "")
    local_materials: List[Dict[str, Any]] = []
    global_materials: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def collect_paragraphs(book: Dict[str, Any], paragraphs: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
        file_id = str(book.get("fileId") or "")
        file_name = str(book.get("fileName") or "")
        if not knowledge_name or not file_id:
            return []
        rows: List[Dict[str, Any]] = []
        for paragraph in paragraphs:
            paragraph_id = str(paragraph.get("paragraphId") or "")
            bbox = paragraph.get("bbox")
            page_idx = paragraph.get("pageIdx")
            key = f"{file_id}:{paragraph_id}"
            if not paragraph_id or key in seen or not bbox or not isinstance(page_idx, int):
                continue
            seen.add(key)
            rows.append(
                {
                    "id": key,
                    "sourceType": source_type,
                    "knowledgeName": knowledge_name,
                    "fileId": file_id,
                    "fileName": file_name,
                    "paragraphId": paragraph_id,
                    "pageIdx": page_idx,
                    "bbox": bbox,
                    "text": truncate_text(str(paragraph.get("text") or ""), 220),
                    "imageUrl": material_image_url(knowledge_name, file_id, paragraph_id),
                }
            )
        return rows

    for item in search_result.get("local", {}).get("results", []):
        local_materials.extend(collect_paragraphs(item.get("book", {}), item.get("paragraphs", []), "local"))

    for book in search_result.get("global", {}).get("books", []):
        global_materials.extend(collect_paragraphs(book, book.get("paragraphs", []), "global"))

    return [*local_materials, *global_materials]


def build_materials_from_search_results(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    materials: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for search_result in search_results:
        for material in build_search_materials(search_result):
            material_id = str(material.get("id") or "")
            if not material_id or material_id in seen:
                continue
            seen.add(material_id)
            materials.append(material)
    return materials


def build_catalog_graph_from_search_results(search_results: List[Dict[str, Any]], limit_books: int = 3) -> Dict[str, Any]:
    graph_books: List[Dict[str, Any]] = []
    seen_books: set[str] = set()

    for search_result in search_results:
        for book in search_result.get("global", {}).get("books", []):
            file_id = str(book.get("fileId") or "")
            if not file_id or file_id in seen_books:
                continue
            selected_ids = [str(item) for item in book.get("selectedCatalogIds", []) if item]
            if not selected_ids:
                continue
            flat_catalogs = flatten_catalog_with_path(book.get("catalog", []))
            selected_set = set(selected_ids)
            context_indexes: set[int] = set()
            for index, catalog in enumerate(flat_catalogs):
                if catalog.get("catalogId") not in selected_set:
                    continue
                for nearby_index in range(max(0, index - 2), min(len(flat_catalogs), index + 3)):
                    context_indexes.add(nearby_index)

            context_catalogs = []
            for index in sorted(context_indexes):
                catalog = flat_catalogs[index]
                context_catalogs.append(
                    {
                        "catalogId": catalog.get("catalogId"),
                        "title": catalog.get("title") or "未命名目录",
                        "summary": catalog.get("summary") or "",
                        "path": catalog.get("path", []),
                        "depth": catalog.get("depth", 1),
                        "selected": catalog.get("catalogId") in selected_set,
                    }
                )

            graph_books.append(
                {
                    "fileId": file_id,
                    "fileName": book.get("fileName") or "未知书籍",
                    "catalogs": context_catalogs,
                    "selectedCatalogIds": selected_ids,
                }
            )
            seen_books.add(file_id)
            if len(graph_books) >= limit_books:
                return {"books": graph_books}

    return {"books": graph_books}


async def search_global_results(knowledge_name: str, query: str, top_k: int, books: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm_selections = await select_global_catalogs_with_llm(query, books, top_k)
    fallback_selections = [] if llm_selections else select_global_catalogs_fallback(query, books, top_k)
    return {
        "topK": top_k,
        "strategy": "llm" if llm_selections else "keyword",
        "books": build_global_books(books, llm_selections or fallback_selections),
        "knowledgeName": knowledge_name,
    }


def truncate_text(value: str, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def format_source(file_name: str, paragraph: Dict[str, Any]) -> str:
    page_idx = paragraph.get("pageIdx")
    page_label = f"p.{page_idx + 1}" if isinstance(page_idx, int) else "页码未知"
    return f"来源：《{file_name}》/{page_label}/{paragraph.get('paragraphId', '段落未知')}"


def format_film_knowledge_context(search_result: Dict[str, Any]) -> str:
    local_results = search_result.get("local", {}).get("results", [])
    global_books = search_result.get("global", {}).get("books", [])
    lines = [
        "影视知识库检索资料如下。资料只用于辅助回答，不要逐字堆砌；如果引用或借用资料观点，请在相关句子后写出处。",
        "",
        "一、相关书籍问题资料有：",
    ]

    if local_results:
        for index, item in enumerate(local_results[:5], 1):
            book = item.get("book", {})
            question = item.get("question", {})
            file_name = book.get("fileName", "未知书籍")
            lines.append(
                f"{index}. 《{file_name}》中有一个相似问题：{question.get('question', '')}"
            )
            if question.get("answerHint"):
                lines.append(f"   答案提示：{truncate_text(question.get('answerHint', ''), 260)}")
            for paragraph in item.get("paragraphs", [])[:12]:
                lines.append(
                    f"   原文片段（{format_source(file_name, paragraph)}）："
                    f"{truncate_text(paragraph.get('text', ''), 280)}"
                )
    else:
        lines.append("未检索到可用的相似问题资料。")

    lines.extend(["", "二、相关书籍目录检索到的段落资料："])
    if global_books:
        for book_index, book in enumerate(global_books[:5], 1):
            file_name = book.get("fileName", "未知书籍")
            lines.append(f"{book_index}. 《{file_name}》")
            selected_catalogs = book.get("selectedCatalogs", [])
            if selected_catalogs:
                for catalog in selected_catalogs[:4]:
                    path = " / ".join(catalog.get("path", []) or [catalog.get("title", "相关目录")])
                    lines.append(f"   相关目录：{path}")
                    if catalog.get("summary"):
                        lines.append(f"   目录摘要：{truncate_text(catalog.get('summary', ''), 260)}")
            for paragraph in book.get("paragraphs", [])[:20]:
                lines.append(
                    f"   目录对应原文（{format_source(file_name, paragraph)}）："
                    f"{truncate_text(paragraph.get('text', ''), 300)}"
                )
    else:
        lines.append("未检索到可用的目录段落资料。")

    return "\n".join(lines)


async def run_graph_extract_file(file_item: Dict[str, Any], batch: Dict[str, Any]) -> None:
    stored_path = file_item["storedPath"]
    source_path = safe_tmp_path(stored_path)
    output_dir = graph_dir_for_file(batch["knowledgeName"], stored_path)
    options = batch["options"]

    def set_state(status: str, progress: int, message: str) -> None:
        if file_item.get("_terminal") and status not in ("done", "failed"):
            return

        file_item["status"] = status
        file_item["progress"] = progress
        file_item["message"] = message
        graph_file_states[stored_path] = file_item.copy()
        summarize_graph_batch(batch)

    file_item["startedAt"] = now_iso()
    if output_dir.exists() and options.get("forceRebuild"):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started_at = graph_timer_start(
        output_dir,
        "graph_extract",
        storedPath=stored_path,
        fileName=file_item.get("name"),
        options=options,
    )
    try:
        validation_started_at = graph_timer_start(output_dir, "validation", storedPath=stored_path)
        set_state("validating", 2, "正在检查解析结果")
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"File not found: {stored_path}")

        content_list_path = find_content_list_path(batch["knowledgeName"], stored_path)
        file_item["contentListPath"] = content_list_path.relative_to(ROOT_DIR).as_posix()
        file_item["outputPath"] = output_dir.relative_to(ROOT_DIR).as_posix()
        append_graph_timing(
            output_dir,
            "validation",
            validation_started_at,
            contentListPath=file_item["contentListPath"],
            outputPath=file_item["outputPath"],
        )

        chunking_started_at = graph_timer_start(
            output_dir,
            "chunking",
            contentListPath=file_item["contentListPath"],
            maxChunkTokens=max(500, int(options["maxChunkTokens"])),
        )
        set_state("chunking", 8, "正在清洗段落并合并 chunk")
        paragraphs = load_paragraphs(content_list_path)
        valid_paragraph_ids = {paragraph["paragraphId"] for paragraph in paragraphs}
        chunks = build_text_chunks(paragraphs, max(500, int(options["maxChunkTokens"])))
        file_item["chunksTotal"] = len(chunks)
        file_item["chunksDone"] = 0
        write_jsonl(output_dir / "paragraphs.jsonl", paragraphs)
        append_graph_timing(
            output_dir,
            "chunking",
            chunking_started_at,
            paragraphCount=len(paragraphs),
            chunkCount=len(chunks),
        )

        llm_started_at = graph_timer_start(
            output_dir,
            "llm_extract",
            chunkCount=len(chunks),
            llmConcurrency=max(1, int(options["llmConcurrency"])),
            llmMaxRetries=int(options.get("llmMaxRetries") or GRAPH_LLM_MAX_RETRIES),
        )
        set_state("llm_extracting", 15, f"正在抽取目录和潜在问题 0/{len(chunks)}")
        llm_semaphore = asyncio.Semaphore(max(1, int(options["llmConcurrency"])))
        chunk_results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        skipped_chunks = 0

        async def run_chunk(index: int, chunk: Dict[str, Any]) -> None:
            nonlocal skipped_chunks
            async with llm_semaphore:
                if file_item.get("_terminal"):
                    return
                chunk_results[index] = await extract_chunk_with_retries(
                    chunk,
                    valid_paragraph_ids,
                    output_dir,
                    int(options.get("llmMaxRetries") or GRAPH_LLM_MAX_RETRIES),
                )
                if chunk_results[index].get("status") == "skipped":
                    skipped_chunks += 1
                if file_item.get("_terminal"):
                    return
                file_item["chunksDone"] = int(file_item.get("chunksDone", 0)) + 1
                progress = 15 + round((file_item["chunksDone"] / len(chunks)) * 55)
                set_state(
                    "llm_extracting",
                    min(progress, 70),
                    f"正在抽取目录和潜在问题 {file_item['chunksDone']}/{len(chunks)}，跳过 {skipped_chunks}",
                )

        await asyncio.gather(*(run_chunk(index, chunk) for index, chunk in enumerate(chunks)))
        completed_chunk_results = [item for item in chunk_results if item is not None]
        write_jsonl(output_dir / "chunk_results.jsonl", completed_chunk_results)
        append_graph_timing(
            output_dir,
            "llm_extract",
            llm_started_at,
            chunkCount=len(chunks),
            completedChunkCount=len(completed_chunk_results),
            skippedChunkCount=skipped_chunks,
        )

        catalog_started_at = graph_timer_start(
            output_dir,
            "catalog_merge",
            mode=options.get("catalogMergeMode"),
            completedChunkCount=len(completed_chunk_results),
        )
        set_state("catalog_merging", 75, "正在合并全书目录")
        if options.get("catalogMergeMode") == "llm":
            catalog_items = await merge_catalog_with_llm(completed_chunk_results, valid_paragraph_ids, output_dir)
        else:
            catalog_items = merge_catalog_with_rules(completed_chunk_results, valid_paragraph_ids, output_dir)
        paragraph_to_catalog_ids = assign_catalog_ids(catalog_items)
        append_graph_timing(
            output_dir,
            "catalog_merge",
            catalog_started_at,
            catalogCount=len(flatten_catalog(catalog_items)),
            mappedParagraphCount=len(paragraph_to_catalog_ids),
        )

        question_started_at = graph_timer_start(output_dir, "question_assembly")
        raw_questions: List[Dict[str, Any]] = []
        seen_questions: set[str] = set()
        for result in completed_chunk_results:
            for question in result.get("questions", []):
                key = question["question"]
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                catalog_ids: List[str] = []
                for paragraph_id in question["paragraphIds"]:
                    for catalog_id in paragraph_to_catalog_ids.get(paragraph_id, []):
                        if catalog_id not in catalog_ids:
                            catalog_ids.append(catalog_id)
                raw_questions.append(
                    {
                        "questionId": f"q_{len(raw_questions) + 1:06d}",
                        **question,
                        "catalogIds": catalog_ids,
                    }
                )
        append_graph_timing(
            output_dir,
            "question_assembly",
            question_started_at,
            questionCount=len(raw_questions),
            duplicateQuestionCount=sum(
                len(result.get("questions", [])) for result in completed_chunk_results
            )
            - len(raw_questions),
        )

        set_state("embedding", 88, "正在向量化潜在问题")
        embedding_meta: Dict[str, Any] = {"enabled": True, "error": None}
        embedding_started_at = graph_timer_start(
            output_dir,
            "embedding",
            questionCount=len(raw_questions),
            batchSize=max(1, GRAPH_EMBEDDING_BATCH_SIZE),
            batchCount=(len(raw_questions) + max(1, GRAPH_EMBEDDING_BATCH_SIZE) - 1)
            // max(1, GRAPH_EMBEDDING_BATCH_SIZE),
            embeddingConcurrency=int(options["embeddingConcurrency"]),
            deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            timeoutSeconds=AZURE_OPENAI_TIMEOUT,
        )
        try:
            embedded_questions = await embed_questions(raw_questions, int(options["embeddingConcurrency"]))
        except Exception as exc:
            embedded_questions = []
            embedding_meta = {"enabled": False, "error": str(exc)[-1000:]}
            append_graph_log(
                output_dir,
                {
                    "event": "embedding_failed",
                    "error": embedding_meta["error"],
                    "questionCount": len(raw_questions),
                },
            )
            append_graph_timing(
                output_dir,
                "embedding",
                embedding_started_at,
                status="failed",
                questionCount=len(raw_questions),
                embeddedQuestionCount=0,
                batchSize=max(1, GRAPH_EMBEDDING_BATCH_SIZE),
                error=embedding_meta["error"],
            )
        else:
            append_graph_timing(
                output_dir,
                "embedding",
                embedding_started_at,
                status="done",
                questionCount=len(raw_questions),
                embeddedQuestionCount=len(embedded_questions),
                batchSize=max(1, GRAPH_EMBEDDING_BATCH_SIZE),
                batchCount=(len(raw_questions) + max(1, GRAPH_EMBEDDING_BATCH_SIZE) - 1)
                // max(1, GRAPH_EMBEDDING_BATCH_SIZE),
            )

        file_id = output_dir.name
        vector_prepare_started_at = graph_timer_start(
            output_dir,
            "vector_row_prepare",
            embeddedQuestionCount=len(embedded_questions),
        )
        vector_rows = [
            {
                "id": f"{file_id}_{question['questionId']}",
                "knowledge_name": batch["knowledgeName"],
                "file_id": file_id,
                "file_name": file_item["name"],
                "question_id": question["questionId"],
                "question": question["question"],
                "answer_hint": question["answerHint"],
                "paragraph_ids": question["paragraphIds"],
                "catalog_ids": question["catalogIds"],
                "embedding": question["embedding"],
                "created_at": now_iso(),
            }
            for question in embedded_questions
        ]
        append_graph_timing(
            output_dir,
            "vector_row_prepare",
            vector_prepare_started_at,
            vectorRowCount=len(vector_rows),
            hasEmbedding=bool(vector_rows),
        )
        milvus_started_at = graph_timer_start(
            output_dir,
            "milvus_write",
            vectorRowCount=len(vector_rows),
            batchSize=max(1, MILVUS_UPSERT_BATCH_SIZE),
            batchCount=(len(vector_rows) + max(1, MILVUS_UPSERT_BATCH_SIZE) - 1)
            // max(1, MILVUS_UPSERT_BATCH_SIZE),
            uri=MILVUS_URI,
            database=MILVUS_DB_NAME,
            collection=MILVUS_QUESTION_COLLECTION,
        )
        milvus_meta = write_questions_to_milvus(vector_rows)
        append_graph_timing(
            output_dir,
            "milvus_write",
            milvus_started_at,
            **milvus_meta,
        )

        writing_started_at = graph_timer_start(
            output_dir,
            "result_writing",
            catalogCount=len(flatten_catalog(catalog_items)),
            questionCount=len(raw_questions),
        )
        set_state("writing", 96, "正在写入结果文件")
        (output_dir / "catalog.json").write_text(
            json.dumps({"items": catalog_items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_jsonl(output_dir / "questions.jsonl", raw_questions)
        meta = {
            "knowledgeName": batch["knowledgeName"],
            "fileId": file_id,
            "fileName": file_item["name"],
            "storedPath": stored_path,
            "contentListPath": content_list_path.relative_to(ROOT_DIR).as_posix(),
            "status": "done",
            "paragraphCount": len(paragraphs),
            "chunkCount": len(chunks),
            "skippedChunkCount": skipped_chunks,
            "catalogCount": len(flatten_catalog(catalog_items)),
            "questionCount": len(raw_questions),
            "embedding": embedding_meta,
            "milvus": milvus_meta,
            "createdAt": now_iso(),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        append_graph_timing(
            output_dir,
            "result_writing",
            writing_started_at,
            files=["catalog.json", "questions.jsonl", "meta.json"],
        )

        file_item["finishedAt"] = now_iso()
        file_item["questionsIndexed"] = len(raw_questions)
        set_state("done", 100, "抽取图完成")
        append_graph_timing(
            output_dir,
            "graph_extract",
            total_started_at,
            status="done",
            paragraphCount=len(paragraphs),
            chunkCount=len(chunks),
            questionCount=len(raw_questions),
            questionsIndexed=len(raw_questions),
        )
    except Exception as exc:
        file_item["finishedAt"] = now_iso()
        file_item["error"] = str(exc)[-1000:]
        file_item["_terminal"] = True
        set_state("failed", 0, "抽取图失败")
        append_graph_timing(
            output_dir,
            "graph_extract",
            total_started_at,
            status="failed",
            error=file_item["error"],
        )


async def process_graph_batch(batch_id: str) -> None:
    batch = graph_batches[batch_id]
    batch["status"] = "running"
    batch["startedAt"] = now_iso()
    summarize_graph_batch(batch)
    graph_file_semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_GRAPH_FILES))

    async def run_file_with_limit(file_item: Dict[str, Any]) -> None:
        async with graph_file_semaphore:
            await run_graph_extract_file(file_item, batch)

    await asyncio.gather(*(run_file_with_limit(file_item) for file_item in batch["files"]))
    summarize_graph_batch(batch)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def find_book_for_material(knowledge_name: str, file_id: str) -> Dict[str, Any]:
    for book in load_graph_books(knowledge_name):
        if book.get("fileId") == file_id:
            return book
    raise HTTPException(status_code=404, detail="Material file not found")


def find_origin_pdf_path(book: Dict[str, Any]) -> Path:
    content_list_path = book.get("contentListPath")
    if content_list_path:
        auto_dir = safe_tmp_path(str(content_list_path)).parent
        origin_files = sorted(auto_dir.glob("*_origin.pdf"))
        if origin_files:
            return origin_files[0]
    stored_path = book.get("storedPath")
    if stored_path:
        return safe_tmp_path(str(stored_path))
    raise HTTPException(status_code=404, detail="Origin PDF not found")


def load_model_page_size(book: Dict[str, Any], page_idx: int) -> Optional[tuple[float, float]]:
    content_list_path = book.get("contentListPath")
    if not content_list_path:
        return None
    auto_dir = safe_tmp_path(str(content_list_path)).parent
    model_files = sorted(auto_dir.glob("*_model.json"))
    if not model_files:
        return None
    try:
        payload = json.loads(model_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or page_idx < 0 or page_idx >= len(payload):
        return None
    page_info = payload[page_idx].get("page_info", {}) if isinstance(payload[page_idx], dict) else {}
    width = page_info.get("width")
    height = page_info.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
        return float(width), float(height)
    return None


def load_content_bbox_source_size(book: Dict[str, Any], page_idx: int) -> Optional[tuple[float, float]]:
    content_list_path = book.get("contentListPath")
    if not content_list_path:
        return None
    try:
        payload = json.loads(safe_tmp_path(str(content_list_path)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    page_boxes = [
        item.get("bbox")
        for item in payload
        if isinstance(item, dict) and item.get("page_idx") == page_idx and isinstance(item.get("bbox"), list)
    ]
    if not page_boxes:
        return None
    max_x = max((float(box[2]) for box in page_boxes if len(box) == 4), default=0)
    max_y = max((float(box[3]) for box in page_boxes if len(box) == 4), default=0)
    if 0 < max_x <= 1100 and 0 < max_y <= 1100:
        return 1000.0, 1000.0
    return None


@app.get("/api/materials/snippet")
def get_material_snippet(
    knowledgeName: str,
    fileId: str,
    paragraphId: str,
    padding: int = 18,
) -> Response:
    book = find_book_for_material(knowledgeName, fileId)
    paragraph = paragraph_map(book).get(paragraphId)
    if not paragraph:
        raise HTTPException(status_code=404, detail="Paragraph not found")

    page_idx = paragraph.get("pageIdx")
    bbox = paragraph.get("bbox")
    if not isinstance(page_idx, int) or not isinstance(bbox, list) or len(bbox) != 4:
        raise HTTPException(status_code=400, detail="Paragraph has no page or bbox")

    try:
        import fitz  # type: ignore

        pdf_path = find_origin_pdf_path(book)
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= doc.page_count:
            raise HTTPException(status_code=400, detail="Invalid page index")
        page = doc[page_idx]
        page_rect = page.rect
        source_size = load_content_bbox_source_size(book, page_idx) or load_model_page_size(book, page_idx) or (
            page_rect.width,
            page_rect.height,
        )
        scale_x = page_rect.width / source_size[0]
        scale_y = page_rect.height / source_size[1]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        pad_x = max(0, padding) * scale_x
        pad_y = max(0, padding) * scale_y
        clip = fitz.Rect(
            max(page_rect.x0, x0 * scale_x - pad_x),
            max(page_rect.y0, y0 * scale_y - pad_y),
            min(page_rect.x1, x1 * scale_x + pad_x),
            min(page_rect.y1, y1 * scale_y + pad_y),
        )
        if clip.is_empty or clip.width <= 1 or clip.height <= 1:
            raise HTTPException(status_code=400, detail="Invalid bbox")
        pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=clip, alpha=False)
        return Response(
            content=pix.tobytes("png"),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Render material snippet failed")
        raise HTTPException(status_code=500, detail=str(exc)[-500:])


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest) -> StreamingResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    async def generate():
        try:
            yield sse_event(
                "meta",
                {
                    "provider": CHAT_PROVIDER,
                    "model": OPENROUTER_CHAT_MODEL if CHAT_PROVIDER == "openrouter" else AZURE_OPENAI_CHAT_DEPLOYMENT,
                    "historyRounds": payload.historyRounds or CHAT_HISTORY_ROUNDS,
                },
            )
            agent, search_result_sink = build_chat_react_agent(payload)
            tool_announced = False
            async for event in agent.astream_events(
                {"messages": build_langchain_messages(payload)},
                version="v2",
            ):
                event_name = event.get("event")
                if event_name == "on_tool_start" and event.get("name") == "film_knowledge_base":
                    tool_announced = True
                    yield sse_event(
                        "tool",
                        {"label": FILM_KNOWLEDGE_TOOL_LABEL, "status": "running"},
                    )
                elif event_name == "on_tool_end" and event.get("name") == "film_knowledge_base":
                    yield sse_event(
                        "tool",
                        {"label": FILM_KNOWLEDGE_TOOL_LABEL, "status": "done"},
                    )
                    yield sse_event(
                        "materials",
                        {"items": build_materials_from_search_results(search_result_sink)},
                    )
                    yield sse_event(
                        "catalogGraph",
                        build_catalog_graph_from_search_results(search_result_sink),
                    )
                elif event_name == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    content = getattr(chunk, "content", "")
                    tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
                    if tool_call_chunks:
                        continue
                    if isinstance(content, list):
                        content = "".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    if content:
                        yield sse_event("delta", {"content": content})

            if tool_announced:
                yield sse_event(
                    "tool",
                    {"label": FILM_KNOWLEDGE_TOOL_LABEL, "status": "finished"},
                )
            yield sse_event("done", {})
        except Exception as exc:
            logger.exception("Chat stream failed")
            yield sse_event("error", {"message": str(exc)[-1000:]})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/search")
async def search_knowledge(payload: SearchRequest) -> Dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    mode = payload.mode.lower().strip()
    if mode not in {"local", "global", "hybrid"}:
        raise HTTPException(status_code=400, detail="mode must be local, global, or hybrid")

    local_top_k = max(1, min(20, int(payload.localTopK or 5)))
    global_top_k = max(1, min(20, int(payload.globalTopK or 5)))
    books = load_graph_books(payload.knowledgeName)
    if not books:
        return {
            "query": query,
            "knowledgeName": payload.knowledgeName,
            "mode": mode,
            "local": {"topK": local_top_k, "strategy": "none", "results": []},
            "global": {"topK": global_top_k, "strategy": "none", "books": [], "knowledgeName": payload.knowledgeName},
            "message": "No graph result files found for this knowledge base",
        }

    local_payload = {"topK": local_top_k, "strategy": "skipped", "results": []}
    global_payload = {
        "topK": global_top_k,
        "strategy": "skipped",
        "books": [],
        "knowledgeName": payload.knowledgeName,
    }

    if mode in {"local", "hybrid"}:
        local_payload = await search_local_results(payload.knowledgeName, query, local_top_k, books)
    if mode in {"global", "hybrid"}:
        global_payload = await search_global_results(payload.knowledgeName, query, global_top_k, books)

    return {
        "query": query,
        "knowledgeName": payload.knowledgeName,
        "mode": mode,
        "bookCount": len(books),
        "local": local_payload,
        "global": global_payload,
    }


@app.post("/api/knowledge/upload")
async def upload_knowledge_files(
    knowledge_name: Annotated[str, Form()],
    files: Annotated[List[UploadFile], File()],
    relative_paths: Annotated[Optional[List[str]], Form()] = None,
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    knowledge_dir = TMP_DIR / safe_segment(knowledge_name)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for index, upload in enumerate(files):
        original_path = (
            relative_paths[index]
            if relative_paths and index < len(relative_paths) and relative_paths[index]
            else upload.filename
        )
        if not original_path:
            skipped.append("unknown")
            continue

        target_relative_path = Path(safe_relative_path(original_path).name)
        extension = target_relative_path.suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            skipped.append(original_path)
            continue

        target_path = knowledge_dir / target_relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        size = 0
        with target_path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                output.write(chunk)

        saved.append(
            {
                "name": target_relative_path.name,
                "relativePath": target_relative_path.as_posix(),
                "extension": extension.lstrip(".").upper(),
                "size": size,
                "storedPath": target_path.relative_to(ROOT_DIR).as_posix(),
            },
        )

    return {
        "knowledgeName": knowledge_name,
        "storedDir": knowledge_dir.relative_to(ROOT_DIR).as_posix(),
        "saved": saved,
        "skipped": skipped,
    }


@app.delete("/api/knowledge/files")
def delete_knowledge_files(payload: DeleteFilesRequest) -> Dict[str, List[str]]:
    deleted: List[str] = []
    missing: List[str] = []

    for stored_path in payload.storedPaths:
        target_path = safe_tmp_path(stored_path)
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            deleted.append(stored_path)

            parent = target_path.parent
            while parent != TMP_DIR and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        else:
            missing.append(stored_path)

    return {"deleted": deleted, "missing": missing}


@app.post("/api/knowledge/parse")
async def parse_knowledge_files(payload: ParseFilesRequest) -> Dict[str, Any]:
    if not payload.files:
        raise HTTPException(status_code=400, detail="No files selected")

    batch_id = uuid.uuid4().hex
    files: List[Dict[str, Any]] = []

    for item in payload.files:
        source_path = safe_tmp_path(item.storedPath)
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {item.storedPath}")

        extension = source_path.suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {item.storedPath}")

        output_dir = output_dir_for_file(payload.knowledgeName, item.storedPath)
        file_state = {
            "storedPath": item.storedPath,
            "name": item.name or source_path.name,
            "status": "queued",
            "progress": 0,
            "error": None,
            "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
            "queuedAt": now_iso(),
            "startedAt": None,
            "finishedAt": None,
        }
        files.append(file_state)
        parse_file_states[item.storedPath] = file_state.copy()

    batch = {
        "id": batch_id,
        "knowledgeName": payload.knowledgeName,
        "status": "queued",
        "createdAt": now_iso(),
        "startedAt": None,
        "finishedAt": None,
        "files": files,
    }
    parse_batches[batch_id] = batch
    summarize_batch(batch)

    asyncio.create_task(process_parse_batch(batch_id))
    return batch


@app.get("/api/knowledge/parse/{batch_id}")
def get_parse_batch(batch_id: str) -> Dict[str, Any]:
    batch = parse_batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Parse batch not found")
    summarize_batch(batch)
    return batch


@app.post("/api/knowledge/parse-status")
def get_parse_file_status(payload: ParseStatusRequest) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []

    for item in payload.files:
        source_path = safe_tmp_path(item.storedPath)
        output_dir = output_dir_for_file(payload.knowledgeName, item.storedPath)
        state = parse_file_states.get(item.storedPath)

        if state:
            files.append(state.copy())
            continue

        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                files.append(
                    {
                        "storedPath": item.storedPath,
                        "name": item.name or source_path.name,
                        "status": "failed",
                        "progress": 0,
                        "error": str(exc)[-1000:],
                        "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
                    }
                )
                continue

            status = "parsed" if meta.get("status") == "parsed" else "failed"
            files.append(
                {
                    "storedPath": item.storedPath,
                    "name": item.name or source_path.name,
                    "status": status,
                    "progress": 100 if status == "parsed" else 0,
                    "error": None if status == "parsed" else meta.get("error"),
                    "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
                    "mineruEndpoint": meta.get("mineruEndpoint"),
                    "responseType": meta.get("responseType"),
                    "resultPath": meta.get("resultPath"),
                    "finishedAt": meta.get("finishedAt"),
                }
            )
            continue

        files.append(
            {
                "storedPath": item.storedPath,
                "name": item.name or source_path.name,
                "status": "queued" if source_path.exists() else "failed",
                "progress": 0,
                "error": None if source_path.exists() else f"File not found: {item.storedPath}",
                "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
            }
        )

    batch = {
        "id": "parse-status",
        "knowledgeName": payload.knowledgeName,
        "status": "queued",
        "createdAt": now_iso(),
        "startedAt": None,
        "finishedAt": None,
        "files": files,
    }
    summarize_batch(batch)
    return batch


@app.post("/api/knowledge/graph-extract")
async def graph_extract_knowledge_files(payload: GraphExtractFilesRequest) -> Dict[str, Any]:
    if not payload.files:
        raise HTTPException(status_code=400, detail="No files selected")

    options = model_to_dict(payload.options)
    options["llmConcurrency"] = max(1, min(100, int(options.get("llmConcurrency") or 1)))
    options["embeddingConcurrency"] = max(1, min(100, int(options.get("embeddingConcurrency") or 1)))
    options["maxChunkTokens"] = max(500, min(20000, int(options.get("maxChunkTokens") or 5000)))
    options["llmMaxRetries"] = max(1, min(10, int(options.get("llmMaxRetries") or GRAPH_LLM_MAX_RETRIES)))
    options["catalogMergeMode"] = "llm" if options.get("catalogMergeMode") == "llm" else "rule"
    options["forceRebuild"] = bool(options.get("forceRebuild"))
    options["maxConcurrentFiles"] = max(1, MAX_CONCURRENT_GRAPH_FILES)

    batch_id = uuid.uuid4().hex
    files: List[Dict[str, Any]] = []

    for item in payload.files:
        source_path = safe_tmp_path(item.storedPath)
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {item.storedPath}")

        output_dir = graph_dir_for_file(payload.knowledgeName, item.storedPath)
        file_state = {
            "storedPath": item.storedPath,
            "name": item.name or source_path.name,
            "status": "queued",
            "progress": 0,
            "message": "等待抽取图",
            "error": None,
            "contentListPath": None,
            "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
            "chunksTotal": 0,
            "chunksDone": 0,
            "questionsIndexed": 0,
            "queuedAt": now_iso(),
            "startedAt": None,
            "finishedAt": None,
        }
        files.append(file_state)
        graph_file_states[item.storedPath] = file_state.copy()

    batch = {
        "id": batch_id,
        "knowledgeName": payload.knowledgeName,
        "status": "queued",
        "createdAt": now_iso(),
        "startedAt": None,
        "finishedAt": None,
        "options": options,
        "files": files,
    }
    graph_batches[batch_id] = batch
    summarize_graph_batch(batch)

    asyncio.create_task(process_graph_batch(batch_id))
    return batch


@app.get("/api/knowledge/graph-extract/{batch_id}")
def get_graph_extract_batch(batch_id: str) -> Dict[str, Any]:
    batch = graph_batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Graph extract batch not found")
    summarize_graph_batch(batch)
    return batch


@app.post("/api/knowledge/graph-status")
def get_graph_file_status(payload: GraphStatusRequest) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []

    for item in payload.files:
        source_path = safe_tmp_path(item.storedPath)
        output_dir = graph_dir_for_file(payload.knowledgeName, item.storedPath)
        state = graph_file_states.get(item.storedPath)

        if state:
            files.append(state.copy())
            continue

        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                files.append(
                    {
                        "storedPath": item.storedPath,
                        "name": item.name or source_path.name,
                        "status": "failed",
                        "progress": 0,
                        "message": "读取图谱结果失败",
                        "error": str(exc)[-1000:],
                        "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
                    }
                )
                continue

            files.append(
                {
                    "storedPath": item.storedPath,
                    "name": meta.get("fileName") or item.name or source_path.name,
                    "status": "done" if meta.get("status") == "done" else "failed",
                    "progress": 100 if meta.get("status") == "done" else 0,
                    "message": "抽取图完成" if meta.get("status") == "done" else "抽取图失败",
                    "error": None if meta.get("status") == "done" else meta.get("error"),
                    "contentListPath": meta.get("contentListPath"),
                    "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
                    "chunksTotal": meta.get("chunkCount", 0),
                    "chunksDone": meta.get("chunkCount", 0),
                    "questionsIndexed": meta.get("questionCount", 0),
                    "finishedAt": meta.get("createdAt"),
                }
            )
            continue

        files.append(
            {
                "storedPath": item.storedPath,
                "name": item.name or source_path.name,
                "status": "queued" if source_path.exists() else "failed",
                "progress": 0,
                "message": "等待抽取图" if source_path.exists() else "文件不存在",
                "error": None if source_path.exists() else f"File not found: {item.storedPath}",
                "outputPath": output_dir.relative_to(ROOT_DIR).as_posix(),
            }
        )

    batch = {
        "id": "graph-status",
        "knowledgeName": payload.knowledgeName,
        "status": "queued",
        "createdAt": now_iso(),
        "startedAt": None,
        "finishedAt": None,
        "files": files,
    }
    summarize_graph_batch(batch)
    return batch
