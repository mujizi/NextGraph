from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List
import os
import shutil

router = APIRouter(prefix="/api/upload", tags=["上传管理"])

# 自动生成数据存储目录
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../nextgraph_data"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 定义冲突检查的请求体
class ConflictCheckRequest(BaseModel):
    paths: List[str]

@router.post("/check_conflicts")
async def check_conflicts(request: ConflictCheckRequest):
    """前端上传前调用此接口，检查文件是否已存在"""
    conflicts = []
    for path in request.paths:
        full_path = os.path.join(UPLOAD_DIR, path.lstrip("/"))
        if os.path.exists(full_path):
            conflicts.append(path)
    return {"conflicts": conflicts}

@router.post("/do_upload")
async def do_upload(
    files: List[UploadFile] = File(...),
    target_paths: List[str] = Form(...)
):
    """执行真实的多文件带路径上传"""
    if len(files) != len(target_paths):
        raise HTTPException(status_code=400, detail="文件数量与路径数量不匹配")
        
    saved_files = []
    for i, file in enumerate(files):
        # 拼接目标路径，保持文件夹层级
        full_path = os.path.join(UPLOAD_DIR, target_paths[i].lstrip("/"))
        
        # 确保子文件夹存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # 写入文件
        with open(full_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        saved_files.append(target_paths[i])
        
    return {"code": 200, "message": "上传成功", "count": len(saved_files)}