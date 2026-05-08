from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import shutil

router = APIRouter(prefix="/api/delete", tags=["删除管理"])

# 定义数据存储的绝对路径（与 add.py 保持完全一致）
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../nextgraph_data"))

# 定义批量删除的请求体
class DeleteRequest(BaseModel):
    paths: List[str]

@router.get("/list")
async def list_files():
    """获取已导入的所有文件列表及其层级结构"""
    # 如果文件夹还没被创建过，直接返回空列表
    if not os.path.exists(DATA_DIR):
        return {"files": [], "total": 0}
    
    file_list = []
    
    # os.walk 会递归遍历文件夹下的所有子目录和文件
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            
            # 计算相对于 nextgraph_data 的相对路径 (前端展示和删除都需要这个路径)
            relative_path = os.path.relpath(full_path, DATA_DIR)
            
            # 统一转换路径分隔符为正斜杠，防止跨平台报错
            relative_path = relative_path.replace("\\", "/")
            
            # 获取文件大小并转换为 KB，保留两位小数，方便前端展示
            size_kb = round(os.path.getsize(full_path) / 1024, 2)
            
            file_list.append({
                "filename": file,
                "path": relative_path,
                "size_kb": size_kb
            })
            
    return {"code": 200, "files": file_list, "total": len(file_list)}

@router.post("/batch")
async def batch_delete(request: DeleteRequest):
    """批量删除文件"""
    if not request.paths:
        raise HTTPException(status_code=400, detail="未提供需要删除的文件路径")

    deleted_count = 0
    failed_paths = []

    for path in request.paths:
        # 安全拼接路径，防止目录穿越攻击
        full_path = os.path.join(DATA_DIR, path.lstrip("/"))
        
        try:
            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                deleted_count += 1
        except Exception as e:
            failed_paths.append({"path": path, "error": str(e)})

    return {
        "code": 200,
        "message": f"成功删除 {deleted_count} 个项目",
        "failed": failed_paths
    }