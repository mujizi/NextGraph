import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入路径更新：
# 1. 对应 backend/app/upload_file/api/add.py
from app.upload_file.api.add import router as upload_api_router
# 2. 对应 backend/app/upload_file/delete/delete.py
from app.upload_file.delete.delete import router as delete_router

app = FastAPI(title="NextGraph Backend")

# [DEBUG] 确认加载
print("=== [DEBUG] 正在加载主程序，当前配置：api/add.py, delete/delete.py ===")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload_api_router)
app.include_router(delete_router)

@app.get("/")
async def root():
    return {"status": "online", "server": "10.1.80.14"}

if __name__ == "__main__":
    # 保持 host 为 0.0.0.0 以便外部访问 10.1.80.14
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)