import uvicorn
from fastapi import FastAPI
from app.vectorization_pipeline.api import router as vector_router

app = FastAPI(title="NextGraph Backend")

# 注册路由
app.include_router(vector_router, prefix="/vectorization", tags=["vectorization"])

@app.get("/")
async def root():
    return {"message": "NextGraph API is running"}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8613)
