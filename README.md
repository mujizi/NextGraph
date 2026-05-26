# NextGraph

![NextGraph 主图](docs/nextgraph.jpg)

前后端分离项目骨架。

- `frontend/`: React + Vite 前端，实现影谱知识库 UI 页面。
- `backend/`: Python 后端占位目录，后续接入接口。
- `scripts/start_frontend.sh`: 前端启动脚本。
- `scripts/restart_frontend.sh`: 前端重启脚本。

## 前端命令

启动前端：

```bash
./scripts/start_frontend.sh
```

重启前端：

```bash
./scripts/restart_frontend.sh
```

指定端口重启：

```bash
PORT=3000 ./scripts/restart_frontend.sh
```

## 后端分工

| 模块 | 子功能 | 类型 | 负责人 | 目标文件路径 |
| --- | --- | --- | --- | --- |
| 新建知识库 | 新增工作目录 | 接口 | ai | `backend/app/knowledge_base/workspace/create` |
| 新建知识库 | 删除工作目录 | 接口 | ai | `backend/app/knowledge_base/workspace/delete` |
| 上传文件 | 上传文件及文件夹，支持 md、Docx、ppt、excel、pdf、txt | 接口 | 乐天 | `backend/app/upload_file/api` |
| 上传文件 | 删除文件 | 接口 | 乐天 | `backend/app/upload_file/delete` |
| 解析文件 | 传入文件列表，mineru 统一处理，开始默认使用 pipeline（进度条） | 接口 | 萌森 | `backend/app/parse_file/api` |
| 抽三元组 | 可控并发（单模态） | 函数 | 润溪 | `backend/app/triple_extraction/functions` |
| 向量数据库 | 写入 | 函数 |  | `backend/app/vector_database/write` |
| 向量数据库 | 搜 | 函数 |  | `backend/app/vector_database/search` |
| 向量数据库 | 追加 | 函数 |  | `backend/app/vector_database/add` |
| 向量化 pipeline | 抽三元组加向量化 + chunk 向量化（进度条） | 接口 | 润溪 | `backend/app/vectorization_pipeline/api` |
| 搜索 local、global、混合 | 搜索接口 | 接口 | 炜希 | `backend/app/search/local_global_hybrid/api` |
| Chat | 待定 | 待定 | 炜希 | `backend/app/chat/pending` |
| 搜索（复用上面搜索） | 使用搜索功能接口 | 待定 | 炜希 | `backend/app/search_reuse/api` |
| 可视化 | 待定 | 待定 |  | `backend/app/visualization` |


conda activate nextgraph
cd /opt/Workspace/CRX/NextGraph
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 5190

bash /opt/Workspace/CRX/NextGraph/scripts/start_frontend.sh



