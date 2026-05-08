1.`app/parse_file/api.py`
职责：对外提供解析任务 API。

- 路由前缀：`/parse_file`
- 主要接口：
  - `POST /mineru/submit`：提交任务
  - `GET /mineru/progress/{task_id}`：查询进度
  - `GET /mineru/result/{task_id}`：查询结果

- `file_paths`：待解析文件绝对路径列表
- `output_root`：结果根目录，默认在 `backend/storage/mineru_output`
- `gpus`：使用哪些 GPU（字符串列表）
- `workers_per_gpu`：每张卡并发 worker 数
- `method`：`auto|txt|ocr` 默认auto
- `backend`：MinerU 后端类型（默认`pipeline`）

2.`app/parse_file/mineru_service.py`：mineru任务执行核心