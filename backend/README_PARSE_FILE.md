# NextGraph Backend 文件解析模块说明

本文档说明当前 `backend` 中与文件解析相关的核心代码，重点覆盖：

- `app/parse_file/api.py`
- `app/parse_file/mineru_service.py`
- `scripts/test_vlm_connect.py`
- `ragflow/deepdoc/parser/mineru_parser.py`（跨仓参考）

## 1. 总体架构

当前后端的文件解析能力是一个“任务化异步流程”：

1. 前端调用 FastAPI 接口提交任务。
2. 后端创建解析任务并进入队列。
3. Worker 进程按 GPU 分配执行 `mineru` 命令。
4. 解析完成后，后处理会读取 `*_content_list.json`，并可调用 VLM 做图片描述增强。
5. 前端轮询进度和结果接口获取状态与产物目录。

## 2. `app/parse_file/api.py`

职责：对外提供解析任务 API。

- 路由前缀：`/parse_file`
- 主要接口：
  - `POST /mineru/submit`：提交任务
  - `GET /mineru/progress/{task_id}`：查询进度
  - `GET /mineru/result/{task_id}`：查询结果

关键数据模型：`MineruParseRequest`

- `file_paths`：待解析文件绝对路径列表
- `output_root`：结果根目录，默认在 `backend/storage/mineru_output`
- `gpus`：使用哪些 GPU（字符串列表）
- `workers_per_gpu`：每张卡并发 worker 数
- `method`：`auto|txt|ocr`
- `backend`：MinerU 后端类型（如 `pipeline`、VLM 相关 backend）
- `lang/start_page/end_page/formula/table/source/vlm_url`：解析细粒度参数

说明：`api.py` 本身不做重处理，主要做参数校验 + 任务分发。

## 3. `app/parse_file/mineru_service.py`

职责：任务执行核心（编排 + 调度 + 结果增强）。

### 3.1 输入预处理

`_prepare_input_for_mineru` 会按扩展名转换：

- Office（`doc/docx/ppt/pptx/xls/xlsx`）-> PDF（依赖 `libreoffice/soffice`）
- 文本（`txt/md`）-> PDF（依赖 `reportlab`）
- 图片（`bmp/tiff/gif/webp`）-> PNG（依赖 Pillow）

目标是把输入统一到 MinerU 更稳定支持的格式。

### 3.2 MinerU 执行

`_run_mineru_for_file(...)` 通过 `subprocess.Popen` 调 `mineru` CLI：

- 构造命令参数：`-p/-o/-m/-b/-l/-s/-e/-f/-t/-u`
- 固定设备参数：`-d cuda:0`
- 由 worker 环境绑定实际 GPU（每个 worker 仅看到一张卡）
- 通过正则解析日志 `batch x/y` 回传任务进度

### 3.3 本地模型配置

`_configure_local_mineru_runtime` 会写 `backend/storage/mineru.local.json`，
把本地模型目录写入 worker 环境变量（无需用户手工导出）。

### 3.4 VLM 描述增强

`_call_vlm_for_enhanced_description`：

- 读取图片 -> base64
- 调用 `${VLM_BASE_URL}/chat/completions`
- 使用 `VLM_API_KEY`、`VLM_MODEL_NAME`
- 生成结构化中文描述（主体/细节/文字/总结/关键词）

该阶段通常在 MinerU 产出 `content_list` 之后执行，属于“后增强”。

## 4. `scripts/test_vlm_connect.py`

职责：独立验证 VLM 服务连通性。

执行三步测试：

1. `GET /models`
2. 文本 `chat/completions`
3. 图片多模态 `chat/completions`（传 `--image` 时启用）

用途：

- 验证网关地址、模型名、API Key 是否可用
- 快速排查“MinerU 成功但图像增强失败”的 VLM 侧问题

## 5. 与 deepdoc/mineru_parser 的关系

`ragflow/deepdoc/parser/mineru_parser.py` 中的 `MinerUParser` 继承了 deepdoc 的 `RAGFlowPdfParser`，
说明 MinerU 在整体架构中是“可接入后端能力”，而 deepdoc 提供统一解析框架和结构化输出约定。

在 NextGraph 这里：

- `parse_file` 模块偏“工程编排层”（任务、并发、落盘、进度、增强）
- deepdoc/mineru_parser 偏“解析框架层”（内容结构规范与解析适配）

## 6. 关键配置项速查

- `VLM_API_KEY`：VLM 鉴权
- `VLM_BASE_URL`：VLM 服务地址
- `VLM_MODEL_NAME`：VLM 模型名
- `MINERU_LOCAL_MODEL_DIR`：本地 MinerU 模型目录
- `MINERU_LOCAL_CONFIG_JSON`：本地 MinerU 配置文件输出位置

## 7. 常见排查路径

1. 提交接口报错：先看 `api.py` 参数是否合法（尤其是 `file_paths/gpus/method`）。
2. 任务跑不动：看 MinerU 命令依赖是否齐全（`mineru`、`libreoffice`、`reportlab`、Pillow）。
3. 结果无 `content_list`：检查输出目录结构和 `_discover_content_json` 命中情况。
4. 图片增强失败：先运行 `scripts/test_vlm_connect.py` 验证 VLM 通道。

