# NextGraph Backend

本次实现补齐了基于 Milvus 的三种检索接口，并新增了：

- **query entity extraction**：通过 Azure OpenAI `gpt-5.4-mini` chat completion 抽取查询实体
- **passage grounding**：根据关系命中的 `passage_ids` 回查 `Passage` 表并返回证据段落

## 检索能力

- `triple`：先做 **query entity extraction**，再做 **entity seed retrieval + relation seed retrieval**，随后按 `relation_ids`、`subject_id`、`object_id` 做 **subgraph expansion**，最后在扩展后的关系子图内做受限向量排序。
- `semantic`：直接对 `Relations.embedding` 做语义向量检索，并返回命中的实体种子。
- `hybrid`：融合 `triple` 与 `semantic` 的候选结果并重新排序。
- `grounded_passages`：从最终命中的 `passage_ids` 回查 `Passage` collection，返回原文证据段落。

---

## 目录结构

```text
backend/
├── app/
│   ├── core/
│   ├── search/
│   └── vector_database/
├── scripts/
└── tests/
```

---

## 环境准备

你可以使用任意 Python 3.11+ 环境（`venv`、`conda`、`poetry`、`uv` 均可）。

示例（venv）：

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

或者如果你使用 conda：

```bash
conda create -n nextgraph python=3.11 -y
conda activate nextgraph
pip install -r backend/requirements.txt
```

---

## 配置

后端通过环境变量读取配置。建议在仓库根目录准备 `.env` 文件。

最小配置示例：

```env
API_KEY=your-azure-openai-key
NEXTGRAPH_MILVUS_URI=http://localhost:19530
NEXTGRAPH_MILVUS_DB=nextGraph
NEXTGRAPH_AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
NEXTGRAPH_AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4-mini
```

如果你的 Milvus 不在本机，请把 `NEXTGRAPH_MILVUS_URI` 改成实际地址（http://10.1.80.16:19530）。

---

## 启动服务

在仓库根目录执行：

```bash
uvicorn backend.app.main:app --reload --port 8000
```

如果你的 Python 环境没有把项目根目录加入模块搜索路径，可以这样启动：

```bash
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

---

## 接口总览

- `POST /api/search`
- `POST /api/search/triple`
- `POST /api/search/semantic`
- `POST /api/search/hybrid`

其中：

- `/api/search`：通用入口，由 `mode` 决定走哪种检索
- `/api/search/triple`：强制走 triple 检索
- `/api/search/semantic`：强制走 semantic 检索
- `/api/search/hybrid`：强制走 hybrid 检索

---

## 通用请求头

```http
Content-Type: application/json
```

---

## 通用入参

三个接口的请求体结构一致：

```json
{
  "query": "谁为星际穿越配乐？",
  "user_id": "test-user-001",
  "kb_id": "test-kb-cinema",
  "mode": "hybrid",
  "top_k": 3,
  "entity_top_k": 4,
  "relation_top_k": 4,
  "expansion_degree": 1
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `query` | `string` | 是 | 查询问题或查询语句 |
| `user_id` | `string` | 是 | 用户作用域过滤条件 |
| `kb_id` | `string` | 是 | 知识库作用域过滤条件 |
| `mode` | `string` | 否 | `triple` / `semantic` / `hybrid`。对于专用接口可忽略，服务端会强制覆盖 |
| `top_k` | `int` | 否 | 最终返回关系条数，默认 `10` |
| `entity_top_k` | `int` | 否 | 实体召回数量，默认 `8` |
| `relation_top_k` | `int` | 否 | 关系召回数量，默认 `8` |
| `expansion_degree` | `int` | 否 | triple 路径下的子图扩展 hop 数，默认 `1` |

说明：虽然核心作用域过滤字段是 `user_id` 和 `kb_id`，但“搜索接口”必须同时提供 `query` 才能执行检索，因此 `query` 也是必填。

---

## 通用出参

三个接口的返回结构一致：

```json
{
  "mode": "triple",
  "query": "谁为星际穿越配乐？",
  "user_id": "test-user-001",
  "kb_id": "test-kb-cinema",
  "results": [
    {
      "id": "rel_music_interstellar",
      "subject_id": "ent_hans",
      "subject_name": "汉斯·季默",
      "object_id": "ent_interstellar",
      "object_name": "星际穿越",
      "relation": "配乐",
      "passage": "《星际穿越》的配乐由汉斯·季默创作。",
      "docment_id": "doc-001",
      "score": 0.7261,
      "score_breakdown": {
        "triple_score": 0.7261
      },
      "source_modes": ["triple"],
      "matched_entity_ids": ["ent_hans", "ent_interstellar"],
      "passage_ids": ["passage-002"]
    }
  ],
  "entity_hits": [
    {
      "id": "ent_interstellar",
      "name": "星际穿越",
      "score": 0.9999,
      "relation_ids": [
        "rel_directed_interstellar",
        "rel_music_interstellar"
      ]
    }
  ],
  "grounded_passages": [
    {
      "id": "passage-002",
      "passage": "《星际穿越》的配乐由汉斯·季默创作。",
      "docment_id": "doc-001",
      "score": 0.7261,
      "matched_relation_ids": ["rel_music_interstellar"]
    }
  ],
  "metadata": {
    "took_ms": 3545.82,
    "entity_top_k": 4,
    "relation_top_k": 4,
    "expansion_degree": 1,
    "result_count": 3,
    "grounded_passage_count": 3,
    "semantic_candidate_count": 0,
    "triple_candidate_count": 6,
    "hybrid_weights": {
      "semantic": 0.55,
      "triple": 0.45
    },
    "embedder": "AzureOpenAIEmbedder",
    "route": {
      "query_entities": ["星际穿越"],
      "seed_entity_ids": ["ent_interstellar"],
      "seed_relation_ids": ["rel_music_interstellar"]
    }
  }
}
```

### 顶层字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `string` | 实际执行的检索模式 |
| `query` | `string` | 原始查询 |
| `user_id` | `string` | 用户作用域 |
| `kb_id` | `string` | 知识库作用域 |
| `results` | `array` | 最终命中的关系结果 |
| `entity_hits` | `array` | 实体 seed 召回结果 |
| `grounded_passages` | `array` | 从 `Passage` 表回查得到的证据段落 |
| `metadata` | `object` | 调试与链路信息 |

### `results` 字段说明

每一项代表一条命中的关系：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `string` | 关系 ID |
| `subject_id` | `string` | 主体实体 ID |
| `subject_name` | `string` | 主体实体名称 |
| `object_id` | `string` | 客体实体 ID |
| `object_name` | `string` | 客体实体名称 |
| `relation` | `string` | 关系文本，例如 `配乐` / `导演` |
| `passage` | `string` | 关系自带的原文证据 |
| `docment_id` | `string\|null` | 文档 ID，沿用当前表结构拼写 |
| `score` | `float` | 当前结果最终排序分数 |
| `score_breakdown` | `object` | 分数拆解，例如 `semantic_score` / `triple_score` / `hybrid_score` |
| `source_modes` | `array[string]` | 结果来源模式 |
| `matched_entity_ids` | `array[string]` | triple 路径中命中的 seed/扩展实体 |
| `passage_ids` | `array[string]` | 证据段落 ID 列表 |

### `entity_hits` 字段说明

每一项代表一个实体召回结果：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `string` | 实体 ID |
| `name` | `string` | 实体名称 |
| `score` | `float` | 实体召回相似度 |
| `relation_ids` | `array[string]` | 与该实体相连的关系 ID |

### `grounded_passages` 字段说明

每一项代表一个证据段落：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `string` | 段落 ID |
| `passage` | `string` | 段落正文 |
| `docment_id` | `string\|null` | 段落所属文档 ID |
| `score` | `float` | 继承自命中关系的最高分 |
| `matched_relation_ids` | `array[string]` | 命中该段落的关系 ID |

### `metadata` 字段说明

常用字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `took_ms` | `float` | 总耗时（毫秒） |
| `entity_top_k` | `int` | 实际使用的实体召回数量 |
| `relation_top_k` | `int` | 实际使用的关系召回数量 |
| `expansion_degree` | `int` | triple 路径扩展 hop 数 |
| `result_count` | `int` | 返回关系数量 |
| `grounded_passage_count` | `int` | 返回证据段落数量 |
| `semantic_candidate_count` | `int` | semantic 路径候选关系数 |
| `triple_candidate_count` | `int` | triple 路径候选关系数 |
| `embedder` | `string` | 当前使用的 embedding backend |
| `route` | `object` | 检索过程细节，例如 `query_entities` / `seed_entity_ids` / `expansion_history` |

---

## 三个接口的调用方法

## 1. Triple 接口

### URL

```http
POST /api/search/triple
```

### curl 示例

```bash
curl -X POST http://127.0.0.1:8000/api/search/triple \
  -H "Content-Type: application/json" \
  -d '{
    "query": "谁为星际穿越配乐？",
    "user_id": "test-user-001",
    "kb_id": "test-kb-cinema",
    "top_k": 3,
    "entity_top_k": 4,
    "relation_top_k": 4,
    "expansion_degree": 1
  }'
```

### 适用场景

- 更关心图谱关联链路
- 希望结合实体扩展做 multi-hop 检索
- 需要 `query_entities`、`expansion_history` 等链路信息

---

## 2. Semantic 接口

### URL

```http
POST /api/search/semantic
```

### curl 示例

```bash
curl -X POST http://127.0.0.1:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "谁为星际穿越配乐？",
    "user_id": "test-user-001",
    "kb_id": "test-kb-cinema",
    "top_k": 3,
    "entity_top_k": 4,
    "relation_top_k": 4
  }'
```

### 适用场景

- 希望直接依赖语义相似度匹配关系
- 不需要 triple 路径的子图扩展解释信息

---

## 3. Hybrid 接口

### URL

```http
POST /api/search/hybrid
```

### curl 示例

```bash
curl -X POST http://127.0.0.1:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "谁为星际穿越配乐？",
    "user_id": "test-user-001",
    "kb_id": "test-kb-cinema",
    "top_k": 3,
    "entity_top_k": 4,
    "relation_top_k": 4,
    "expansion_degree": 1
  }'
```

### 适用场景

- 同时利用 semantic 与 triple 两条路径
- 希望在纯语义匹配和图扩展之间取得平衡

---

## 通用入口 `/api/search`

如果你想只用一个入口，可以直接调用：

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "谁为星际穿越配乐？",
    "user_id": "test-user-001",
    "kb_id": "test-kb-cinema",
    "mode": "hybrid",
    "top_k": 3,
    "entity_top_k": 4,
    "relation_top_k": 4,
    "expansion_degree": 1
  }'
```

其中 `mode` 可选：
- `triple`
- `semantic`
- `hybrid`

---

## Demo 数据与验证

在仓库根目录执行：

```bash
python backend/scripts/demo_search.py
```

如果你的环境没有把仓库根目录加入 `PYTHONPATH`，可以使用：

```bash
PYTHONPATH=. python backend/scripts/demo_search.py
```

该脚本会：

1. 连接 Milvus（连接地址由环境变量决定）
2. 自动创建 `entities_demo`、`relations_demo`、`passages_demo` 集合（示例脚本默认不碰正式 `entities` / `relations` / `passages`）
3. 写入一组电影领域示例数据
4. 分别打印 `triple` / `semantic` / `hybrid` 检索结果

---

## 关键环境变量

- `NEXTGRAPH_MILVUS_URI`
- `NEXTGRAPH_MILVUS_DB`
- `NEXTGRAPH_ENTITIES_COLLECTION`
- `NEXTGRAPH_RELATIONS_COLLECTION`
- `NEXTGRAPH_PASSAGES_COLLECTION`
- `NEXTGRAPH_EMBEDDER_BACKEND`：`auto` / `azure_openai` / `vector_graph_rag`
- `NEXTGRAPH_EMBEDDING_MODEL`
- `NEXTGRAPH_AZURE_OPENAI_ENDPOINT`
- `NEXTGRAPH_AZURE_OPENAI_CHAT_DEPLOYMENT`（默认 `gpt-5.4-mini`，用于 query entity extraction）
- `NEXTGRAPH_AZURE_OPENAI_CHAT_MAX_COMPLETION_TOKENS`
- `API_KEY`（Azure OpenAI Key）
- `OPENAI_API_KEY`
- `NEXTGRAPH_OPENAI_BASE_URL`

`NEXTGRAPH_EMBEDDER_BACKEND=auto` 当前会直接走 Azure OpenAI `text-embedding-3-small`；不再保留本地 `hash` fallback。
