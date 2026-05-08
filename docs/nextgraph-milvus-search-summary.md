# NextGraph 检索能力实现汇报

## 1. 实现目标

本次工作的目标是，在 **Milvus + 现有知识图谱表结构** 上，完成一套可实际运行的检索后端，支持：

1. **Triple 检索**
2. **Semantic 检索**
3. **Hybrid 检索**
4. **Query Entity Extraction（查询实体抽取）**
5. **Passage Grounding（证据段落回溯）**

同时要求：

- 检索严格按 `user_id + kb_id` 做作用域隔离
- 兼容当前真实表结构
- 可以直接写入 Milvus 测试数据并验证
- 返回结果可解释，可附带原文证据

---

## 2. 当前使用的真实表结构

### Entities
- `user_id`
- `kb_id`
- `id`
- `name`
- `embedding`
- `relation_ids`

### Relations
- `user_id`
- `kb_id`
- `id`
- `subject_id`
- `object_id`
- `relation`
- `passage`
- `embedding`
- `passage_ids`
- `docment_id`

### Passage
- `user_id`
- `kb_id`
- `id`
- `embedding`
- `passage`
- `docment_id`

> 说明：`docment_id` 字段名按当前数据库契约保留，没有擅自改成 `document_id`。

---

## 3. 总体方法设计

整体检索链路分成三部分：

### 3.1 Query Entity Extraction
先从用户问题里抽出检索锚点实体，例如：

- 问题：`谁为星际穿越配乐？`
- 抽取结果：`["星际穿越"]` 或 `["星际穿越", "汉斯·季默"]`

这一步不是本地规则，而是通过 **Azure OpenAI `gpt-5.4-mini`** 完成。

### 3.2 三种检索路径

#### A. Triple 检索
Triple 检索强调“图上的关系扩展”，流程如下：

1. 对 query 做 **实体抽取**
2. 用抽取出的实体去 `Entities` 里做 **entity seed retrieval**
3. 同时用 query 去 `Relations` 做 **relation seed retrieval**
4. 基于：
   - `entity.relation_ids`
   - `relation.subject_id`
   - `relation.object_id`
   做 **subgraph expansion（子图扩展）**
5. 在扩展后的关系子图中再做一次**受限向量排序**
6. 输出关系结果，并根据 `passage_ids` 回查 `Passage` 表，返回证据段落

#### B. Semantic 检索
Semantic 检索强调“语义相似度直接召回”，流程如下：

1. 对 query 做 **实体抽取**
2. 对抽取出的实体做 entity seed retrieval
3. 同时直接对 `Relations.embedding` 做 **relation 向量检索**
4. 输出关系结果
5. 回查 `Passage` 表，返回 grounding 证据

#### C. Hybrid 检索
Hybrid 检索同时使用上面两条路径：

- 先独立跑 `semantic` 与 `triple`
- 再把两边候选关系按权重做融合排序

当前融合权重：
- `semantic = 0.55`
- `triple = 0.45`

### 3.3 Passage Grounding
所有最终命中的关系结果，都会进一步通过 `passage_ids` 去 `Passage` 表回查真实原文段落，返回：

- 段落 ID
- 段落正文
- 文档 ID
- 命中的关系 ID
- 继承关系分数后的段落分数

这样可以保证结果不仅有“关系结论”，还有“原文证据”。

---

## 4. 关键实现方式

## 4.1 Embedding 与 LLM 配置

### Query Embedding
使用 **Azure OpenAI `text-embedding-3-small`**

### Query Entity Extraction
使用 **Azure OpenAI `gpt-5.4-mini` chat completion**

作用：
- 将 query 中适合图谱检索的实体抽出来
- 提高 entity seed retrieval 的准确性

## 4.2 检索作用域控制
所有查询都带上：

- `user_id`
- `kb_id`

在 Milvus 层统一生成 filter：

```text
user_id == ... and kb_id == ...
```

这样保证：

- 不同用户之间隔离
- 同一用户下不同知识库之间也隔离

## 4.3 Triple 子图扩展
Triple 路径参考了 `/opt/vector-graph-rag` 的检索思想，但按照当前 NextGraph 表结构落地。

扩展逻辑：

1. 从 seed entities 收集 `relation_ids`
2. 与 seed relations 合并，形成初始关系集合
3. 对每条 relation，读取：
   - `subject_id`
   - `object_id`
4. 发现新的实体后，再根据实体的 `relation_ids` 继续扩展
5. 形成扩展后的候选关系子图

这一步相当于在 Milvus 里模拟“图扩展”，不需要额外图数据库。

## 4.4 噪声过滤 / 空结果策略
在扩展测试中发现一个问题：

> 当 query 和当前知识库完全无关时，系统会返回低分噪声结果。

例如：
- 在实验室知识库里问：`谁为星际穿越配乐？`

本来应该返回空，但早期版本会误召回实验室数据。

为此增加了阈值过滤：

- `entity_score_threshold = 0.2`
- `relation_score_threshold = 0.3`

作用：
- 过滤低分实体 seed
- 过滤低分关系候选
- 对明显无关 query 返回空结果，而不是误召回噪声

---

## 5. 代码实现落点

### 检索服务
- `backend/app/search/local_global_hybrid/service.py`

负责：
- query entity extraction 接入
- triple / semantic / hybrid 三条路径
- subgraph expansion
- passage grounding
- 最终返回结构拼装

### 查询实体抽取
- `backend/app/search/query_entity_extractor.py`

负责：
- 通过 Azure OpenAI `gpt-5.4-mini` 提取 query entities

### Milvus 访问层
- `backend/app/vector_database/search/repository.py`

负责：
- 搜索 `Entities`
- 搜索 `Relations`
- 根据 ID 查询 `Entities`
- 根据 ID 查询 `Relations`
- 根据 ID 查询 `Passage`
- 创建测试 collection
- 插入测试数据

### 配置
- `backend/app/core/config.py`

负责：
- Milvus URI / DB
- collection 名
- Azure OpenAI endpoint / key / deployment
- 检索参数
- 阈值参数

### API 入口
- `backend/app/main.py`
- `backend/app/search/local_global_hybrid/api.py`

提供接口：

- `POST /api/search`
- `POST /api/search/triple`
- `POST /api/search/semantic`
- `POST /api/search/hybrid`

### 文档与测试
- `backend/README.md`
- `backend/tests/test_search_service.py`
- `backend/tests/test_api.py`
- `backend/scripts/demo_search.py`

---

## 6. 返回结果设计

统一返回：

- `results`：关系检索结果
- `entity_hits`：实体 seed 命中
- `grounded_passages`：证据段落
- `metadata`：检索链路信息

### 关系结果 `results`
每条关系包含：
- `subject_id`
- `subject_name`
- `object_id`
- `object_name`
- `relation`
- `passage`
- `docment_id`
- `score`
- `source_modes`
- `matched_entity_ids`
- `passage_ids`

### 证据段落 `grounded_passages`
每条证据包含：
- `id`
- `passage`
- `docment_id`
- `score`
- `matched_relation_ids`

### metadata
包含：
- `query_entities`
- `seed_entity_ids`
- `seed_relation_ids`
- `expanded_relation_count`
- `expanded_entity_count`
- `expansion_history`
- `semantic_candidate_count`
- `triple_candidate_count`
- `took_ms`

---

## 7. 测试数据设计

本次不是只用单一例子，而是扩成了多类知识库。

### 7.1 电影知识库
作用域：
- `demo-user-001 / demo-kb-movie`

覆盖关系：
- 导演
- 配乐
- 发行
- 主演
- 合作

### 7.2 实验室知识库
作用域：
- `demo-user-001 / demo-kb-lab`

覆盖关系：
- 认识
- 合作
- 指导
- 构建

这个库主要用来测：
- 英文实体
- 人物关系
- 多跳扩展
- 同用户不同知识库隔离

### 7.3 历史知识库
作用域：
- `demo-user-002 / demo-kb-history`

覆盖关系：
- 统一
- 都城

这个库主要用来测：
- 中文历史实体
- 不同用户隔离

---

## 8. 真实测试结果

## 8.1 电影库：配乐问题
问题：
- `谁为星际穿越配乐？`

三种模式 Top1 一致：

- `rel_music_interstellar`
- `relation = 配乐`
- `subject = 汉斯·季默`
- `object = 星际穿越`

证据段落：
- `《星际穿越》的配乐由汉斯·季默创作。`

## 8.2 电影库：导演问题
问题：
- `谁导演了盗梦空间？`

三种模式 Top1 一致：

- `rel_directed_inception`
- `relation = 导演`
- `subject = 克里斯托弗·诺兰`
- `object = 盗梦空间`

证据段落：
- `克里斯托弗·诺兰还导演了《盗梦空间》。`

## 8.3 电影库：合作问题
问题：
- `诺兰和谁合作过？`

三种模式 Top1 一致：

- `rel_collab_hans`
- `relation = 合作`
- `subject = 克里斯托弗·诺兰`
- `object = 汉斯·季默`

证据段落：
- `克里斯托弗·诺兰曾与汉斯·季默多次合作。`

## 8.4 实验室库：指导问题
问题：
- `谁指导了 Alice？`

三种模式 Top1 一致：

- `rel_carol_mentors_alice`
- `relation = 指导`
- `subject = Carol`
- `object = Alice`

证据段落：
- `Carol mentors Alice on graph retrieval experiments.`

## 8.5 实验室库：合作问题
问题：
- `Bob 和谁合作？`

三种模式 Top1 一致：

- `rel_bob_works_carol`
- `relation = 合作`
- `subject = Bob`
- `object = Carol`

证据段落：
- `Bob works with Carol on the vision pipeline.`

## 8.6 历史库：统一问题
问题：
- `谁统一了中国？`

三种模式 Top1 一致：

- `rel_qin_unify_china`
- `relation = 统一`
- `subject = 秦始皇`
- `object = 中国`

证据段落：
- `秦始皇完成了对中国的统一。`

## 8.7 同用户不同知识库隔离测试
问题：
- 在实验室库里问：`谁为星际穿越配乐？`

修复前：
- 会误召回实验室低分噪声数据

修复后：
- `result_count = 0`
- 三种模式都返回空结果

说明：
- **同用户不同知识库隔离已生效**
- **低分噪声误召回问题已解决**

---

## 9. 单元测试与真实验证

### 单元测试
执行：
```bash
pytest backend/tests -q
```

结果：
- `5 passed`

### 真实 Milvus 写入 / 读取验证
真实创建并写入测试 collection：

- `entities_ng_test`
- `relations_ng_test`
- `passages_ng_test`

并完成了：
- 写入验证
- 检索验证
- API 调用验证
- passage grounding 验证

---

## 10. 本次实现中发现并修复的问题

### 问题 1：无关 query 在错误知识库中误召回
- 已通过实体/关系阈值过滤修复

### 问题 2：早期模式执行有冗余
- 已修复为：
  - semantic 只跑 semantic
  - triple 只跑 triple
  - hybrid 才同时跑两条路

---

## 11. 当前实现的特点

### 已完成
- Milvus 两层实体/关系检索
- Triple 路径图扩展
- Query entity extraction
- Passage grounding
- Scope 隔离
- 空结果策略
- 多数据类型测试

### 还未完成
- Relation 的 LLM rerank
- 最终 answer generation
- Query entity extraction 缓存

---

## 12. 最终结论

本次已经完成了一套**可运行、可解释、可验证**的检索后端能力，具备以下特点：

1. **支持 triple / semantic / hybrid 三种检索模式**
2. **支持 Azure OpenAI 驱动的 query entity extraction**
3. **支持基于 Passage 表的证据回溯**
4. **支持 user_id + kb_id 的严格知识库隔离**
5. **已通过多知识库、多问题类型、真实 Milvus 写入读取验证**
6. **已修复错误知识库误召回的问题**

整体上，这套实现已经可以作为 NextGraph 当前阶段的**后端检索基础能力**，并可继续向更完整的 Graph RAG 方向扩展。
