# Entity Graph API 输出格式

本文档描述接口 `POST /api/entity-graph/extract-and-expand` 的响应结构。

协议版本：

```text
nextgraph.entity-graph.v1
```

## 顶层响应结构

```json
{
  "protocol": "nextgraph.entity-graph.v1",
  "question": "主角和谁有冲突，老师和家人分别怎么影响他？",
  "user_id": "admin_user",
  "kb_id": "0616",
  "extracted_entities": ["主角", "老师", "家人"],
  "entity_graphs": [],
  "merged_graph": {
    "nodes": [],
    "links": []
  },
  "metadata": {}
}
```

## 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `protocol` | `string` | 返回协议版本。当前固定为 `nextgraph.entity-graph.v1`。 |
| `question` | `string` | 原始查询问题。 |
| `user_id` | `string` | 用户作用域。 |
| `kb_id` | `string` | 知识库作用域。 |
| `extracted_entities` | `string[]` | 从问题中抽取出的实体词列表。 |
| `entity_graphs` | `EntityGraph[]` | 每个抽取实体词对应的扩图结果。 |
| `merged_graph` | `GraphPayload` | 将所有 `entity_graphs[].graph` 合并后的总图。 |
| `metadata` | `object` | 本次请求的执行摘要和统计信息。 |

## `entity_graphs` 结构

```json
{
  "entity_text": "主角",
  "depth": 3,
  "relation_limit": 12,
  "center_entity_id": "e_hero_001",
  "matched_entities": [
    {
      "id": "e_hero_001",
      "name": "主角",
      "relation_ids": ["r_101", "r_102", "r_103"],
      "match_mode": "name_exact",
      "source_entity_text": "主角"
    }
  ],
  "graph": {
    "nodes": [],
    "links": []
  },
  "metadata": {
    "match_strategy": "name_exact",
    "matched_entity_count": 1,
    "seed_entity_ids": ["e_hero_001"]
  }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `entity_text` | `string` | 从问题里抽取出的实体词。 |
| `depth` | `int` | 该实体实际使用的扩展跳数。 |
| `relation_limit` | `int` | 该实体每一跳最多扩展的关系数量。 |
| `center_entity_id` | `string \| null` | 本次图扩展使用的中心实体 ID。 |
| `matched_entities` | `MatchedEntity[]` | 该实体词映射到数据库中的实体列表。 |
| `graph` | `GraphPayload` | 该实体对应的子图。 |
| `metadata` | `object` | 该实体扩图过程的摘要。 |

## `matched_entities` 结构

```json
{
  "id": "e_hero_001",
  "name": "主角",
  "relation_ids": ["r_101", "r_102", "r_103"],
  "match_mode": "name_exact",
  "source_entity_text": "主角"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `string` | 数据库中的实体 ID。 |
| `name` | `string` | 数据库中的实体名称。 |
| `relation_ids` | `string[]` | 该实体关联的关系 ID 列表。 |
| `match_mode` | `string` | 匹配方式，当前支持 `name_exact`、`vector_fallback`。 |
| `source_entity_text` | `string` | 触发本次匹配的原始抽取实体词。 |

## `graph` 结构

```json
{
  "nodes": [
    {
      "id": "e_hero_001",
      "label": "主角",
      "kind": "entity",
      "is_seed": true
    }
  ],
  "links": [
    {
      "id": "r_101",
      "source": "e_hero_001",
      "target": "e_teacher_003",
      "label": "被指导",
      "matched_entity_ids": ["e_hero_001"]
    }
  ]
}
```

### `nodes`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `string` | 实体 ID。 |
| `label` | `string` | 实体显示名称。 |
| `kind` | `string` | 节点类型，当前固定为 `entity`。 |
| `is_seed` | `bool` | 是否属于本次查询命中的种子实体。 |

### `links`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `string` | 关系 ID。 |
| `source` | `string` | 起点实体 ID。 |
| `target` | `string` | 终点实体 ID。 |
| `label` | `string` | 关系名称。 |
| `matched_entity_ids` | `string[]` | 这条关系和哪些种子实体有关。 |

## `metadata` 结构

顶层 `metadata` 当前包含以下字段：

```json
{
  "max_entities": 3,
  "name_match_limit": 3,
  "fallback_vector_limit": 3,
  "default_depth": 2,
  "default_relation_limit": 8,
  "returned_entity_graph_count": 2,
  "merged_node_count": 3,
  "merged_link_count": 2
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `max_entities` | `int` | 最多抽取多少个实体词。 |
| `name_match_limit` | `int` | 每个实体词按名称最多匹配多少个数据库实体。 |
| `fallback_vector_limit` | `int` | 当名称匹配不到时，向量兜底最多返回多少个实体。 |
| `default_depth` | `int` | 默认扩展跳数。 |
| `default_relation_limit` | `int` | 默认每跳最多扩展多少条关系。 |
| `returned_entity_graph_count` | `int` | 本次响应返回了多少个实体子图。 |
| `merged_node_count` | `int` | 合并总图中的节点数量。 |
| `merged_link_count` | `int` | 合并总图中的边数量。 |

## 完整返回示例

```json
{
  "protocol": "nextgraph.entity-graph.v1",
  "question": "主角和谁有冲突，老师和家人分别怎么影响他？",
  "user_id": "admin_user",
  "kb_id": "0616",
  "extracted_entities": ["主角", "老师", "家人"],
  "entity_graphs": [
    {
      "entity_text": "主角",
      "depth": 3,
      "relation_limit": 12,
      "center_entity_id": "e_hero_001",
      "matched_entities": [
        {
          "id": "e_hero_001",
          "name": "主角",
          "relation_ids": ["r_101", "r_102", "r_103"],
          "match_mode": "name_exact",
          "source_entity_text": "主角"
        }
      ],
      "graph": {
        "nodes": [
          {
            "id": "e_hero_001",
            "label": "主角",
            "kind": "entity",
            "is_seed": true
          },
          {
            "id": "e_teacher_003",
            "label": "老师",
            "kind": "entity",
            "is_seed": false
          },
          {
            "id": "e_family_008",
            "label": "家人",
            "kind": "entity",
            "is_seed": false
          }
        ],
        "links": [
          {
            "id": "r_101",
            "source": "e_hero_001",
            "target": "e_teacher_003",
            "label": "被指导",
            "matched_entity_ids": ["e_hero_001"]
          },
          {
            "id": "r_102",
            "source": "e_family_008",
            "target": "e_hero_001",
            "label": "施加压力",
            "matched_entity_ids": ["e_hero_001"]
          }
        ]
      },
      "metadata": {
        "match_strategy": "name_exact",
        "matched_entity_count": 1,
        "seed_entity_ids": ["e_hero_001"]
      }
    },
    {
      "entity_text": "老师",
      "depth": 1,
      "relation_limit": 5,
      "center_entity_id": "e_teacher_003",
      "matched_entities": [
        {
          "id": "e_teacher_003",
          "name": "老师",
          "relation_ids": ["r_101", "r_201"],
          "match_mode": "name_exact",
          "source_entity_text": "老师"
        }
      ],
      "graph": {
        "nodes": [
          {
            "id": "e_teacher_003",
            "label": "老师",
            "kind": "entity",
            "is_seed": true
          },
          {
            "id": "e_hero_001",
            "label": "主角",
            "kind": "entity",
            "is_seed": false
          }
        ],
        "links": [
          {
            "id": "r_101",
            "source": "e_hero_001",
            "target": "e_teacher_003",
            "label": "被指导",
            "matched_entity_ids": ["e_teacher_003"]
          }
        ]
      },
      "metadata": {
        "match_strategy": "name_exact",
        "matched_entity_count": 1,
        "seed_entity_ids": ["e_teacher_003"]
      }
    }
  ],
  "merged_graph": {
    "nodes": [
      {
        "id": "e_hero_001",
        "label": "主角",
        "kind": "entity",
        "is_seed": true
      },
      {
        "id": "e_teacher_003",
        "label": "老师",
        "kind": "entity",
        "is_seed": true
      },
      {
        "id": "e_family_008",
        "label": "家人",
        "kind": "entity",
        "is_seed": false
      }
    ],
    "links": [
      {
        "id": "r_101",
        "source": "e_hero_001",
        "target": "e_teacher_003",
        "label": "被指导",
        "matched_entity_ids": ["e_hero_001", "e_teacher_003"]
      },
      {
        "id": "r_102",
        "source": "e_family_008",
        "target": "e_hero_001",
        "label": "施加压力",
        "matched_entity_ids": ["e_hero_001"]
      }
    ]
  },
  "metadata": {
    "max_entities": 3,
    "name_match_limit": 3,
    "fallback_vector_limit": 3,
    "default_depth": 2,
    "default_relation_limit": 8,
    "returned_entity_graph_count": 2,
    "merged_node_count": 3,
    "merged_link_count": 2
  }
}
```

## 前端使用建议

- 如果只是画总图，直接使用 `merged_graph.nodes` 和 `merged_graph.links`。
- 如果要做“按抽取实体切换子图”的交互，使用 `entity_graphs`。
- 如果要展示匹配命中来源，可读取 `matched_entities[].match_mode`。
