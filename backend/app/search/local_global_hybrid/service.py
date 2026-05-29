from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import Settings
from backend.app.search.local_global_hybrid.schemas import (
    EntityNeighborhoodRequest,
    EntityHit,
    GroundedPassage,
    RagAnswerRequest,
    RagAnswerResponse,
    RelationHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from backend.app.search.answer_generator import AnswerGenerator
from backend.app.search.query_fact_rewriter import IdentityQueryFactRewriter, QueryFactRewriter
from backend.app.search.query_entity_extractor import AzureLLMQueryEntityExtractor, QueryEntityExtractor
from backend.app.vector_database.search.embedder import QueryEmbedder
from backend.app.vector_database.search.repository import MilvusGraphRepository


@dataclass(slots=True)
class RouteResult:
    entity_hits: list[EntityHit] = field(default_factory=list)
    relation_candidates: dict[str, RelationHit] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TripleExpansionResult:
    relation_ids: list[str]
    relation_rows: list[dict[str, Any]]
    entity_rows: list[dict[str, Any]]
    expansion_history: list[dict[str, Any]]
    matched_entity_ids: dict[str, set[str]]


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: MilvusGraphRepository,
        embedder: QueryEmbedder,
        query_entity_extractor: QueryEntityExtractor | None = None,
        query_fact_rewriter: QueryFactRewriter | None = None,
        answer_generator: AnswerGenerator | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.embedder = embedder
        self.query_entity_extractor = query_entity_extractor or AzureLLMQueryEntityExtractor(settings)
        self.query_fact_rewriter = query_fact_rewriter or IdentityQueryFactRewriter()
        self.answer_generator = answer_generator

    @staticmethod
    def _filter_rows_by_score(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
        return [row for row in rows if float(row.get("score", 0.0)) >= threshold]

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        print(f"DEBUG: Search request for query: '{request.query}' in kb: '{request.kb_id}'")
        retrieval_query = self._rewrite_query_for_retrieval(request)
        query_vector = self.embedder.embed(retrieval_query)

        semantic_route: RouteResult | None = None
        triple_route: RouteResult | None = None

        if request.mode == SearchMode.semantic:
            semantic_route = self._semantic_route(query_vector=query_vector, request=request)
            results = list(semantic_route.relation_candidates.values())[: request.top_k]
            entity_hits = semantic_route.entity_hits
            route_metadata = semantic_route.metadata
        elif request.mode == SearchMode.triple:
            triple_route = self._triple_route(query_vector=query_vector, request=request)
            results = list(triple_route.relation_candidates.values())[: request.top_k]
            entity_hits = triple_route.entity_hits
            route_metadata = triple_route.metadata
        else:
            semantic_route = self._semantic_route(query_vector=query_vector, request=request)
            triple_route = self._triple_route(query_vector=query_vector, request=request)
            results = self._merge_hybrid(
                semantic_candidates=semantic_route.relation_candidates,
                triple_candidates=triple_route.relation_candidates,
                top_k=request.top_k,
            )
            entity_hits = semantic_route.entity_hits or triple_route.entity_hits
            route_metadata = {"semantic": semantic_route.metadata, "triple": triple_route.metadata}

        grounded_passages = self._ground_passages(results, request=request)
        took_ms = round((time.perf_counter() - started) * 1000, 3)
        print(f"DEBUG: Search finished. Results: {len(results)}, Entity hits: {len(entity_hits)}")
        return SearchResponse(
            mode=request.mode,
            query=request.query,
            user_id=request.user_id,
            kb_id=request.kb_id,
            results=results,
            entity_hits=entity_hits,
            grounded_passages=grounded_passages,
            metadata={
                "took_ms": took_ms,
                "entity_top_k": request.entity_top_k,
                "relation_top_k": request.relation_top_k,
                "expansion_degree": request.expansion_degree,
                "result_count": len(results),
                "grounded_passage_count": len(grounded_passages),
                "semantic_candidate_count": len(semantic_route.relation_candidates) if semantic_route else 0,
                "triple_candidate_count": len(triple_route.relation_candidates) if triple_route else 0,
                "hybrid_weights": {"semantic": self.settings.hybrid_semantic_weight, "triple": self.settings.hybrid_triple_weight},
                "thresholds": {"entity": self.settings.entity_score_threshold, "relation": self.settings.relation_score_threshold},
                "embedder": self.embedder.__class__.__name__,
                "retrieval_query": retrieval_query,
                "route": route_metadata,
            },
        )

    def answer(self, request: RagAnswerRequest) -> RagAnswerResponse:
        retrieval_response = self.search(
            SearchRequest(
                query=request.query,
                user_id=request.user_id,
                kb_id=request.kb_id,
                mode=request.mode,
                top_k=max(request.top_k, request.answer_top_k),
                entity_top_k=request.entity_top_k,
                relation_top_k=request.relation_top_k,
                expansion_degree=request.expansion_degree,
            )
        )
        selected_results = retrieval_response.results[: request.answer_top_k]
        selected_passages = retrieval_response.grounded_passages[: request.passage_top_k]
        retrieval_query = str(retrieval_response.metadata.get("retrieval_query") or request.query)
        context = self._build_rag_context(
            query=request.query,
            retrieval_query=retrieval_query,
            relation_hits=selected_results,
            grounded_passages=selected_passages,
        )
        answer_text = self._generate_answer(query=request.query, retrieval_query=retrieval_query, context=context)
        return RagAnswerResponse(
            mode=retrieval_response.mode,
            query=request.query,
            retrieval_query=retrieval_query,
            user_id=request.user_id,
            kb_id=request.kb_id,
            answer=answer_text,
            results=selected_results,
            grounded_passages=selected_passages,
            metadata={
                **retrieval_response.metadata,
                "answer_top_k": request.answer_top_k,
                "passage_top_k": request.passage_top_k,
                "context_preview": context[:2000],
            },
        )

    def search_trace(self, request: SearchRequest) -> dict[str, Any]:
        response = self.search(request)
        triple_route = response.metadata.get("route", {}).get("triple", {}) if request.mode == SearchMode.hybrid else response.metadata.get("route", {})
        semantic_route = response.metadata.get("route", {}).get("semantic", {}) if request.mode == SearchMode.hybrid else response.metadata.get("route", {})
        seed_graph = self._build_seed_graph(response.entity_hits)
        expanded_graph = triple_route.get("graph_snapshot") or {"nodes": [], "links": []}
        graph_snapshot = self._build_result_graph(response.results)
        steps = [
            {
                "id": "query",
                "title": "解析问题",
                "summary": f"解析查询并生成 embedding：{request.query}",
                "highlights": list(triple_route.get("query_entities") or semantic_route.get("query_entities") or []),
            },
            {
                "id": "seed",
                "title": "召回种子实体",
                "summary": f"命中 {len(response.entity_hits)} 个候选实体",
                "entity_ids": list(triple_route.get("seed_entity_ids") or semantic_route.get("seed_entity_ids") or []),
            },
            {
                "id": "expand",
                "title": "扩展关系子图",
                "summary": f"扩展到 {triple_route.get('expanded_relation_count', 0)} 条关系",
                "history": list(triple_route.get("expansion_history") or []),
            },
            {
                "id": "grounding",
                "title": "证据回溯",
                "summary": f"回查 {len(response.grounded_passages)} 段原文证据",
                "passage_ids": [item.id for item in response.grounded_passages],
            },
        ]
        return {
            "search": response.model_dump(),
            "trace": {
                "mode": response.mode,
                "steps": steps,
                "graph": graph_snapshot,
                "seed_graph": seed_graph,
                "expanded_graph": expanded_graph,
                "result_graph": graph_snapshot,
                "seed_entities": [item.model_dump() for item in response.entity_hits],
                "result_relations": [item.model_dump() for item in response.results],
                "grounded_passages": [item.model_dump() for item in response.grounded_passages],
            },
        }

    def entity_neighborhood(self, request: EntityNeighborhoodRequest) -> dict[str, Any]:
        seed_rows = self.repository.get_entities_by_ids(
            request.seed_entity_ids,
            user_id=request.user_id,
            kb_id=request.kb_id,
        )
        if not seed_rows:
            return {"graph": {"nodes": [], "links": []}, "center_entity_id": request.center_entity_id}

        entity_lookup = {row["id"]: dict(row) for row in seed_rows}
        center_entity_id = request.center_entity_id or request.seed_entity_ids[0]
        if center_entity_id not in entity_lookup and seed_rows:
            center_entity_id = seed_rows[0]["id"]

        visited_entity_ids: set[str] = set(entity_lookup)
        visible_entity_ids: set[str] = set(entity_lookup)
        sampled_relation_ids: set[str] = set()
        frontier_entity_ids: set[str] = {center_entity_id}

        for hop in range(request.depth):
            relation_ids: set[str] = set()
            for index, entity_id in enumerate(sorted(frontier_entity_ids)):
                entity = entity_lookup.get(entity_id)
                if not entity:
                    continue
                candidate_relation_ids = set(entity.get("relation_ids") or []) - sampled_relation_ids
                if not candidate_relation_ids:
                    continue
                relation_ids.update(
                    self._stable_sample_ids(
                        candidate_relation_ids,
                        shuffle_seed=request.shuffle_seed + hop * 997 + index * 131,
                        limit=request.relation_limit,
                    )
                )

            if not relation_ids:
                break
            relation_rows = self.repository.get_relations_by_ids(
                sorted(relation_ids),
                user_id=request.user_id,
                kb_id=request.kb_id,
            )

            next_frontier: set[str] = set()
            new_entity_ids: set[str] = set()
            for row in relation_rows:
                sampled_relation_ids.add(row["id"])
                for entity_id in [row.get("subject_id"), row.get("object_id")]:
                    if not entity_id:
                        continue
                    visible_entity_ids.add(entity_id)
                    if entity_id not in visited_entity_ids:
                        new_entity_ids.add(entity_id)
                        next_frontier.add(entity_id)

            if new_entity_ids:
                fetched_entities = self.repository.get_entities_by_ids(
                    sorted(new_entity_ids),
                    user_id=request.user_id,
                    kb_id=request.kb_id,
                )
                entity_lookup.update({row["id"]: row for row in fetched_entities})
                visited_entity_ids.update(new_entity_ids)

            frontier_entity_ids = next_frontier
            if not frontier_entity_ids:
                break

        closure_relation_ids: set[str] = set(sampled_relation_ids)
        for entity_id in visible_entity_ids:
            entity = entity_lookup.get(entity_id)
            if not entity:
                continue
            closure_relation_ids.update(entity.get("relation_ids") or [])

        closure_relation_rows = self.repository.get_relations_by_ids(
            sorted(closure_relation_ids),
            user_id=request.user_id,
            kb_id=request.kb_id,
        )
        visible_relation_rows = [
            row
            for row in closure_relation_rows
            if row.get("subject_id") in visible_entity_ids and row.get("object_id") in visible_entity_ids
        ]
        visible_entity_rows = [entity_lookup[entity_id] for entity_id in visible_entity_ids if entity_id in entity_lookup]

        graph = self._build_graph_snapshot(
            entity_rows=visible_entity_rows,
            relation_rows=visible_relation_rows,
            matched_entity_ids={},
        )
        seed_entity_ids = set(request.seed_entity_ids)
        for node in graph["nodes"]:
            node["is_seed"] = node.get("id") in seed_entity_ids

        return {"graph": graph, "center_entity_id": center_entity_id}

    def _extract_query_entities(self, request: SearchRequest) -> list[str]:
        entities = self.query_entity_extractor.extract(request.query, user_id=request.user_id, kb_id=request.kb_id, semantic_limit=request.entity_top_k)
        print(f"DEBUG: Extracted entities: {entities}")
        return entities

    def _rewrite_query_for_retrieval(self, request: SearchRequest) -> str:
        try:
            rewritten = self.query_fact_rewriter.rewrite(
                request.query,
                user_id=request.user_id,
                kb_id=request.kb_id,
            )
        except Exception as exc:
            print(f"DEBUG: Query rewrite failed, fallback to original query. Error: {exc}")
            return request.query
        return rewritten or request.query

    def _retrieve_seed_entities(self, *, request: SearchRequest, fallback_query_vector: list[float]) -> tuple[list[str], list[dict[str, Any]]]:
        query_entities = self._extract_query_entities(request)
        if not query_entities:
            print("DEBUG: No entities extracted, falling back to query embedding search.")
            rows = self.repository.search_entities(fallback_query_vector, user_id=request.user_id, kb_id=request.kb_id, limit=request.entity_top_k)
            filtered = self._filter_rows_by_score(rows, self.settings.entity_score_threshold)
            print(f"DEBUG: Fallback search found {len(rows)} rows, {len(filtered)} after threshold.")
            return [], filtered

        aggregated: dict[str, dict[str, Any]] = {}
        for query_entity in query_entities:
            rows = self.repository.search_entities(self.embedder.embed(query_entity), user_id=request.user_id, kb_id=request.kb_id, limit=request.entity_top_k)
            print(f"DEBUG: Entity search for '{query_entity}' found {len(rows)} rows.")
            for row in rows:
                entity_id = row["id"]
                if entity_id not in aggregated or float(row.get("score", 0.0)) > float(aggregated[entity_id].get("score", 0.0)):
                    aggregated[entity_id] = row

        sorted_rows = sorted(aggregated.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)[: request.entity_top_k]
        filtered = self._filter_rows_by_score(sorted_rows, self.settings.entity_score_threshold)
        print(f"DEBUG: Aggregated {len(aggregated)} rows, {len(filtered)} after threshold.")
        return query_entities, filtered

    def _semantic_route(self, *, query_vector: list[float], request: SearchRequest) -> RouteResult:
        query_entities, entity_rows = self._retrieve_seed_entities(request=request, fallback_query_vector=query_vector)
        relation_rows = self._filter_rows_by_score(
            self.repository.search_relations(query_vector, user_id=request.user_id, kb_id=request.kb_id, limit=request.relation_top_k),
            self.settings.relation_score_threshold,
        )
        entity_hits = [EntityHit(id=row["id"], name=row.get("name", ""), score=float(row.get("score", 0.0)), relation_ids=list(row.get("relation_ids") or [])) for row in entity_rows]
        relation_candidates = self._build_relation_hits(relation_rows, base_breakdown_key="semantic_score", source_mode="semantic")
        return RouteResult(entity_hits=entity_hits, relation_candidates=relation_candidates, metadata={"query_entities": query_entities, "seed_entity_ids": [row["id"] for row in entity_rows], "seed_relation_ids": [row["id"] for row in relation_rows]})

    def _triple_route(self, *, query_vector: list[float], request: SearchRequest) -> RouteResult:
        query_entities, seed_entity_rows = self._retrieve_seed_entities(request=request, fallback_query_vector=query_vector)
        seed_relation_rows = self._filter_rows_by_score(
            self.repository.search_relations(query_vector, user_id=request.user_id, kb_id=request.kb_id, limit=request.relation_top_k),
            self.settings.relation_score_threshold,
        )
        entity_hits = [EntityHit(id=row["id"], name=row.get("name", ""), score=float(row.get("score", 0.0)), relation_ids=list(row.get("relation_ids") or [])) for row in seed_entity_rows]
        expansion = self._expand_subgraph(seed_entity_rows=seed_entity_rows, seed_relation_rows=seed_relation_rows, request=request)
        if not expansion.relation_ids:
            return RouteResult(
                entity_hits=entity_hits,
                relation_candidates={},
                metadata={
                    "query_entities": query_entities,
                    "seed_entity_ids": [row["id"] for row in seed_entity_rows],
                    "seed_relation_ids": [row["id"] for row in seed_relation_rows],
                    "expanded_relation_count": 0,
                    "expanded_entity_count": len(expansion.entity_rows),
                    "expansion_history": expansion.expansion_history,
                    "graph_snapshot": self._build_graph_snapshot(
                        entity_rows=expansion.entity_rows,
                        relation_rows=expansion.relation_rows,
                        matched_entity_ids=expansion.matched_entity_ids,
                    ),
                },
            )

        rerank_limit = min(len(expansion.relation_ids), max(request.top_k, request.relation_top_k, self.settings.default_relation_top_k))
        reranked_rows = self._filter_rows_by_score(
            self.repository.search_relations(query_vector, user_id=request.user_id, kb_id=request.kb_id, limit=rerank_limit, relation_ids=expansion.relation_ids),
            self.settings.relation_score_threshold,
        )
        relation_candidates = self._build_relation_hits(reranked_rows, base_breakdown_key="triple_score", source_mode="triple", matched_entity_ids=expansion.matched_entity_ids)
        return RouteResult(
            entity_hits=entity_hits,
            relation_candidates=relation_candidates,
            metadata={
                "query_entities": query_entities,
                "seed_entity_ids": [row["id"] for row in seed_entity_rows],
                "seed_relation_ids": [row["id"] for row in seed_relation_rows],
                "expanded_relation_count": len(expansion.relation_ids),
                "expanded_entity_count": len(expansion.entity_rows),
                "expansion_history": expansion.expansion_history,
                "graph_snapshot": self._build_graph_snapshot(
                    entity_rows=expansion.entity_rows,
                    relation_rows=expansion.relation_rows,
                    matched_entity_ids=expansion.matched_entity_ids,
                ),
            },
        )

    def _expand_subgraph(self, *, seed_entity_rows: list[dict[str, Any]], seed_relation_rows: list[dict[str, Any]], request: SearchRequest) -> TripleExpansionResult:
        entity_lookup = {row["id"]: dict(row) for row in seed_entity_rows}
        relation_lookup = {row["id"]: dict(row) for row in seed_relation_rows}
        entity_ids = {row["id"] for row in seed_entity_rows}
        relation_ids = set(relation_lookup)
        matched_entity_ids: dict[str, set[str]] = {}
        history: list[dict[str, Any]] = []

        init_relation_ids: set[str] = set(relation_ids)
        for row in seed_entity_rows:
            for relation_id in row.get("relation_ids") or []:
                init_relation_ids.add(relation_id)
                matched_entity_ids.setdefault(relation_id, set()).add(row["id"])

        missing_init_relations = sorted(init_relation_ids - set(relation_lookup))
        if missing_init_relations:
            fetched_rows = self.repository.get_relations_by_ids(missing_init_relations, user_id=request.user_id, kb_id=request.kb_id)
            relation_lookup.update({row["id"]: row for row in fetched_rows})
        relation_ids = set(relation_lookup)
        history.append({"step": 0, "operation": "init_merge", "seed_entity_count": len(seed_entity_rows), "seed_relation_count": len(seed_relation_rows), "total_relations": len(relation_ids)})

        degree = request.expansion_degree or self.settings.default_expansion_degree
        for hop in range(degree):
            new_entity_ids: set[str] = set()
            for relation_id in list(relation_ids):
                relation = relation_lookup.get(relation_id)
                if not relation:
                    continue
                for entity_id in [relation.get("subject_id"), relation.get("object_id")]:
                    if entity_id and entity_id not in entity_ids:
                        new_entity_ids.add(entity_id)

            if new_entity_ids:
                fetched_entities = self.repository.get_entities_by_ids(sorted(new_entity_ids), user_id=request.user_id, kb_id=request.kb_id)
                entity_lookup.update({row["id"]: row for row in fetched_entities})
                entity_ids.update(new_entity_ids)

            new_relation_ids: set[str] = set()
            for entity_id in new_entity_ids:
                entity = entity_lookup.get(entity_id)
                if not entity:
                    continue
                for relation_id in entity.get("relation_ids") or []:
                    if relation_id not in relation_ids:
                        new_relation_ids.add(relation_id)
                    matched_entity_ids.setdefault(relation_id, set()).add(entity_id)

            if new_relation_ids:
                fetched_relations = self.repository.get_relations_by_ids(sorted(new_relation_ids), user_id=request.user_id, kb_id=request.kb_id)
                relation_lookup.update({row["id"]: row for row in fetched_relations})
                relation_ids.update(new_relation_ids)

            history.append({"step": hop + 1, "operation": f"expand_degree_{hop + 1}", "new_entity_ids": sorted(new_entity_ids), "new_relation_ids": sorted(new_relation_ids), "total_entities": len(entity_ids), "total_relations": len(relation_ids)})
            if not new_entity_ids and not new_relation_ids:
                break

        relation_rows = [relation_lookup[rid] for rid in sorted(relation_ids) if rid in relation_lookup]
        entity_rows = [entity_lookup[eid] for eid in sorted(entity_ids) if eid in entity_lookup]
        return TripleExpansionResult(relation_ids=sorted(relation_ids), relation_rows=relation_rows, entity_rows=entity_rows, expansion_history=history, matched_entity_ids=matched_entity_ids)

    def _build_relation_hits(self, relation_rows: list[dict[str, Any]], *, base_breakdown_key: str, source_mode: str, matched_entity_ids: dict[str, set[str]] | None = None) -> dict[str, RelationHit]:
        if not relation_rows:
            return {}

        entity_ids = sorted({entity_id for row in relation_rows for entity_id in [row.get("subject_id"), row.get("object_id")] if entity_id})
        scoped_user_id = relation_rows[0].get("user_id", "")
        scoped_kb_id = relation_rows[0].get("kb_id", "")
        entity_rows = self.repository.get_entities_by_ids(entity_ids, user_id=scoped_user_id, kb_id=scoped_kb_id)
        entity_lookup = {row["id"]: row for row in entity_rows}

        candidates: dict[str, RelationHit] = {}
        for row in relation_rows:
            relation_id = row["id"]
            score = float(row.get("score", 0.0))
            candidates[relation_id] = RelationHit(
                id=relation_id,
                subject_id=row.get("subject_id", ""),
                subject_name=entity_lookup.get(row.get("subject_id", ""), {}).get("name", row.get("subject_id", "")),
                object_id=row.get("object_id", ""),
                object_name=entity_lookup.get(row.get("object_id", ""), {}).get("name", row.get("object_id", "")),
                relation=row.get("relation", ""),
                describe=row.get("describe", "") or row.get("relation", ""),
                passage=row.get("passage", ""),
                docment_id=row.get("docment_id"),
                score=score,
                score_breakdown={base_breakdown_key: score},
                source_modes=[source_mode],
                matched_entity_ids=sorted((matched_entity_ids or {}).get(relation_id, set())),
                passage_ids=list(row.get("passage_ids") or []),
            )
        return dict(sorted(candidates.items(), key=lambda item: item[1].score, reverse=True))

    def _ground_passages(self, relation_hits: list[RelationHit], *, request: SearchRequest) -> list[GroundedPassage]:
        if not relation_hits:
            return []

        passage_scores: dict[str, float] = {}
        passage_to_relations: dict[str, set[str]] = {}
        for hit in relation_hits:
            for passage_id in hit.passage_ids:
                passage_scores[passage_id] = max(passage_scores.get(passage_id, float("-inf")), hit.score)
                passage_to_relations.setdefault(passage_id, set()).add(hit.id)

        passage_rows = self.repository.get_passages_by_ids(sorted(passage_scores), user_id=request.user_id, kb_id=request.kb_id)
        grounded = [GroundedPassage(id=row["id"], passage=row.get("passage", ""), docment_id=row.get("docment_id"), score=float(passage_scores.get(row["id"], 0.0)), matched_relation_ids=sorted(passage_to_relations.get(row["id"], set()))) for row in passage_rows]
        grounded.sort(key=lambda item: item.score, reverse=True)
        return grounded[: request.top_k]

    def _build_graph_snapshot(
        self,
        *,
        entity_rows: list[dict[str, Any]],
        relation_rows: list[dict[str, Any]],
        matched_entity_ids: dict[str, set[str]],
    ) -> dict[str, Any]:
        entity_lookup = {row["id"]: row for row in entity_rows}
        nodes: dict[str, dict[str, Any]] = {}
        links: list[dict[str, Any]] = []
        for row in relation_rows:
            subject_id = row.get("subject_id")
            object_id = row.get("object_id")
            if not subject_id or not object_id:
                continue
            subject = entity_lookup.get(subject_id, {})
            object_ = entity_lookup.get(object_id, {})
            nodes.setdefault(
                subject_id,
                {
                    "id": subject_id,
                    "label": subject.get("name", subject_id),
                    "kind": "entity",
                    "is_seed": any(subject_id in values for values in matched_entity_ids.values()),
                },
            )
            nodes.setdefault(
                object_id,
                {
                    "id": object_id,
                    "label": object_.get("name", object_id),
                    "kind": "entity",
                    "is_seed": any(object_id in values for values in matched_entity_ids.values()),
                },
            )
            links.append(
                {
                    "id": row.get("id"),
                    "source": subject_id,
                    "target": object_id,
                    "label": row.get("relation", "related_to"),
                    "matched_entity_ids": sorted(matched_entity_ids.get(row.get("id", ""), set())),
                }
            )
        return {"nodes": list(nodes.values()), "links": links}

    def _build_result_graph(self, relation_hits: list[RelationHit]) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        links: list[dict[str, Any]] = []
        for hit in relation_hits:
            nodes.setdefault(hit.subject_id, {"id": hit.subject_id, "label": hit.subject_name, "kind": "entity", "is_seed": bool(hit.matched_entity_ids)})
            nodes.setdefault(hit.object_id, {"id": hit.object_id, "label": hit.object_name, "kind": "entity", "is_seed": bool(hit.matched_entity_ids)})
            links.append(
                {
                    "id": hit.id,
                    "source": hit.subject_id,
                    "target": hit.object_id,
                    "label": hit.relation,
                    "matched_entity_ids": hit.matched_entity_ids,
                }
            )
        return {"nodes": list(nodes.values()), "links": links}

    @staticmethod
    def _stable_sample_ids(ids: set[str], *, shuffle_seed: int, limit: int) -> list[str]:
        ranked: list[tuple[int, str]] = []
        for index, value in enumerate(ids):
            score = shuffle_seed * 131 + index * 17
            for char in value:
                score = (score * 33 + ord(char)) & 0xFFFFFFFF
            ranked.append((score, value))
        ranked.sort(key=lambda item: item[0])
        return [value for _, value in ranked[:limit]]

    def _build_seed_graph(self, entity_hits: list[EntityHit]) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": hit.id,
                    "label": hit.name or hit.id,
                    "kind": "entity",
                    "is_seed": True,
                }
                for hit in entity_hits
            ],
            "links": [],
        }

    def _merge_hybrid(self, *, semantic_candidates: dict[str, RelationHit], triple_candidates: dict[str, RelationHit], top_k: int) -> list[RelationHit]:
        merged: list[RelationHit] = []
        all_ids = set(semantic_candidates) | set(triple_candidates)
        for relation_id in all_ids:
            semantic_hit = semantic_candidates.get(relation_id)
            triple_hit = triple_candidates.get(relation_id)
            base_hit = semantic_hit or triple_hit
            assert base_hit is not None
            semantic_score = semantic_hit.score if semantic_hit else 0.0
            triple_score = triple_hit.score if triple_hit else 0.0
            hybrid_score = semantic_score * self.settings.hybrid_semantic_weight + triple_score * self.settings.hybrid_triple_weight
            merged.append(RelationHit(
                id=base_hit.id,
                subject_id=base_hit.subject_id,
                subject_name=base_hit.subject_name,
                object_id=base_hit.object_id,
                object_name=base_hit.object_name,
                relation=base_hit.relation,
                describe=base_hit.describe,
                passage=base_hit.passage,
                docment_id=base_hit.docment_id,
                score=hybrid_score,
                score_breakdown={"semantic_score": semantic_score, "triple_score": triple_score, "hybrid_score": hybrid_score},
                source_modes=[*(["semantic"] if semantic_hit else []), *(["triple"] if triple_hit else [])],
                matched_entity_ids=sorted(set((semantic_hit.matched_entity_ids if semantic_hit else [])) | set((triple_hit.matched_entity_ids if triple_hit else []))),
                passage_ids=base_hit.passage_ids,
            ))
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[:top_k]

    def _build_rag_context(
        self,
        *,
        query: str,
        retrieval_query: str,
        relation_hits: list[RelationHit],
        grounded_passages: list[GroundedPassage],
    ) -> str:
        relation_lines: list[str] = []
        for index, hit in enumerate(relation_hits, start=1):
            relation_lines.append(
                "\n".join(
                    [
                        f"[Relation {index}]",
                        f"triple: {hit.subject_name} --({hit.relation})-> {hit.object_name}",
                        f"describe: {hit.describe}",
                        f"score: {hit.score:.4f}",
                    ]
                )
            )

        passage_lines: list[str] = []
        for index, passage in enumerate(grounded_passages, start=1):
            passage_lines.append(
                "\n".join(
                    [
                        f"[Passage {index}]",
                        f"id: {passage.id}",
                        f"docment_id: {passage.docment_id or ''}",
                        f"text: {passage.passage}",
                    ]
                )
            )

        return (
            f"Question:\n{query}\n\n"
            f"Retrieval Query:\n{retrieval_query}\n\n"
            "Knowledge Graph Relation Summaries:\n"
            f"{chr(10).join(relation_lines) if relation_lines else 'None'}\n\n"
            "Grounded Evidence Passages:\n"
            f"{chr(10).join(passage_lines) if passage_lines else 'None'}"
        )

    def _generate_answer(self, *, query: str, retrieval_query: str, context: str) -> str:
        if not self.answer_generator:
            if "None" in context and "Grounded Evidence Passages:\nNone" in context:
                return "知识库中没有检索到足够证据，暂时无法回答这个问题。"
            return context
        try:
            answer = self.answer_generator.generate(
                query=query,
                retrieval_query=retrieval_query,
                context=context,
            )
        except Exception as exc:
            print(f"DEBUG: Answer generation failed, fallback to context summary. Error: {exc}")
            return "知识库已检索到相关材料，但答案生成失败。请先查看返回的关系摘要和证据原文。"
        return answer or "知识库已检索到相关材料，但未生成有效答案。"
