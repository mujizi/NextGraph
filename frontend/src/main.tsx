import React from "react";
import { createRoot } from "react-dom/client";
import ForceGraph2D from "react-force-graph-2d";
import { forceCollide, forceManyBody, forceX, forceY } from "d3-force-3d";
import { BookOpen, Film, Network, Search, SlidersHorizontal } from "lucide-react";
import appLogo from "./assets/app-logo.png";
import "./styles.css";

type GraphNode = {
  id: string;
  label?: string;
  name?: string;
  kind?: string;
  is_seed?: boolean;
  is_cluster_center?: boolean;
  cluster_ids?: string[];
  primary_cluster_id?: string | null;
  cluster_target_x?: number;
  cluster_target_y?: number;
  color?: string;
};

type GraphLink = {
  id: string;
  source: string;
  target: string;
  label: string;
  describe?: string;
  matched_entity_ids?: string[];
  cluster_ids?: string[];
  primary_cluster_id?: string | null;
  color?: string;
};

type RelationResult = {
  id: string;
  subject_id: string;
  subject_name: string;
  object_id: string;
  object_name: string;
  relation: string;
  describe?: string;
  passage: string;
  docment_id?: string;
  score: number;
  score_breakdown?: Record<string, number>;
  source_modes: string[];
  matched_entity_ids: string[];
  passage_ids?: string[];
};

type GroundedPassage = {
  id: string;
  passage: string;
  docment_id?: string;
  score: number;
  matched_relation_ids?: string[];
};

type QueryApiResponse = {
  question: string;
  user_id: string;
  kb_id: string;
  mode: string;
  answer?: string | null;
  retrieval_query?: string | null;
  results: RelationResult[];
  grounded_passages: GroundedPassage[];
  entity_hits?: Array<{
    id: string;
    name: string;
    score: number;
    relation_ids?: string[];
  }>;
  metadata?: Record<string, unknown>;
};

type MatchedEntity = {
  id: string;
  name: string;
  relation_ids?: string[];
  match_mode: string;
  source_entity_text: string;
};

type ExtractedEntityGraph = {
  entity_text: string;
  depth: number;
  relation_limit: number;
  center_entity_id?: string | null;
  matched_entities: MatchedEntity[];
  graph: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  metadata?: Record<string, unknown>;
};

type EntityGraphApiResponse = {
  protocol: string;
  question: string;
  user_id: string;
  kb_id: string;
  extracted_entities: string[];
  entity_graphs: ExtractedEntityGraph[];
  merged_graph: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  metadata?: Record<string, unknown>;
};

type GraphNodeDatum = GraphNode & {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
};

type GraphLinkDatum = GraphLink & {
  source: string | GraphNodeDatum;
  target: string | GraphNodeDatum;
};

type HoveredRelationCard = {
  relation: GraphLink;
  x: number;
  y: number;
};

const DEFAULT_USER_ID = "admin_user";
const DEFAULT_KB_ID = import.meta.env.VITE_DEFAULT_KB_ID ?? "0616";
const FIXED_ENTITY_GRAPH_DEPTH = 4;
const FIXED_ENTITY_GRAPH_RELATION_LIMIT = 30;
const QUERY_TOP_K = 5;
const QUERY_ENTITY_TOP_K = 2;
const QUERY_RELATION_TOP_K = 5;
const FIXED_HYBRID_EXPANSION_DEGREE = 0;
const ANSWER_TOP_K = 5;
const PASSAGE_TOP_K = 1;
const ANSWER_TOKEN_INTERVAL_MS = 18;
const MAX_RENDER_NODES = 120;
const MAX_RENDER_LINKS = 180;
const RELATION_TOOLTIP_DELAY_MS = 700;
function resolveApiRoot(configuredRoot: string, fallbackRoot = "") {
  const trimmedRoot = configuredRoot.replace(/\/$/, "");
  if (!trimmedRoot) {
    return fallbackRoot;
  }
  if (typeof window === "undefined") {
    return trimmedRoot;
  }

  try {
    const url = new URL(trimmedRoot);
    if (url.hostname === "127.0.0.1" || url.hostname === "localhost") {
      // In dev, browser-side 127.0.0.1 means the user's laptop, not the remote
      // server. Keep requests same-origin so Vite proxies them server-side.
      return fallbackRoot;
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return trimmedRoot;
  }
}

const QUERY_API_ROOT = resolveApiRoot(import.meta.env.VITE_QUERY_API_BASE_URL ?? "http://127.0.0.1:8710", "");
const QUERY_API_STREAM_PATH = "/api/search/external/query/stream";
const ENTITY_GRAPH_API_ROOT = resolveApiRoot(
  import.meta.env.VITE_ENTITY_GRAPH_API_BASE_URL ?? "http://127.0.0.1:8711",
  "",
);

function buildAbsoluteApiUrl(root: string, path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${root}${normalizedPath}`;
}

async function readResponseDetail(response: Response): Promise<string> {
  try {
    const errorData = (await response.clone().json()) as { detail?: string; message?: string };
    return String(errorData.detail || errorData.message || "").trim();
  } catch {
    try {
      return (await response.text()).trim();
    } catch {
      return "";
    }
  }
}

function splitAnswerTokens(delta: string) {
  return Array.from(delta);
}

function waitForAnswerToken(signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ANSWER_TOKEN_INTERVAL_MS);
    signal.addEventListener("abort", abort, { once: true });
  });
}

function getNodeLabel(node: GraphNode) {
  return node.name || node.label || node.id;
}

function formatSeedEntityNames(entityGraphResult: EntityGraphApiResponse | null) {
  const names =
    entityGraphResult?.entity_graphs
      .flatMap((item) => item.matched_entities.map((entity) => entity.name))
      .filter(Boolean) || [];
  return names.slice(0, 3).join(" / ") || "等待实体抽取";
}

function formatHeroQuery(queryText: string) {
  const trimmed = queryText.trim();
  if (!trimmed) return "输入查询后，这里会显示当前检索主题。";
  return trimmed.length > 64 ? `${trimmed.slice(0, 64)}...` : trimmed;
}

function buildGraphFromRelations(relations: RelationResult[]) {
  const nodes = new Map<string, GraphNode>();
  const links: GraphLink[] = [];

  relations.forEach((relation) => {
    const subjectIsSeed = relation.matched_entity_ids?.includes(relation.subject_id) || false;
    const objectIsSeed = relation.matched_entity_ids?.includes(relation.object_id) || false;
    nodes.set(relation.subject_id, {
      id: relation.subject_id,
      label: relation.subject_name,
      kind: "entity",
      is_seed: subjectIsSeed,
    });
    nodes.set(relation.object_id, {
      id: relation.object_id,
      label: relation.object_name,
      kind: "entity",
      is_seed: objectIsSeed,
    });
    links.push({
      id: relation.id,
      source: relation.subject_id,
      target: relation.object_id,
      label: relation.relation || "相关",
      matched_entity_ids: relation.matched_entity_ids,
    });
  });

  return { nodes: Array.from(nodes.values()), links };
}

function buildReferenceMap(queryResult: QueryApiResponse | null) {
  const references = queryResult?.grounded_passages || [];
  return new Map(references.map((item) => [item.id, item]));
}

function buildRelationReferenceLinks(queryResult: QueryApiResponse | null) {
  const relationToPassages = new Map<string, Set<string>>();

  queryResult?.grounded_passages.forEach((passage) => {
    (passage.matched_relation_ids || []).forEach((relationId) => {
      if (!relationToPassages.has(relationId)) relationToPassages.set(relationId, new Set());
      relationToPassages.get(relationId)?.add(passage.id);
    });
  });

  return { relationToPassages };
}

function hashText(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function buildClusteredEntityGraph(entityGraphData: EntityGraphApiResponse) {
  const clusterDescriptors = entityGraphData.entity_graphs.map((item, index) => ({
    clusterId: item.center_entity_id || `cluster:${index}:${item.entity_text}`,
    entityText: item.entity_text,
    centerNodeId: item.center_entity_id || null,
  }));

  const radius = Math.max(240, clusterDescriptors.length * 100);
  const clusterTargets = new Map<string, { x: number; y: number }>();
  clusterDescriptors.forEach((item, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(clusterDescriptors.length, 1);
    clusterTargets.set(item.clusterId, {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius * 0.72,
    });
  });

  const nodes = new Map<string, GraphNode>();
  const links = new Map<string, GraphLink>();

  entityGraphData.entity_graphs.forEach((entityGraph, index) => {
    const clusterId = clusterDescriptors[index]?.clusterId || `cluster:${index}:${entityGraph.entity_text}`;
    const clusterTarget = clusterTargets.get(clusterId) || { x: 0, y: 0 };

    entityGraph.graph.nodes.forEach((node, nodeIndex) => {
      const existing = nodes.get(node.id);
      const clusterIds = new Set<string>(existing?.cluster_ids || []);
      clusterIds.add(clusterId);
      const isClusterCenter = Boolean(
        node.id === entityGraph.center_entity_id || existing?.is_cluster_center,
      );
      const jitterSeed = hashText(`${clusterId}:${node.id}:${nodeIndex}`);
      const jitterAngle = (jitterSeed % 360) * (Math.PI / 180);
      const jitterRadius = isClusterCenter ? 0 : 48 + (jitterSeed % 90);

      nodes.set(node.id, {
        ...existing,
        ...node,
        is_seed: Boolean(existing?.is_seed || node.is_seed),
        is_cluster_center: isClusterCenter,
        cluster_ids: Array.from(clusterIds),
        primary_cluster_id: existing?.primary_cluster_id || clusterId,
        cluster_target_x: existing?.cluster_target_x ?? clusterTarget.x,
        cluster_target_y: existing?.cluster_target_y ?? clusterTarget.y,
        x:
          existing?.x ??
          clusterTarget.x + Math.cos(jitterAngle) * jitterRadius,
        y:
          existing?.y ??
          clusterTarget.y + Math.sin(jitterAngle) * jitterRadius,
      });
    });

    entityGraph.graph.links.forEach((link) => {
      const existing = links.get(link.id);
      const clusterIds = new Set<string>(existing?.cluster_ids || []);
      clusterIds.add(clusterId);
      links.set(link.id, {
        ...existing,
        ...link,
        cluster_ids: Array.from(clusterIds),
        primary_cluster_id: existing?.primary_cluster_id || clusterId,
      });
    });
  });

  if (nodes.size === 0) {
    return entityGraphData.merged_graph || { nodes: [], links: [] };
  }

  return {
    nodes: Array.from(nodes.values()),
    links: Array.from(links.values()),
  };
}

function trimGraphForRender(
  graph: { nodes: GraphNode[]; links: GraphLink[] },
  options: {
    centerId: string | null;
    selectedRelationId: string | null;
    renderDepth: number;
    relationLimit: number;
    maxNodes: number;
    maxLinks: number;
  },
) {
  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  const scoreLink = (link: GraphLink) =>
    (link.id === options.selectedRelationId ? 1000 : 0) +
    ((link.source === options.centerId || link.target === options.centerId) ? 200 : 0) +
    ((link.matched_entity_ids?.length || 0) * 10);
  const rankedLinks = [...graph.links].sort((left, right) => scoreLink(right) - scoreLink(left));
  const adjacency = new Map<string, GraphLink[]>();
  for (const link of rankedLinks) {
    const sourceId = String(link.source);
    const targetId = String(link.target);
    if (!adjacency.has(sourceId)) adjacency.set(sourceId, []);
    if (!adjacency.has(targetId)) adjacency.set(targetId, []);
    adjacency.get(sourceId)?.push(link);
    adjacency.get(targetId)?.push(link);
  }

  const preferredRootIds = graph.nodes
    .filter((node) => node.is_cluster_center)
    .map((node) => node.id);
  const rootIds = preferredRootIds.length > 0 ? preferredRootIds : [options.centerId || graph.nodes[0]?.id || ""].filter(Boolean);
  if (rootIds.length === 0) return { nodes: [], links: [] };

  const depthByNode = new Map<string, number>(rootIds.map((nodeId) => [nodeId, 0]));
  const queue: string[] = [...rootIds];
  const keptNodeIds = new Set<string>(rootIds);
  const keptLinkIds = new Set<string>();

  while (queue.length > 0 && keptNodeIds.size < options.maxNodes && keptLinkIds.size < options.maxLinks) {
    const currentId = queue.shift();
    if (!currentId) break;
    const currentDepth = depthByNode.get(currentId) ?? 0;
    if (currentDepth >= options.renderDepth) continue;
    const candidateLinks = (adjacency.get(currentId) || []).slice(0, options.relationLimit);
    for (const link of candidateLinks) {
      if (keptLinkIds.size >= options.maxLinks) break;
      const sourceId = String(link.source);
      const targetId = String(link.target);
      const nextId = sourceId === currentId ? targetId : sourceId;
      if (!nodeMap.has(nextId)) continue;
      if (!keptNodeIds.has(nextId) && keptNodeIds.size >= options.maxNodes) continue;
      keptLinkIds.add(link.id);
      if (!keptNodeIds.has(nextId)) {
        keptNodeIds.add(nextId);
        depthByNode.set(nextId, currentDepth + 1);
        queue.push(nextId);
      }
      keptNodeIds.add(sourceId);
      keptNodeIds.add(targetId);
    }
  }

  const pinNode = (nodeId: string | null) => {
    if (nodeId && nodeMap.has(nodeId)) keptNodeIds.add(nodeId);
  };

  pinNode(options.centerId);
  const rankedNodes = [...graph.nodes].sort((left, right) => {
    const leftScore =
      (left.is_cluster_center ? 1200 : 0) +
      (left.id === options.centerId ? 1000 : 0) +
      (left.is_seed ? 100 : 0);
    const rightScore =
      (right.is_cluster_center ? 1200 : 0) +
      (right.id === options.centerId ? 1000 : 0) +
      (right.is_seed ? 100 : 0);
    return rightScore - leftScore;
  });

  for (const node of rankedNodes) {
    if (keptNodeIds.size >= options.maxNodes) break;
    keptNodeIds.add(node.id);
  }

  const keptNodes = rankedNodes.filter((node) => keptNodeIds.has(node.id));
  const finalLinks = rankedLinks.filter(
    (link) =>
      keptLinkIds.has(link.id) &&
      keptNodeIds.has(String(link.source)) &&
      keptNodeIds.has(String(link.target)),
  );
  return { nodes: keptNodes, links: finalLinks };
}

function App() {
  const [selectedKbId, setSelectedKbId] = React.useState(DEFAULT_KB_ID);
  const [queryText, setQueryText] = React.useState("");
  const [searchLoading, setSearchLoading] = React.useState(false);
  const [searchError, setSearchError] = React.useState("");
  const [queryResult, setQueryResult] = React.useState<QueryApiResponse | null>(null);
  const [isAnswerStreaming, setIsAnswerStreaming] = React.useState(false);
  const [answerError, setAnswerError] = React.useState("");
  const [entityGraphResult, setEntityGraphResult] = React.useState<EntityGraphApiResponse | null>(null);
  const [graphLoading, setGraphLoading] = React.useState(false);
  const [graphError, setGraphError] = React.useState("");
  const [graphData, setGraphData] = React.useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
    nodes: [],
    links: [],
  });
  const [graphCenterId, setGraphCenterId] = React.useState<string | null>(null);
  const [selectedRelationId, setSelectedRelationId] = React.useState<string | null>(null);
  const [hoveredRelationId, setHoveredRelationId] = React.useState<string | null>(null);
  const [hoveredRelationCard, setHoveredRelationCard] = React.useState<HoveredRelationCard | null>(null);
  const [graphDepth, setGraphDepth] = React.useState(FIXED_ENTITY_GRAPH_DEPTH);
  const [relationLimit, setRelationLimit] = React.useState(FIXED_ENTITY_GRAPH_RELATION_LIMIT);
  const graphStageRef = React.useRef<HTMLDivElement | null>(null);
  const hoverTimerRef = React.useRef<number | null>(null);
  const pointerRef = React.useRef({ x: 0, y: 0 });
  const requestIdRef = React.useRef(0);
  const graphRef = React.useRef<any>(null);
  const queryStreamAbortRef = React.useRef<AbortController | null>(null);

  const referenceMap = React.useMemo(() => buildReferenceMap(queryResult), [queryResult]);
  const { relationToPassages } = React.useMemo(() => buildRelationReferenceLinks(queryResult), [queryResult]);

  const runSearch = React.useCallback(() => {
    if (!selectedKbId || !queryText.trim()) return;
    queryStreamAbortRef.current?.abort();
    requestIdRef.current += 1;
    const requestId = requestIdRef.current;
    const abortController = new AbortController();
    queryStreamAbortRef.current = abortController;
    setSearchLoading(true);
    setIsAnswerStreaming(false);
    setGraphLoading(true);
    setSearchError("");
    setGraphError("");
    setAnswerError("");
    setQueryResult(null);
    setEntityGraphResult(null);
    setGraphData({ nodes: [], links: [] });
    setGraphCenterId(null);
    setSelectedRelationId(null);
    setHoveredRelationId(null);
    setHoveredRelationCard(null);
    const question = queryText.trim();

    const queryTask = (async () => {
      const queryResponse = await fetch(buildAbsoluteApiUrl(QUERY_API_ROOT, "/api/search/external/query"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          question,
          user_id: DEFAULT_USER_ID,
          kb_id: selectedKbId,
          mode: "hybrid",
          top_k: QUERY_TOP_K,
          entity_top_k: QUERY_ENTITY_TOP_K,
          relation_top_k: QUERY_RELATION_TOP_K,
          expansion_degree: FIXED_HYBRID_EXPANSION_DEGREE,
          include_answer: false,
        }),
      });
      if (!queryResponse.ok) {
        const detail = await readResponseDetail(queryResponse);
        throw new Error(detail || "查询接口调用失败");
      }
      return (await queryResponse.json()) as QueryApiResponse;
    })();

    const streamAnswer = async () => {
      setIsAnswerStreaming(true);
      const answerResponse = await fetch(buildAbsoluteApiUrl(QUERY_API_ROOT, QUERY_API_STREAM_PATH), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          question,
          user_id: DEFAULT_USER_ID,
          kb_id: selectedKbId,
          mode: "hybrid",
          top_k: QUERY_TOP_K,
          entity_top_k: QUERY_ENTITY_TOP_K,
          relation_top_k: QUERY_RELATION_TOP_K,
          expansion_degree: FIXED_HYBRID_EXPANSION_DEGREE,
          include_answer: true,
          answer_top_k: ANSWER_TOP_K,
          passage_top_k: PASSAGE_TOP_K,
        }),
      });
      if (!answerResponse.ok) {
        const detail = await readResponseDetail(answerResponse);
        throw new Error(detail || "AI 回答接口调用失败");
      }
      if (!answerResponse.body) {
        throw new Error("AI 回答流未返回可读取内容");
      }

      const reader = answerResponse.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let renderedAnswer = "";
      let receivedAnswerDelta = false;
      let tokenRenderChain = Promise.resolve();

      const appendAnswerDelta = (delta: string) => {
        if (!delta) return tokenRenderChain;
        receivedAnswerDelta = true;
        const tokens = splitAnswerTokens(delta);
        tokenRenderChain = tokenRenderChain.then(async () => {
          for (const token of tokens) {
            if (requestId !== requestIdRef.current) return;
            renderedAnswer += token;
            setQueryResult((current) =>
              current
                ? {
                    ...current,
                    answer: renderedAnswer,
                  }
                : current,
            );
            await waitForAnswerToken(abortController.signal);
          }
        });
        return tokenRenderChain;
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          const event = JSON.parse(trimmed) as
            | {
                type: "meta";
                question?: string;
                user_id?: string;
                kb_id?: string;
                mode?: string;
                retrieval_query?: string | null;
                results?: RelationResult[];
                grounded_passages?: GroundedPassage[];
                entity_hits?: QueryApiResponse["entity_hits"];
                metadata?: Record<string, unknown>;
              }
            | { type: "answer_delta"; delta: string }
            | { type: "done"; answer?: string; metadata?: Record<string, unknown> }
            | { type: "error"; message?: string };

          if (requestId !== requestIdRef.current) continue;

          if (event.type === "meta") {
            setQueryResult((current) =>
              current
                ? {
                    ...current,
                    answer: current.answer || "",
                    retrieval_query: event.retrieval_query || current.retrieval_query,
                    results: event.results || current.results,
                    grounded_passages: event.grounded_passages || current.grounded_passages,
                    entity_hits: event.entity_hits || current.entity_hits,
                    metadata: event.metadata ? { ...(current.metadata || {}), ...event.metadata } : current.metadata,
                  }
                : {
                    question: event.question || question,
                    user_id: event.user_id || DEFAULT_USER_ID,
                    kb_id: event.kb_id || selectedKbId,
                    mode: event.mode || "hybrid",
                    answer: "",
                    retrieval_query: event.retrieval_query || question,
                    results: event.results || [],
                    grounded_passages: event.grounded_passages || [],
                    entity_hits: event.entity_hits || [],
                    metadata: event.metadata || {},
                  },
            );
            continue;
          }

          if (event.type === "answer_delta") {
            void appendAnswerDelta(event.delta);
            continue;
          }

          if (event.type === "done") {
            if (!receivedAnswerDelta && event.answer) {
              void appendAnswerDelta(event.answer);
            }
            setQueryResult((current) =>
              current
                ? {
                    ...current,
                    metadata: event.metadata ? { ...(current.metadata || {}), ...event.metadata } : current.metadata,
                  }
                : current,
            );
            continue;
          }

          if (event.type === "error") {
            throw new Error(event.message || "流式回答失败");
          }
        }

        if (done) break;
      }

      await tokenRenderChain;
    };

    const entityGraphTask = (async () => {
      const entityGraphResponse = await fetch(buildAbsoluteApiUrl(ENTITY_GRAPH_API_ROOT, "/api/entity-graph/extract-and-expand"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          user_id: DEFAULT_USER_ID,
          kb_id: selectedKbId,
          max_entities: 3,
          name_match_limit: 3,
          fallback_vector_limit: 3,
          default_depth: FIXED_ENTITY_GRAPH_DEPTH,
          default_relation_limit: FIXED_ENTITY_GRAPH_RELATION_LIMIT,
          shuffle_seed: 0,
        }),
      });
      if (!entityGraphResponse.ok) {
        const detail = await readResponseDetail(entityGraphResponse);
        throw new Error(detail || "实体图接口调用失败");
      }
      return (await entityGraphResponse.json()) as EntityGraphApiResponse;
    })();

    void queryTask
      .then((data) => {
        if (requestId !== requestIdRef.current) return;
        if (data) {
          setQueryResult({
            ...data,
            answer: "",
          });
          void streamAnswer()
            .catch((error: unknown) => {
              if (requestId !== requestIdRef.current) return;
              if (error instanceof DOMException && error.name === "AbortError") return;
              setAnswerError(error instanceof Error ? error.message : "AI 回答失败");
            })
            .finally(() => {
              if (requestId !== requestIdRef.current) return;
              setIsAnswerStreaming(false);
            });
        }
      })
      .catch((error: unknown) => {
        if (requestId !== requestIdRef.current) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        setSearchError(error instanceof Error ? error.message : "搜索失败");
      })
      .finally(() => {
        if (requestId !== requestIdRef.current) return;
        setSearchLoading(false);
      });

    void entityGraphTask
      .then((entityGraphData) => {
        if (requestId !== requestIdRef.current) return;
        setEntityGraphResult(entityGraphData);

        const mergedGraph = buildClusteredEntityGraph(entityGraphData);
        const firstCenterId =
          entityGraphData.entity_graphs.find((item) => item.center_entity_id)?.center_entity_id ||
          mergedGraph.nodes[0]?.id ||
          null;

        setGraphData(mergedGraph);
        setGraphCenterId(firstCenterId);
      })
      .catch((error: unknown) => {
        if (requestId !== requestIdRef.current) return;
        setGraphError(error instanceof Error ? error.message : "实体图生成失败");
      })
      .finally(() => {
        if (requestId !== requestIdRef.current) return;
        setGraphLoading(false);
      });
  }, [queryText, selectedKbId]);

  React.useEffect(() => {
    return () => {
      queryStreamAbortRef.current?.abort();
      if (hoverTimerRef.current !== null) {
        window.clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  const clearHoveredRelationCard = React.useCallback(() => {
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    setHoveredRelationCard(null);
  }, []);

  const scheduleHoveredRelationCard = React.useCallback(
    (relation: GraphLink) => {
      if (hoverTimerRef.current !== null) {
        window.clearTimeout(hoverTimerRef.current);
      }
      hoverTimerRef.current = window.setTimeout(() => {
        setHoveredRelationCard({
          relation,
          x: pointerRef.current.x,
          y: pointerRef.current.y,
        });
        hoverTimerRef.current = null;
      }, RELATION_TOOLTIP_DELAY_MS);
    },
    [],
  );

  const handleGraphPointerMove = React.useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const stageRect = graphStageRef.current?.getBoundingClientRect();
    if (!stageRect) return;
    pointerRef.current = {
      x: event.clientX - stageRect.left,
      y: event.clientY - stageRect.top,
    };
    setHoveredRelationCard((current) =>
      current
        ? {
            ...current,
            x: pointerRef.current.x,
            y: pointerRef.current.y,
          }
        : current,
    );
  }, []);

  const handleLinkHover = React.useCallback(
    (link: GraphLinkDatum | null) => {
      const nextId = link?.id || null;
      setHoveredRelationId(nextId);
      clearHoveredRelationCard();
      if (!nextId) return;
      if (!link?.describe?.trim()) return;
      scheduleHoveredRelationCard({
        id: link.id,
        source: String(typeof link.source === "string" ? link.source : link.source.id),
        target: String(typeof link.target === "string" ? link.target : link.target.id),
        label: link.label,
        describe: link.describe,
        matched_entity_ids: link.matched_entity_ids,
        color: link.color,
      });
    },
    [clearHoveredRelationCard, scheduleHoveredRelationCard],
  );
  const activeNodeIds = React.useMemo(() => {
    const ids = new Set<string>();
    graphData.links.forEach((link) => {
      if (link.id === selectedRelationId) {
        ids.add(String(link.source));
        ids.add(String(link.target));
      }
    });
    return ids;
  }, [graphData.links, selectedRelationId]);

  const enrichedGraph = React.useMemo(
    () => ({
      nodes: graphData.nodes.map((node) => ({
        ...node,
        color:
          node.id === graphCenterId
            ? "#2f80ff"
            : node.is_cluster_center
              ? "#4b8cff"
            : activeNodeIds.has(node.id)
              ? "#58b8ff"
              : node.is_seed
                ? "#82aefc"
                : "#cfe2ff",
      })),
      links: graphData.links,
    }),
    [activeNodeIds, graphCenterId, graphData.links, graphData.nodes],
  );

  const renderGraph = React.useMemo(
    () =>
      trimGraphForRender(enrichedGraph, {
        centerId: graphCenterId,
        selectedRelationId,
        renderDepth: graphDepth,
        relationLimit,
        maxNodes: MAX_RENDER_NODES,
        maxLinks: MAX_RENDER_LINKS,
      }),
    [enrichedGraph, graphCenterId, graphDepth, relationLimit, selectedRelationId],
  );

  const selectedRelation = React.useMemo(
    () => queryResult?.results.find((item) => item.id === selectedRelationId) || null,
    [queryResult, selectedRelationId],
  );

  const entityGraphReferenceFallback = React.useMemo(() => {
    const nodeLabels = new Map(graphData.nodes.map((node) => [node.id, getNodeLabel(node)]));
    return graphData.links
      .filter((link) => link.describe?.trim())
      .slice(0, 30)
      .map((link, index): GroundedPassage => {
        const sourceName = nodeLabels.get(String(link.source)) || String(link.source);
        const targetName = nodeLabels.get(String(link.target)) || String(link.target);
        return {
          id: `entity-graph-reference-${link.id}-${index}`,
          passage: `${sourceName} -> ${targetName}｜${link.label || "相关"}：${link.describe?.trim()}`,
          docment_id: "实体图关系说明",
          score: 0,
          matched_relation_ids: [link.id],
        };
      });
  }, [graphData.links, graphData.nodes]);

  const visibleReferences = React.useMemo(() => {
    if (!queryResult) return entityGraphReferenceFallback;
    if (selectedRelationId) {
      const passageIds = Array.from(relationToPassages.get(selectedRelationId) || []);
      if (passageIds.length > 0) {
        return passageIds
          .map((id) => referenceMap.get(id))
          .filter((item): item is GroundedPassage => Boolean(item));
      }
      const fromRelationPassage = selectedRelation?.passage?.trim();
      if (fromRelationPassage) {
        return [
          {
            id: selectedRelation.id,
            passage: fromRelationPassage,
            docment_id: selectedRelation.docment_id,
            score: selectedRelation.score,
            matched_relation_ids: [selectedRelation.id],
          },
        ];
      }
    }

    return queryResult.grounded_passages.length > 0 ? queryResult.grounded_passages : entityGraphReferenceFallback;
  }, [entityGraphReferenceFallback, queryResult, referenceMap, relationToPassages, selectedRelation, selectedRelationId]);

  const similarDescribeResults = React.useMemo(() => {
    if (!queryResult) return [];
    return [...queryResult.results]
      .filter((item) => item.describe?.trim())
      .sort((left, right) => {
        const leftScore = left.score_breakdown?.semantic_score ?? left.score_breakdown?.hybrid_score ?? left.score;
        const rightScore = right.score_breakdown?.semantic_score ?? right.score_breakdown?.hybrid_score ?? right.score;
        return rightScore - leftScore;
      })
      .slice(0, 10);
  }, [queryResult]);

  const graphSummary = React.useMemo(() => {
    return `${graphData.nodes.length} 个节点 / ${graphData.links.length} 条关系`;
  }, [graphData.links.length, graphData.nodes.length]);

  const answerText = React.useMemo(() => queryResult?.answer?.trim() || "", [queryResult]);

  const renderGraphSummary = React.useMemo(() => {
    return `当前渲染 ${renderGraph.nodes.length} / ${graphData.nodes.length} 个节点，${renderGraph.links.length} / ${graphData.links.length} 条关系`;
  }, [graphData.links.length, graphData.nodes.length, renderGraph.links.length, renderGraph.nodes.length]);

  const heroStats = React.useMemo(
    () => [
      { label: "当前 KB", value: selectedKbId || "--" },
      { label: "命中实体", value: entityGraphResult?.extracted_entities.length || "--" },
      { label: "证据片段", value: visibleReferences.length || "--" },
    ],
    [entityGraphResult, selectedKbId, visibleReferences.length],
  );

  React.useEffect(() => {
    const graph = graphRef.current;
    if (!graph || renderGraph.nodes.length === 0) return;

    graph.d3Force("charge", forceManyBody().strength(-60));
    graph.d3Force(
      "collide",
      forceCollide<GraphNodeDatum>((node) => (node.is_cluster_center ? 38 : 18)).strength(0.85),
    );
    graph.d3Force(
      "cluster-x",
      forceX<GraphNodeDatum>((node) => node.cluster_target_x ?? 0).strength((node) =>
        node.is_cluster_center ? 0.42 : 0.18,
      ),
    );
    graph.d3Force(
      "cluster-y",
      forceY<GraphNodeDatum>((node) => node.cluster_target_y ?? 0).strength((node) =>
        node.is_cluster_center ? 0.42 : 0.18,
      ),
    );

    // 给节点一点初速度，避免新前端在聚类初始位置接近稳定态时看起来完全不动。
    renderGraph.nodes.forEach((node) => {
      const graphNode = node as GraphNodeDatum;
      graphNode.vx = (graphNode.vx ?? 0) + (Math.random() - 0.5) * 0.9;
      graphNode.vy = (graphNode.vy ?? 0) + (Math.random() - 0.5) * 0.9;
    });

    graph.d3ReheatSimulation();
  }, [renderGraph]);

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-nav">
          <div className="hero-brand">
            <img src={appLogo} alt="NextGraph" />
            <div>
              <p className="eyebrow">NextGraph Workspace</p>
              <p className="brand-title">Knowledge Search Console</p>
            </div>
          </div>
          <div className="hero-tabs" aria-label="sections">
            <span>Overview</span>
            <span>Search</span>
            <span>Graph</span>
            <span>Evidence</span>
          </div>
        </div>

        <div className="hero-layout">
          <div className="hero-copyblock">
            <p className="eyebrow">Enterprise Search Workspace</p>
            <h1>知识库搜索、关系图谱与证据阅读一体化工作台。</h1>
            <p className="hero-copy">
              面向知识库检索与分析流程，将查询输入、图谱联动和证据阅读整合在同一页面中，
              减少切换成本，方便快速定位实体关系与支撑材料。
            </p>

            <div className="hero-stats">
              {heroStats.map((item) => (
                <div key={item.label} className="hero-stat">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </div>

          <aside className="hero-poster">
            <div className="poster-frame">
              <p className="poster-kicker">Current Session</p>
              <strong>{selectedKbId || "Select a knowledge base"}</strong>
              <span>{formatHeroQuery(queryText)}</span>
              <div className="poster-divider" />
              <div className="poster-meta">
                <span>
                  <Film size={14} />
                  {graphSummary}
                </span>
                <span>
                  <BookOpen size={14} />
                  {formatSeedEntityNames(entityGraphResult)}
                </span>
              </div>
            </div>
          </aside>
        </div>
      </header>

      <main className="workspace">
        <section className="search-panel card">
          <div className="section-head">
            <div>
              <span className="section-tag">Search Desk</span>
              <h2>检索台</h2>
              <p className="section-copy">选择知识库、调整图谱范围并发起查询，结果会同步联动到图谱与证据面板。</p>
            </div>
            <div className="stat-chip">
              <Network size={16} />
              <span>{graphSummary}</span>
            </div>
          </div>

          <div className="curation-strip">
            <article className="curation-note">
              <span className="curation-index">01</span>
              <div>
                <strong>精确输入</strong>
                <p>查询尽量明确，便于系统快速定位相关实体、关系与证据来源。</p>
              </div>
            </article>
            <article className="curation-note">
              <span className="curation-index">02</span>
              <div>
                <strong>局部分析</strong>
                <p>点击节点即可重设图谱中心，用更小的范围观察关系结构。</p>
              </div>
            </article>
            <article className="curation-note">
              <span className="curation-index">03</span>
              <div>
                <strong>证据联动</strong>
                <p>节点、关系和文本证据保持同步，方便核对结果的支撑依据。</p>
              </div>
            </article>
          </div>

          <div className="control-grid">
            <label className="field">
              <span>知识库</span>
              <input
                type="text"
                value={selectedKbId}
                onChange={(event) => setSelectedKbId(event.target.value)}
                placeholder="请输入 kb_id"
              />
            </label>

            <label className="field">
              <span>图谱跳数</span>
              <div className="range-field">
                <input
                  type="range"
                  min="1"
                  max={FIXED_ENTITY_GRAPH_DEPTH}
                  value={graphDepth}
                  onChange={(event) => setGraphDepth(Number(event.target.value))}
                />
                <strong>{FIXED_ENTITY_GRAPH_DEPTH}</strong>
                <strong>{graphDepth}</strong>
              </div>
            </label>

            <label className="field">
              <span>周边关系数</span>
              <div className="range-field">
                <input
                  type="range"
                  min="4"
                  max={FIXED_ENTITY_GRAPH_RELATION_LIMIT}
                  value={relationLimit}
                  onChange={(event) => setRelationLimit(Number(event.target.value))}
                />
                <strong>{FIXED_ENTITY_GRAPH_RELATION_LIMIT}</strong>
                <strong>{relationLimit}</strong>
              </div>
            </label>
          </div>

          <label className="field query-field">
            <span>问题</span>
            <textarea
              rows={4}
              placeholder="例如：这个知识库里谁和某个产品、技术或事件关系最密切？"
              value={queryText}
              onChange={(event) => setQueryText(event.target.value)}
            />
          </label>

          <div className="panel-actions">
            <button className="primary-button" onClick={() => void runSearch()} disabled={searchLoading || !selectedKbId}>
              <Search size={18} />
              <span>{searchLoading ? "搜索中..." : "开始搜索"}</span>
            </button>
            <button
              className="ghost-button"
              onClick={() => {
                setSelectedRelationId(null);
              }}
              disabled={!queryResult && !entityGraphResult}
            >
              <SlidersHorizontal size={18} />
              <span>清除高亮</span>
            </button>
          </div>

          {answerText || isAnswerStreaming || queryResult?.retrieval_query || answerError ? (
            <div className="answer-panel query-answer-panel">
              <div className="answer-head">
                <div>
                  <span className="answer-kicker">AI Answer</span>
                  <strong>{answerText ? "问题回答" : isAnswerStreaming ? "正在生成回答..." : "问题回答"}</strong>
                </div>
                <div className="reference-meta">
                  <span>{queryResult?.grounded_passages.length || 0} 段参考资料</span>
                </div>
              </div>
              <p>{answerText || (answerError ? "AI 回答暂时不可用，你可以先参考下方图谱和右侧证据。" : "系统已完成检索，正在组织回答...")}</p>
              {queryResult?.retrieval_query?.trim() ? (
                <div className="answer-query">
                  <span>系统检索理解</span>
                  <p>{queryResult.retrieval_query.trim()}</p>
                </div>
              ) : null}
              {answerError ? <p className="feedback warn">{answerError}</p> : null}
            </div>
          ) : searchLoading ? (
            <div className="answer-panel query-answer-panel">
              <span className="answer-kicker">AI Answer</span>
              <strong>正在生成回答...</strong>
              <p>系统会结合检索到的关系和参考资料组织答案。</p>
            </div>
          ) : queryResult ? (
            <div className="answer-panel query-answer-panel">
              <span className="answer-kicker">AI Answer</span>
              <strong>本次未生成回答</strong>
              <p>已经返回关系和参考资料，你可以继续查看下方图谱和右侧证据片段。</p>
            </div>
          ) : null}

          {searchError ? <p className="feedback error">{searchError}</p> : null}
          {graphError ? <p className="feedback warn">{graphError}</p> : null}
        </section>

        <section className="content-grid">
          <section className="card graph-card">
            <div className="section-head">
              <div>
                <span className="section-tag">Graph Stage</span>
                <h2>知识网络画布</h2>
                <p className="section-copy">使用实体图接口返回的自定义结构，展示抽取实体合并后的关系网络。</p>
              </div>
              <div className="meta-strip">
                <span>中心：{graphCenterId || "未选中"}</span>
                <span>{graphLoading ? "加载中..." : "已同步"}</span>
              </div>
            </div>

            <div className="graph-hint">
              <span>后端固定按每个实体 4 跳、每跳最多 30 条关系扩图；滑块只控制前端渲染范围。</span>
            </div>

            <div className="graph-hint">
              <span>{renderGraphSummary}</span>
            </div>

            <div
              ref={graphStageRef}
              className="graph-stage"
              onMouseMove={handleGraphPointerMove}
              onMouseLeave={() => {
                setHoveredRelationId(null);
                clearHoveredRelationCard();
              }}
            >
              {renderGraph.nodes.length > 0 ? (
                <ForceGraph2D
                  ref={graphRef}
                  graphData={renderGraph}
                  backgroundColor="transparent"
                  nodeRelSize={7}
                  cooldownTicks={Number.POSITIVE_INFINITY}
                  d3AlphaDecay={0.018}
                  d3VelocityDecay={0.22}
                  enableNodeDrag
                  onEngineStop={() => {
                    graphRef.current?.d3ReheatSimulation();
                  }}
                  linkWidth={(link: GraphLinkDatum) =>
                    link.id === selectedRelationId ? 4.2 : hoveredRelationId === link.id ? 2.4 : 1.2
                  }
                  linkColor={(link: GraphLinkDatum) =>
                    link.id === selectedRelationId
                      ? "#225cff"
                      : hoveredRelationId === link.id
                        ? "rgba(57, 119, 255, 0.82)"
                        : "rgba(124, 160, 220, 0.5)"
                  }
                  linkDirectionalArrowLength={(link: GraphLinkDatum) =>
                    link.id === selectedRelationId ? 11 : hoveredRelationId === link.id ? 8 : 6
                  }
                  linkDirectionalArrowRelPos={1}
                  linkDirectionalArrowColor={(link: GraphLinkDatum) =>
                    link.id === selectedRelationId
                      ? "#225cff"
                      : hoveredRelationId === link.id
                        ? "rgba(57, 119, 255, 0.82)"
                        : "rgba(124, 160, 220, 0.5)"
                  }
                  linkDirectionalParticles={(link: GraphLinkDatum) => (link.id === selectedRelationId ? 4 : hoveredRelationId === link.id ? 2 : 0)}
                  linkDirectionalParticleWidth={(link: GraphLinkDatum) => (link.id === selectedRelationId ? 3.2 : 2)}
                  nodeCanvasObject={(node: GraphNodeDatum, ctx, globalScale) => {
                    const label = getNodeLabel(node);
                    const fontSize = node.id === graphCenterId ? 15 : 12;
                    const radius = node.id === graphCenterId ? 8 : node.is_seed ? 6.5 : 5.5;
                    ctx.beginPath();
                    ctx.arc(node.x || 0, node.y || 0, radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = node.color || "#cfe2ff";
                    ctx.shadowColor = "rgba(92, 152, 255, 0.26)";
                    ctx.shadowBlur = node.id === graphCenterId ? 18 : 10;
                    ctx.fill();
                    ctx.shadowBlur = 0;

                    if (globalScale >= 0.75 || node.id === graphCenterId) {
                      ctx.font = `${fontSize / globalScale}px "Space Grotesk", "PingFang SC", sans-serif`;
                      ctx.fillStyle = "#21456f";
                      ctx.textAlign = "center";
                      ctx.textBaseline = "top";
                      ctx.fillText(label, node.x || 0, (node.y || 0) + radius + 4);
                    }
                  }}
                  nodePointerAreaPaint={(node: GraphNodeDatum, color, ctx) => {
                    const radius = node.id === graphCenterId ? 16 : node.is_seed ? 14 : 12;
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(node.x || 0, node.y || 0, radius, 0, 2 * Math.PI, false);
                    ctx.fill();
                  }}
                  linkCanvasObjectMode={() => "after"}
                  linkCanvasObject={(link: GraphLinkDatum, ctx, globalScale) => {
                    const source = link.source as GraphNodeDatum;
                    const target = link.target as GraphNodeDatum;
                    if (typeof source.x !== "number" || typeof source.y !== "number") return;
                    if (typeof target.x !== "number" || typeof target.y !== "number") return;
                    const isSelected = link.id === selectedRelationId;
                    const isHovered = link.id === hoveredRelationId;
                    if (globalScale < 1.2 && !isSelected && !isHovered) return;
                    const label = link.label || "相关";
                    const midX = (source.x + target.x) / 2;
                    const midY = (source.y + target.y) / 2;
                    if (isSelected) {
                      ctx.save();
                      ctx.strokeStyle = "rgba(34, 92, 255, 0.28)";
                      ctx.lineWidth = 10 / globalScale;
                      ctx.beginPath();
                      ctx.moveTo(source.x, source.y);
                      ctx.lineTo(target.x, target.y);
                      ctx.stroke();
                      ctx.restore();
                    }
                    ctx.save();
                    ctx.font = `${(isSelected ? 11 : 10) / globalScale}px "Space Grotesk", "PingFang SC", sans-serif`;
                    ctx.fillStyle = isSelected ? "#225cff" : isHovered ? "#2d67e8" : "rgba(78, 114, 162, 0.82)";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(label, midX, midY);
                    ctx.restore();
                  }}
                  onLinkHover={handleLinkHover}
                  onLinkClick={(link: GraphLinkDatum) => {
                    setSelectedRelationId(link.id);
                    clearHoveredRelationCard();
                  }}
                />
              ) : (
                <div className="empty-state">
                  <Network size={32} />
                  <p>{searchLoading || graphLoading ? "正在生成知识网络..." : "搜索后在这里显示知识库关系网络"}</p>
                </div>
              )}
              {hoveredRelationCard?.relation.describe?.trim() ? (
                <div
                  className="graph-tooltip"
                  style={{
                    left: Math.min(hoveredRelationCard.x + 18, 520),
                    top: Math.max(hoveredRelationCard.y - 18, 18),
                  }}
                >
                  <strong>
                    {getNodeLabel(graphData.nodes.find((node) => node.id === hoveredRelationCard.relation.source) || { id: hoveredRelationCard.relation.source })} →{" "}
                    {getNodeLabel(graphData.nodes.find((node) => node.id === hoveredRelationCard.relation.target) || { id: hoveredRelationCard.relation.target })}
                  </strong>
                  <span>{hoveredRelationCard.relation.label || "相关"}</span>
                  <p>{hoveredRelationCard.relation.describe?.trim()}</p>
                </div>
              ) : null}
            </div>
          </section>

          <aside className="card references-card">
            <div className="section-head">
              <div>
                <span className="section-tag">Evidence Roll</span>
                <h2>参考资料</h2>
                <p className="section-copy">集中查看命中关系对应的原文片段，便于快速验证检索结果。</p>
              </div>
              <div className="stat-chip">
                <span>{visibleReferences.length} 段</span>
              </div>
            </div>

            {selectedRelation ? (
              <div className="relation-focus">
                <strong>
                  {selectedRelation.subject_name} → {selectedRelation.object_name}
                </strong>
                <p>{selectedRelation.relation || "相关"}</p>
                {selectedRelation.describe?.trim() ? (
                  <div className="relation-description">
                    <span>关系说明</span>
                    <p>{selectedRelation.describe.trim()}</p>
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="references-list">
              {similarDescribeResults.length > 0 ? (
                <div className="relation-focus">
                  <strong>问题向量最相近的 10 条描述</strong>
                  <p>按关系检索分数排序，优先展示 `describe` 内容。</p>
                  {similarDescribeResults.map((item, index) => (
                    <article key={`describe-${item.id}-${index}`} className="reference-item">
                      <div className="reference-meta">
                        <span>DES #{index + 1}</span>
                        <span>{item.subject_name} → {item.object_name}</span>
                        <span>score {(item.score_breakdown?.semantic_score ?? item.score).toFixed(3)}</span>
                      </div>
                      <p>{item.describe?.trim()}</p>
                    </article>
                  ))}
                </div>
              ) : null}

              {visibleReferences.length > 0 ? (
                visibleReferences.map((item, index) => (
                  <article key={`${item.id}-${index}`} className="reference-item">
                    <div className="reference-meta">
                      <span>#{index + 1}</span>
                      <span>{item.docment_id || "未标注文档"}</span>
                      <span>score {item.score.toFixed(3)}</span>
                    </div>
                    <p>{item.passage}</p>
                  </article>
                ))
              ) : (
                <div className="empty-state slim">
                  <p>当前没有可展示的参考资料。</p>
                </div>
              )}
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
