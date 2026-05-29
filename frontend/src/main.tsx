import React from "react";
import { createRoot } from "react-dom/client";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force-3d";
import ForceGraph2D from "react-force-graph-2d";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";
import {
  Bot,
  BrainCircuit,
  Cpu,
  Database,
  FolderUp,
  MessageSquareShare,
  Network,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";
import appLogo from "./assets/app-logo.png";
import "./styles.css";

type ModuleKey = "overview" | "builder" | "query";

type FileSummary = {
  chunks?: number;
  entities?: number;
  relations?: number;
  failed_embeddings?: number;
};

type SourceDocument = {
  name: string;
  path: string;
  stage: string;
  state: string;
  progress: number;
  parse_progress: number;
  vector_progress: number;
  summary?: FileSummary;
};

type LibraryPreview = {
  counts: {
    entities: number;
    relations: number;
    passages: number;
  };
  top_entities: Array<{
    id: string;
    name: string;
    relation_count: number;
  }>;
  samples: {
    entities: Array<Record<string, unknown>>;
    relations: Array<Record<string, unknown>>;
    passages: Array<Record<string, unknown>>;
  };
  graph_preview: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
};

type LibraryRecord = {
  kb_id: string;
  user_id: string;
  milvus_db: string;
  latest_task_id: string;
  latest_status: string;
  latest_stage: string;
  latest_message: string;
  updated_at: number;
  task_count: number;
  file_count: number;
  completed: number;
  failed: number;
  chunk_count: number;
  entity_count: number;
  relation_count: number;
  passage_count: number;
  completion_ratio: number;
  source_documents: SourceDocument[];
  live_preview?: LibraryPreview;
  live_preview_error?: string;
};

type LibraryOverview = {
  kb_id: string;
  user_id: string;
  milvus_db: string;
  latest_task: DatabaseBuildProgress;
  summary: {
    file_count: number;
    completed: number;
    failed: number;
    chunk_count: number;
    entity_count: number;
    relation_count: number;
    failed_embeddings: number;
    completion_ratio: number;
    source_documents: SourceDocument[];
  };
  milvus_preview: LibraryPreview | null;
  milvus_error: string | null;
};

type UploadedFile = {
  id: string;
  name: string;
  path: string;
  size: number;
  uploaded_at: number;
  extension: string;
};

type UploadedDocument = {
  id: string;
  name: string;
  path: string;
  type: string;
  uploadedAt: string;
  size: string;
  selected: boolean;
};

type BuildFileProgress = {
  path: string;
  name: string;
  stage: string;
  state: string;
  progress: number;
  parse_progress: number;
  vector_progress: number;
  summary?: FileSummary;
};

type DatabaseBuildProgress = {
  task_id: string;
  status: string;
  stage: string;
  message: string;
  kb_id: string;
  milvus_db: string;
  progress_percent: number;
  files: BuildFileProgress[];
};

type SearchResponse = {
  mode: string;
  query: string;
  user_id: string;
  kb_id: string;
  results: RelationResult[];
  grounded_passages: GroundedPassage[];
  metadata?: Record<string, unknown>;
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

type GraphNode = {
  id: string;
  label?: string;
  name?: string;
  kind?: string;
  is_seed?: boolean;
  color?: string;
};

type GraphLink = {
  id: string;
  source: string;
  target: string;
  label: string;
  matched_entity_ids?: string[];
  color?: string;
};

type GraphDensityMode = "focus" | "balanced" | "panorama";

type PreparedGraphNode = GraphNode & {
  degree: number;
  displayLabel: string;
  emphasis: number;
  val: number;
};

type PreparedGraphLink = GraphLink & {
  source: string;
  target: string;
  emphasis: number;
};

type TopologyNodeType = "domain" | "concept" | "tech" | "person" | "org" | "event";

type TopologyNode = PreparedGraphNode & {
  topologyType: TopologyNodeType;
  fill: string;
  glow: string;
  border: string;
};

type TopologyLink = PreparedGraphLink & {
  stroke: string;
  textStroke: string;
};

type SimulatedTopologyNode = TopologyNode & {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

type SimulatedTopologyLink = {
  id: string;
  source: SimulatedTopologyNode;
  target: SimulatedTopologyNode;
  label: string;
  stroke: string;
  textStroke: string;
};

type GraphCanvasHandle = {
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
};

type QueryGraphStage = "seed" | "result";

type SearchTrace = {
  search: SearchResponse;
  trace: {
    mode: string;
    steps: Array<{
      id: string;
      title: string;
      summary: string;
      highlights?: string[];
      entity_ids?: string[];
      history?: Array<Record<string, unknown>>;
      passage_ids?: string[];
    }>;
    graph: {
      nodes: GraphNode[];
      links: GraphLink[];
    };
    seed_graph?: {
      nodes: GraphNode[];
      links: GraphLink[];
    };
    result_graph?: {
      nodes: GraphNode[];
      links: GraphLink[];
    };
    seed_entities: Array<{
      id: string;
      name: string;
      score: number;
    }>;
    result_relations: RelationResult[];
    grounded_passages: GroundedPassage[];
  };
};

type RagAnswerResponse = {
  mode: string;
  query: string;
  retrieval_query: string;
  user_id: string;
  kb_id: string;
  answer: string;
  results: RelationResult[];
  grounded_passages: GroundedPassage[];
  metadata: Record<string, unknown>;
};

type EntityNeighborhoodResponse = {
  graph: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  center_entity_id?: string | null;
};

const API_ROOT = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_USER_ID = "admin_user";
const DEFAULT_MILVUS_DB = import.meta.env.NEXTGRAPH_MILVUS_DB || "crx";

function apiUrl(path: string) {
  return `${API_ROOT}${path}`;
}

function formatTime(timestamp?: number) {
  if (!timestamp) return "--";
  return new Date(timestamp * 1000).toLocaleString("zh-CN", { hour12: false });
}

function formatUploadedAt(value: number) {
  return new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });
}

function formatFileSize(size: number) {
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

function extractHighlightTerms(text: string) {
  return Array.from(
    new Set(
      text
        .split(/[\s,，。！？；：、"'“”指標()（）【】\-_/]+/)
        .map((item) => item.trim())
        .filter((item) => item.length >= 2),
    ),
  );
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function colorFromSeed(seed: string) {
  const palette = ["#67ccff", "#ffd166", "#64f0cd", "#ff8fab", "#c4a1ff", "#ffb86b"];
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = (hash * 31 + seed.charCodeAt(index)) >>> 0;
  }
  return palette[hash % palette.length];
}

function isOpaqueEntityId(value: string) {
  return /^e_[a-z0-9]{8,}$/i.test(value);
}

function getReadableNodeLabel(node: { label?: string; name?: string; id: string }) {
  const raw = node.name || node.label || node.id;
  if (!raw) return "未命名实体";
  return raw;
}

function truncateLabel(label: string, maxLength = 12) {
  if (label.length <= maxLength) return label;
  return `${label.slice(0, maxLength)}...`;
}

function prepareGraphData(
  rawNodes: GraphNode[],
  rawLinks: GraphLink[],
  options?: {
    selectedRelationId?: string | null;
    densityMode?: GraphDensityMode;
    preferredNodeLimit?: number;
    preferredLinkLimit?: number;
  },
) {
  const degreeMap = new Map<string, number>();
  for (const node of rawNodes) {
    degreeMap.set(node.id, 0);
  }
  for (const link of rawLinks) {
    degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
    degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
  }

  const preparedNodes: PreparedGraphNode[] = rawNodes.map((node) => {
    const degree = degreeMap.get(node.id) || 0;
    const emphasis = (node.is_seed ? 4 : 0) + Math.min(degree, 4);
    return {
      ...node,
      degree,
      displayLabel: truncateLabel(getReadableNodeLabel(node), 12),
      emphasis,
      val: node.is_seed ? 12 : Math.min(10, 5 + degree * 0.8),
    };
  });

  const nodeMap = new Map(preparedNodes.map((node) => [node.id, node]));
  const selectedRelationId = options?.selectedRelationId ?? null;
  const selectedLink = selectedRelationId
    ? rawLinks.find((link) => link.id === selectedRelationId) || null
    : null;
  const selectedNodeIds = new Set<string>();
  if (selectedLink) {
    selectedNodeIds.add(selectedLink.source);
    selectedNodeIds.add(selectedLink.target);
  }

  const preparedLinks: PreparedGraphLink[] = rawLinks.map((link) => {
    const active = link.id === selectedRelationId;
    const sourceDegree = degreeMap.get(link.source) || 0;
    const targetDegree = degreeMap.get(link.target) || 0;
    return {
      ...link,
      emphasis:
        (active ? 8 : 0) +
        (selectedNodeIds.has(link.source) || selectedNodeIds.has(link.target) ? 4 : 0) +
        sourceDegree +
        targetDegree,
    };
  });

  const densityMode = options?.densityMode ?? "balanced";
  const linkLimits: Record<GraphDensityMode, number> = {
    focus: options?.preferredLinkLimit ?? 10,
    balanced: options?.preferredLinkLimit ?? 18,
    panorama: options?.preferredLinkLimit ?? 32,
  };
  const nodeLimits: Record<GraphDensityMode, number> = {
    focus: options?.preferredNodeLimit ?? 14,
    balanced: options?.preferredNodeLimit ?? 24,
    panorama: options?.preferredNodeLimit ?? 42,
  };

  const sortedLinks = [...preparedLinks].sort((left, right) => right.emphasis - left.emphasis);
  const visibleLinks =
    densityMode === "panorama" ? sortedLinks : sortedLinks.slice(0, linkLimits[densityMode]);

  const visibleNodeIds = new Set<string>();
  visibleLinks.forEach((link) => {
    visibleNodeIds.add(link.source);
    visibleNodeIds.add(link.target);
  });
  preparedNodes.forEach((node) => {
    if (node.is_seed || selectedNodeIds.has(node.id)) visibleNodeIds.add(node.id);
  });

  let visibleNodes = preparedNodes.filter((node) => visibleNodeIds.has(node.id));
  if (visibleNodes.length > nodeLimits[densityMode]) {
    const pinnedNodeIds = new Set(
      visibleNodes.filter((node) => node.is_seed || selectedNodeIds.has(node.id)).map((node) => node.id),
    );
    const rankedVisibleNodes = [...visibleNodes].sort((left, right) => right.emphasis - left.emphasis);
    const nextVisibleNodeIds = new Set<string>();
    for (const node of rankedVisibleNodes) {
      if (nextVisibleNodeIds.size >= nodeLimits[densityMode] && !pinnedNodeIds.has(node.id)) continue;
      nextVisibleNodeIds.add(node.id);
    }
    visibleNodes = rankedVisibleNodes.filter((node) => nextVisibleNodeIds.has(node.id));
  }

  const finalNodeIds = new Set(visibleNodes.map((node) => node.id));
  const finalLinks = visibleLinks.filter(
    (link) => finalNodeIds.has(link.source) && finalNodeIds.has(link.target),
  );

  return {
    nodes: visibleNodes.map((node) => ({
      ...node,
      color: selectedNodeIds.has(node.id)
        ? "#ffcf70"
        : node.color || (node.is_seed ? "#ffd166" : "#64f0cd"),
    })),
    links: finalLinks,
    stats: {
      totalNodes: rawNodes.length,
      totalLinks: rawLinks.length,
      visibleNodes: visibleNodes.length,
      visibleLinks: finalLinks.length,
    },
    nodeMap,
  };
}

function filterGraphByHops(
  rawNodes: GraphNode[],
  rawLinks: GraphLink[],
  seedIds: string[],
  maxHops = 2,
) {
  if (!seedIds.length) {
    return { nodes: rawNodes, links: rawLinks };
  }

  const adjacency = new Map<string, Set<string>>();
  rawLinks.forEach((link) => {
    if (!adjacency.has(link.source)) adjacency.set(link.source, new Set());
    if (!adjacency.has(link.target)) adjacency.set(link.target, new Set());
    adjacency.get(link.source)?.add(link.target);
    adjacency.get(link.target)?.add(link.source);
  });

  const visited = new Set<string>(seedIds);
  let frontier = new Set<string>(seedIds);

  for (let hop = 0; hop < maxHops; hop += 1) {
    const nextFrontier = new Set<string>();
    frontier.forEach((nodeId) => {
      adjacency.get(nodeId)?.forEach((neighborId) => {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          nextFrontier.add(neighborId);
        }
      });
    });
    frontier = nextFrontier;
    if (frontier.size === 0) break;
  }

  const nodes = rawNodes.filter((node) => visited.has(node.id));
  const nodeIds = new Set(nodes.map((node) => node.id));
  const links = rawLinks.filter((link) => nodeIds.has(link.source) && nodeIds.has(link.target));
  return { nodes, links };
}

function shouldRenderNodeLabel(node: PreparedGraphNode, globalScale: number) {
  if (node.is_seed) return true;
  if (node.emphasis >= 6) return true;
  if (globalScale >= 1.3 && node.degree >= 2) return true;
  return globalScale >= 2;
}

function classifyTopologyNode(node: PreparedGraphNode): TopologyNodeType {
  const source = `${node.kind || ""} ${node.label || ""} ${node.name || ""}`.toLowerCase();
  if (/(gpu|cpu|框架|工程|技术|模型|训练|检索|rag|prompt|llm|agent|embedding)/.test(source)) {
    return "tech";
  }
  if (/(人物|person|作者|用户|工程师)/.test(source)) {
    return "person";
  }
  if (/(组织|公司|团队|org|openai|meta|google|anthropic)/.test(source)) {
    return "org";
  }
  if (/(事件|发布|演进|event)/.test(source)) {
    return "event";
  }
  if (/(领域|domain|行业|应用|场景)/.test(source)) {
    return "domain";
  }
  return "concept";
}

function topologyTypeStyle(type: TopologyNodeType) {
  switch (type) {
    case "tech":
      return {
        fill: "rgba(31, 126, 173, 0.42)",
        glow: "rgba(44, 200, 255, 0.3)",
        border: "rgba(73, 200, 255, 0.72)",
      };
    case "person":
      return {
        fill: "rgba(0, 157, 123, 0.34)",
        glow: "rgba(84, 255, 188, 0.24)",
        border: "rgba(84, 255, 188, 0.58)",
      };
    case "org":
      return {
        fill: "rgba(94, 88, 223, 0.3)",
        glow: "rgba(133, 126, 255, 0.22)",
        border: "rgba(139, 133, 255, 0.55)",
      };
    case "event":
      return {
        fill: "rgba(230, 103, 61, 0.28)",
        glow: "rgba(255, 153, 107, 0.22)",
        border: "rgba(255, 162, 118, 0.54)",
      };
    case "domain":
      return {
        fill: "rgba(34, 79, 129, 0.26)",
        glow: "rgba(97, 147, 218, 0.18)",
        border: "rgba(110, 161, 233, 0.44)",
      };
    case "concept":
    default:
      return {
        fill: "rgba(66, 78, 120, 0.24)",
        glow: "rgba(137, 153, 196, 0.16)",
        border: "rgba(124, 141, 186, 0.36)",
      };
  }
}

function buildTopologyGraph(
  graph: ReturnType<typeof prepareGraphData>,
  selectedRelationId: string | null,
) {
  const nodeLinks = new Map<string, number>();
  graph.links.forEach((link) => {
    const sourceId = typeof link.source === "string" ? link.source : String(link.source);
    const targetId = typeof link.target === "string" ? link.target : String(link.target);
    nodeLinks.set(sourceId, (nodeLinks.get(sourceId) || 0) + 1);
    nodeLinks.set(targetId, (nodeLinks.get(targetId) || 0) + 1);
  });

  const focusRelation = graph.links.find((link) => link.id === selectedRelationId) || null;
  const focusNodeIds = new Set<string>();
  if (focusRelation) {
    focusNodeIds.add(String(focusRelation.source));
    focusNodeIds.add(String(focusRelation.target));
  }

  const nodes: TopologyNode[] = graph.nodes.map((node) => {
    const topologyType = classifyTopologyNode(node);
    const style = topologyTypeStyle(topologyType);
    const isFocus = focusNodeIds.has(node.id);
    const isSeed = Boolean(node.is_seed);
    return {
      ...node,
      topologyType,
      fill: isSeed ? "rgba(34, 167, 126, 0.46)" : isFocus ? "rgba(26, 126, 156, 0.62)" : style.fill,
      glow: isSeed ? "rgba(84, 255, 188, 0.24)" : isFocus ? "rgba(39, 215, 255, 0.34)" : style.glow,
      border: isSeed ? "rgba(84, 255, 188, 0.68)" : isFocus ? "rgba(68, 223, 255, 0.86)" : style.border,
      val: isSeed ? 25 : isFocus ? 20 : Math.max(13, Math.min(22, 10 + (nodeLinks.get(node.id) || 0) * 2)),
    };
  });

  const links: TopologyLink[] = graph.links.map((link) => ({
    ...link,
    stroke:
      link.id === selectedRelationId
        ? "rgba(81, 220, 255, 0.92)"
        : "rgba(139, 162, 203, 0.38)",
    textStroke:
      link.id === selectedRelationId
        ? "rgba(31, 180, 228, 0.25)"
        : "rgba(83, 95, 136, 0.18)",
  }));

  return {
    nodes,
    links,
    stats: graph.stats,
  };
}

function configureGraphForces(instance: any, mode: GraphDensityMode, dimensions: 2 | 3 = 2) {
  if (!instance?.d3Force) return;
  const chargeStrength =
    dimensions === 3
      ? mode === "focus"
        ? -520
        : mode === "balanced"
          ? -760
          : -980
      : mode === "focus"
        ? -280
        : mode === "balanced"
          ? -420
          : -560;
  const collisionRadius =
    dimensions === 3
      ? mode === "focus"
        ? 32
        : mode === "balanced"
          ? 40
          : 48
      : mode === "focus"
        ? 24
        : mode === "balanced"
          ? 30
          : 36;

  instance.d3Force("charge", forceManyBody().strength(chargeStrength));
  instance.d3Force("collision", forceCollide(collisionRadius).iterations(2));
  const linkForce = instance.d3Force("link");
  if (linkForce?.distance) {
    linkForce.distance((link: any) => {
      const active = Boolean(link?.id && link.id === instance.__selectedRelationId);
      if (dimensions === 3) {
        return active ? 260 : mode === "focus" ? 200 : mode === "balanced" ? 250 : 310;
      }
      return active ? 180 : mode === "focus" ? 140 : mode === "balanced" ? 175 : 210;
    });
    linkForce.strength(
      dimensions === 3
        ? mode === "focus"
          ? 0.08
          : 0.06
        : mode === "focus"
          ? 0.18
          : 0.12,
    );
  }
  instance.d3ReheatSimulation?.();
}

function relationToSentence(relation: RelationResult) {
  const relationText = relation.relation?.trim() || "相关";
  return `${relation.subject_name}与${relation.object_name}的关系是“${relationText}”。`;
}

function buildGraphFromResultRelations(relations: RelationResult[]) {
  const nodes = new Map<string, GraphNode>();
  const links: GraphLink[] = [];

  relations.forEach((relation) => {
    nodes.set(relation.subject_id, {
      id: relation.subject_id,
      label: relation.subject_name,
      kind: "entity",
      is_seed: (relation.matched_entity_ids?.length || 0) > 0,
    });
    nodes.set(relation.object_id, {
      id: relation.object_id,
      label: relation.object_name,
      kind: "entity",
      is_seed: (relation.matched_entity_ids?.length || 0) > 0,
    });
    links.push({
      id: relation.id,
      source: relation.subject_id,
      target: relation.object_id,
      label: relation.relation,
      matched_entity_ids: relation.matched_entity_ids,
    });
  });

  return {
    nodes: Array.from(nodes.values()),
    links,
  };
}

function getTraceGraphByStage(trace: SearchTrace, stage: QueryGraphStage) {
  if (stage === "seed") {
    return trace.trace.seed_graph || {
      nodes: trace.trace.seed_entities.map((entity) => ({
        id: entity.id,
        label: entity.name,
        kind: "entity",
        is_seed: true,
      })),
      links: [],
    };
  }

  const resultGraph = trace.trace.result_graph || trace.trace.graph;
  if (resultGraph.nodes.length > 0 || resultGraph.links.length > 0) {
    return resultGraph;
  }
  return buildGraphFromResultRelations(trace.trace.result_relations);
}

function buildSeedNeighborhoodGraph(
  trace: SearchTrace,
  maxDepth: 1 | 2 | 3,
  maxRelations: number,
  shuffleNonce: number,
) {
  const expandedGraph = trace.trace.expanded_graph;
  const fallbackGraph = trace.trace.result_graph || trace.trace.graph;
  const sourceGraph =
    expandedGraph && (expandedGraph.nodes.length > 0 || expandedGraph.links.length > 0)
      ? expandedGraph
      : fallbackGraph;
  const centerId = trace.trace.seed_entities[0]?.id;

  if (!centerId || sourceGraph.nodes.length === 0) {
    return trace.trace.seed_graph || { nodes: [], links: [] };
  }

  const adjacency = new Map<string, GraphLink[]>();
  sourceGraph.links.forEach((link) => {
    const source = String(link.source);
    const target = String(link.target);
    adjacency.set(source, [...(adjacency.get(source) || []), link]);
    adjacency.set(target, [...(adjacency.get(target) || []), link]);
  });

  const visitedNodes = new Set<string>([centerId]);
  const traversedLinkIds = new Set<string>();
  let frontier = new Set<string>([centerId]);

  for (let hop = 0; hop < maxDepth; hop += 1) {
    const nextFrontier = new Set<string>();
    frontier.forEach((nodeId) => {
      (adjacency.get(nodeId) || []).forEach((link) => {
        traversedLinkIds.add(link.id);
        const source = String(link.source);
        const target = String(link.target);
        const neighborId = source === nodeId ? target : source;
        if (!visitedNodes.has(neighborId)) {
          visitedNodes.add(neighborId);
          nextFrontier.add(neighborId);
        }
      });
    });
    frontier = nextFrontier;
    if (frontier.size === 0) break;
  }

  const traversedLinks = sourceGraph.links.filter((link) => traversedLinkIds.has(link.id));
  const shuffledLinks = [...traversedLinks]
    .map((link, index) => {
      let hash = shuffleNonce * 131 + index * 17;
      for (const char of link.id) hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
      return { link, hash };
    })
    .sort((left, right) => left.hash - right.hash)
    .map((entry) => entry.link);

  const sampledPrimaryLinks = shuffledLinks.slice(0, maxRelations);
  const visibleNodeIds = new Set<string>([centerId]);
  sampledPrimaryLinks.forEach((link) => {
    visibleNodeIds.add(String(link.source));
    visibleNodeIds.add(String(link.target));
  });

  const closureLinks = sourceGraph.links.filter((link) => {
    const source = String(link.source);
    const target = String(link.target);
    return visibleNodeIds.has(source) && visibleNodeIds.has(target);
  });

  const visibleNodes = sourceGraph.nodes
    .filter((node) => visibleNodeIds.has(node.id))
    .map((node) => ({
      ...node,
      is_seed: node.id === centerId || node.is_seed,
    }));

  return {
    nodes: visibleNodes,
    links: closureLinks,
  };
}

function buildAnswerSummary(trace: SearchTrace) {
  const relations = trace.trace.result_relations || [];
  const passages = trace.trace.grounded_passages || [];

  if (relations.length === 0) {
    return {
      answer: "当前知识库里没有检索到足够相关的关系，暂时无法给出明确回答。",
      support: passages[0]?.passage?.trim() || "",
    };
  }

  const uniqueSentences = Array.from(
    new Set(relations.slice(0, 3).map((item) => relationToSentence(item))),
  );
  const answer = uniqueSentences.join("");

  const support =
    passages.find((item) => item.matched_relation_ids?.includes(relations[0]?.id))?.passage?.trim() ||
    passages[0]?.passage?.trim() ||
    relations[0]?.passage?.trim() ||
    "";

  return { answer, support };
}

function HighlightText({
  text,
  terms,
}: {
  text: string;
  terms: string[];
}) {
  const filteredTerms = React.useMemo(
    () =>
      Array.from(new Set(terms.map((item) => item.trim()).filter((item) => item.length >= 2))).sort(
        (left, right) => right.length - left.length,
      ),
    [terms],
  );

  const parts = React.useMemo(() => {
    if (!text || filteredTerms.length === 0) return [text];
    const pattern = new RegExp(`(${filteredTerms.map(escapeRegExp).join("|")})`, "gi");
    return text.split(pattern).filter(Boolean);
  }, [filteredTerms, text]);

  if (!text) return null;

  return (
    <>
      {parts.map((part, index) => {
        const matched = filteredTerms.some((term) => term.toLowerCase() === part.toLowerCase());
        return matched ? (
          <mark key={`${part}-${index}`}>{part}</mark>
        ) : (
          <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>
        );
      })}
    </>
  );
}

function buildAnswerFromTrace(trace: SearchTrace | null) {
  if (!trace) return "";
  const topRelation = trace.trace.result_relations[0];
  if (!topRelation) return "暂时没有检索到足够可靠的答案。";

  const matchedEvidence = trace.trace.grounded_passages.find((item) =>
    item.matched_relation_ids?.includes(topRelation.id),
  );
  const evidenceText = matchedEvidence?.passage || topRelation.passage || "";

  if (evidenceText) {
    return evidenceText.length > 140 ? `${evidenceText.slice(0, 140).trim()}...` : evidenceText;
  }

  return `${topRelation.subject_name} 与 ${topRelation.object_name} 的关系是“${topRelation.relation}”。`;
}

function useElementSize<T extends HTMLElement>() {
  const ref = React.useRef<T | null>(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });

  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const update = () => {
      setSize({
        width: element.clientWidth,
        height: element.clientHeight,
      });
    };

    update();

    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size] as const;
}

const TopologyCanvas = React.forwardRef<
  GraphCanvasHandle,
  {
    width: number;
    height: number;
    nodes: TopologyNode[];
    links: TopologyLink[];
    centerNodeId: string | null;
    dimUnfocused?: boolean;
    selectedRelationId: string | null;
    hoveredNodeId: string | null;
    hoveredLinkId: string | null;
    onHoveredNodeChange: (value: string | null) => void;
    onHoveredLinkChange: (value: string | null) => void;
    onSelectedRelationChange: (value: string | null) => void;
    onScaleChange: (value: number) => void;
  }
>(function TopologyCanvas(
  {
    width,
    height,
    nodes,
    links,
    centerNodeId,
    dimUnfocused = true,
    selectedRelationId,
    hoveredNodeId,
    hoveredLinkId,
    onHoveredNodeChange,
    onHoveredLinkChange,
    onSelectedRelationChange,
    onScaleChange,
  },
  ref,
) {
  const svgRef = React.useRef<SVGSVGElement | null>(null);
  const simulationRef = React.useRef<any>(null);
  const positionCacheRef = React.useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map());
  const [simNodes, setSimNodes] = React.useState<SimulatedTopologyNode[]>([]);
  const [simLinks, setSimLinks] = React.useState<SimulatedTopologyLink[]>([]);
  const [zoomScale, setZoomScale] = React.useState(1);
  const [zoomOffset, setZoomOffset] = React.useState({ x: 0, y: 0 });
  const [draggedNodeId, setDraggedNodeId] = React.useState<string | null>(null);
  const [isPanning, setIsPanning] = React.useState(false);
  const pointerRef = React.useRef({ x: 0, y: 0 });

  const connectedNodeIds = React.useMemo(() => {
    const ids = new Set<string>();
    const activeLinkId = hoveredLinkId || selectedRelationId;
    if (centerNodeId) ids.add(centerNodeId);
    if (hoveredNodeId) {
      ids.add(hoveredNodeId);
      simLinks.forEach((link) => {
        if (link.source.id === hoveredNodeId || link.target.id === hoveredNodeId) {
          ids.add(link.source.id);
          ids.add(link.target.id);
        }
      });
    }
    if (activeLinkId) {
      const activeLink = simLinks.find((link) => link.id === activeLinkId);
      if (activeLink) {
        ids.add(activeLink.source.id);
        ids.add(activeLink.target.id);
      }
    }
    return ids;
  }, [centerNodeId, hoveredLinkId, hoveredNodeId, selectedRelationId, simLinks]);

  React.useEffect(() => {
    onScaleChange(Math.round(zoomScale * 100));
  }, [onScaleChange, zoomScale]);

  React.useEffect(() => {
    if (!width || !height || nodes.length === 0) {
      setSimNodes([]);
      setSimLinks([]);
      simulationRef.current?.stop?.();
      return;
    }

    const nodeCopies: SimulatedTopologyNode[] = nodes.map((node, index) => {
      const cached = positionCacheRef.current.get(node.id);
      const ringAngle = (index / Math.max(nodes.length, 1)) * Math.PI * 2;
      const radius = node.id === centerNodeId ? 0 : 130 + (index % 4) * 28;
      return {
        ...node,
        x: cached?.x ?? width / 2 + Math.cos(ringAngle) * radius,
        y: cached?.y ?? height / 2 + Math.sin(ringAngle) * radius,
        vx: cached?.vx ?? 0,
        vy: cached?.vy ?? 0,
      };
    });

    const nodeMap = new Map(nodeCopies.map((node) => [node.id, node]));
    const linkCopies: SimulatedTopologyLink[] = links
      .map((link) => {
        const source = nodeMap.get(String(link.source));
        const target = nodeMap.get(String(link.target));
        if (!source || !target) return null;
        return {
          id: link.id,
          source,
          target,
          label: link.label,
          stroke: link.stroke,
          textStroke: link.textStroke,
        };
      })
      .filter(Boolean) as SimulatedTopologyLink[];

    const simulation = forceSimulation(nodeCopies as any)
      .force(
        "link",
        forceLink(linkCopies as any)
          .id((d: any) => d.id)
          .distance((link: any) => {
            if (link.source?.id === centerNodeId || link.target?.id === centerNodeId) return 118;
            return 152;
          })
          .strength((link: any) => {
            if (link.source?.id === centerNodeId || link.target?.id === centerNodeId) return 0.32;
            return 0.12;
          }),
      )
      .force("charge", forceManyBody().strength(-320))
      .force("center", forceCenter(width / 2, height / 2).strength(0.09))
      .force(
        "collision",
        forceCollide().radius((node: any) => {
          const simulatedNode = node as SimulatedTopologyNode;
          return (simulatedNode.val || 20) + 24;
        }),
      )
      .alpha(0.9)
      .alphaDecay(0.06)
      .alphaMin(0.006);

    simulation.on("tick", () => {
      nodeCopies.forEach((node) => {
        positionCacheRef.current.set(node.id, {
          x: node.x,
          y: node.y,
          vx: node.vx || 0,
          vy: node.vy || 0,
        });
      });
      setSimNodes([...nodeCopies]);
      setSimLinks([...linkCopies]);
    });

    simulationRef.current = simulation;
    return () => simulation.stop();
  }, [centerNodeId, height, links, nodes, width]);

  const toCanvasPoint = React.useCallback(
    (clientX: number, clientY: number) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return {
        x: (clientX - rect.left - zoomOffset.x) / zoomScale,
        y: (clientY - rect.top - zoomOffset.y) / zoomScale,
      };
    },
    [zoomOffset.x, zoomOffset.y, zoomScale],
  );

  const zoomAt = React.useCallback(
    (nextScale: number, anchor?: { x: number; y: number }) => {
      const clamped = Math.max(0.2, Math.min(5.2, nextScale));
      const focus = anchor || { x: width / 2, y: height / 2 };
      setZoomOffset((current) => ({
        x: focus.x - ((focus.x - current.x) / zoomScale) * clamped,
        y: focus.y - ((focus.y - current.y) / zoomScale) * clamped,
      }));
      setZoomScale(clamped);
    },
    [height, width, zoomScale],
  );

  React.useImperativeHandle(
    ref,
    () => ({
      zoomIn: () => zoomAt(zoomScale * 1.3),
      zoomOut: () => zoomAt(zoomScale / 1.3),
      resetView: () => {
        setZoomScale(1);
        setZoomOffset({ x: 0, y: 0 });
        simulationRef.current?.alpha?.(0.75)?.restart?.();
      },
    }),
    [zoomAt, zoomScale],
  );

  React.useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) return;

    const handleNativeWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = svgElement.getBoundingClientRect();
      const anchor = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      zoomAt(zoomScale * (event.deltaY < 0 ? 1.12 : 0.88), anchor);
    };

    svgElement.addEventListener("wheel", handleNativeWheel, { passive: false });
    return () => {
      svgElement.removeEventListener("wheel", handleNativeWheel);
    };
  }, [zoomAt, zoomScale]);

  const handleSvgMouseDown = (event: React.MouseEvent<SVGSVGElement>) => {
    pointerRef.current = { x: event.clientX, y: event.clientY };
    setIsPanning(true);
    onHoveredNodeChange(null);
    onHoveredLinkChange(null);
  };

  const handleNodeMouseDown = (event: React.MouseEvent, node: SimulatedTopologyNode) => {
    event.stopPropagation();
    pointerRef.current = { x: event.clientX, y: event.clientY };
    node.fx = node.x;
    node.fy = node.y;
    setDraggedNodeId(node.id);
    simulationRef.current?.alpha?.(0.22)?.restart?.();
    onHoveredNodeChange(node.id);
  };

  const handleMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (draggedNodeId) {
      const point = toCanvasPoint(event.clientX, event.clientY);
      const node = simNodes.find((entry) => entry.id === draggedNodeId);
      if (!node) return;
      node.fx = point.x;
      node.fy = point.y;
      node.x = point.x;
      node.y = point.y;
      setSimNodes([...simNodes]);
      return;
    }

    if (isPanning) {
      const deltaX = event.clientX - pointerRef.current.x;
      const deltaY = event.clientY - pointerRef.current.y;
      pointerRef.current = { x: event.clientX, y: event.clientY };
      setZoomOffset((current) => ({ x: current.x + deltaX, y: current.y + deltaY }));
    }
  };

  const releaseDrag = React.useCallback(() => {
    if (draggedNodeId) {
      const node = simNodes.find((entry) => entry.id === draggedNodeId);
      if (node) {
        node.fx = null;
        node.fy = null;
      }
      simulationRef.current?.alpha?.(0.18)?.restart?.();
    }
    setDraggedNodeId(null);
    setIsPanning(false);
  }, [draggedNodeId, simNodes]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      onMouseDown={handleSvgMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={releaseDrag}
      onMouseLeave={releaseDrag}
    >
      <defs>
        <marker id="graph-arrow-normal" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="8" markerHeight="8" orient="auto">
          <path d="M 0 0 L 12 6 L 0 12 z" fill="rgba(139, 162, 203, 0.75)" />
        </marker>
        <marker id="graph-arrow-active" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="9" markerHeight="9" orient="auto">
          <path d="M 0 0 L 12 6 L 0 12 z" fill="#56dfff" />
        </marker>
      </defs>
      <g transform={`translate(${zoomOffset.x}, ${zoomOffset.y}) scale(${zoomScale})`}>
        {simLinks.map((link) => {
          const isActive = hoveredLinkId === link.id || selectedRelationId === link.id;
          const touchesHoveredNode = Boolean(
            hoveredNodeId && (link.source.id === hoveredNodeId || link.target.id === hoveredNodeId),
          );
          const shouldHighlight = isActive || touchesHoveredNode;
          const isCenterLink = Boolean(centerNodeId && (link.source.id === centerNodeId || link.target.id === centerNodeId));
          const opacity = dimUnfocused
            ? shouldHighlight
              ? 1
              : isCenterLink
                ? 0.46
                : 0.12
            : shouldHighlight
              ? 1
              : isCenterLink
                ? 0.82
                : 0.68;
          const midX = (link.source.x + link.target.x) / 2;
          const midY = (link.source.y + link.target.y) / 2;
          const labelWidth = Math.max(link.label.length * 8.4, 54);

          return (
            <g
              key={link.id}
              style={{ opacity }}
              onMouseEnter={() => {
                onHoveredLinkChange(link.id);
                onSelectedRelationChange(link.id);
              }}
              onMouseLeave={() => onHoveredLinkChange(null)}
            >
              <line
                className={shouldHighlight ? "graph-link-line graph-link-line-active" : "graph-link-line"}
                x1={link.source.x}
                y1={link.source.y}
                x2={link.target.x}
                y2={link.target.y}
                stroke={shouldHighlight ? "#56dfff" : link.stroke}
                strokeWidth={shouldHighlight ? 2.8 : isCenterLink ? 1.9 : 1.2}
                strokeDasharray={shouldHighlight ? "8 7" : "6 8"}
                markerEnd={shouldHighlight ? "url(#graph-arrow-active)" : "url(#graph-arrow-normal)"}
              />
              <rect
                x={midX - labelWidth / 2}
                y={midY - 10}
                width={labelWidth}
                height={20}
                rx={6}
                fill={shouldHighlight ? "rgba(10, 26, 46, 0.95)" : "rgba(10, 19, 38, 0.66)"}
                stroke={shouldHighlight ? "rgba(86, 223, 255, 0.28)" : link.textStroke}
                strokeWidth={1}
              />
              <text
                x={midX}
                y={midY + 3}
                textAnchor="middle"
                fontSize="12px"
                fill={shouldHighlight ? "#8cf2ff" : "rgba(196, 209, 237, 0.72)"}
                pointerEvents="none"
              >
                {link.label}
              </text>
            </g>
          );
        })}

        {simNodes.map((node) => {
          const isCenter = node.id === centerNodeId;
          const isHovered = hoveredNodeId === node.id;
          const isConnected = connectedNodeIds.has(node.id);
          const opacity = dimUnfocused ? (isCenter ? 1 : isHovered || isConnected ? 0.94 : 0.2) : 0.94;
          const radius = Math.max(isCenter ? 34 : 22, node.val);
          const fontSize = isCenter ? 18 : 12;
          const nodeFill = isCenter ? "rgba(206, 145, 31, 0.9)" : node.fill;
          const nodeGlow = isCenter ? "rgba(255, 196, 79, 0.3)" : node.glow;
          const nodeBorder = isCenter ? "rgba(255, 220, 117, 0.96)" : node.border;
          const nodeText = isCenter ? "#fff1b3" : "rgba(216, 225, 246, 0.88)";

          return (
            <g
              key={node.id}
              transform={`translate(${node.x}, ${node.y})`}
              style={{ opacity }}
              onMouseDown={(event) => handleNodeMouseDown(event, node)}
              onMouseEnter={() => onHoveredNodeChange(node.id)}
              onMouseLeave={() => onHoveredNodeChange(null)}
            >
              <circle r={radius + 18} fill={nodeGlow} />
              <circle
                r={radius}
                fill={nodeFill}
                stroke={isCenter ? "#ffd76b" : nodeBorder}
                strokeWidth={isCenter ? 3 : 2}
              />
              <circle r={Math.max(radius * 0.16, 3)} cy={-radius + 6} fill="rgba(189, 210, 255, 0.44)" />
              {isCenter ? (
                <circle
                  r={radius + 12}
                  fill="none"
                  stroke="rgba(255, 214, 103, 0.72)"
                  strokeWidth={2}
                  strokeDasharray="8 7"
                />
              ) : null}
              <text
                textAnchor="middle"
                y="4"
                fontSize={`${fontSize}px`}
                fill={nodeText}
                pointerEvents="none"
              >
                {node.displayLabel}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
});

function toUploadedDocument(file: UploadedFile): UploadedDocument {
  return {
    id: file.id,
    name: file.name,
    path: file.path,
    type: file.extension.toUpperCase(),
    uploadedAt: formatUploadedAt(file.uploaded_at),
    size: formatFileSize(file.size),
    selected: false,
  };
}

function statusTone(status: string) {
  if (status === "completed") return "good";
  if (status === "partial_failed") return "warn";
  if (status === "failed") return "danger";
  return "neutral";
}

function App() {
  const [module, setModule] = React.useState<ModuleKey>("overview");
  const [catalog, setCatalog] = React.useState<LibraryRecord[]>([]);
  const [catalogLoading, setCatalogLoading] = React.useState(true);
  const [selectedKbId, setSelectedKbId] = React.useState("");
  const [overview, setOverview] = React.useState<LibraryOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = React.useState(false);
  const [documents, setDocuments] = React.useState<UploadedDocument[]>([]);
  const [builderTask, setBuilderTask] = React.useState<DatabaseBuildProgress | null>(null);
  const [builderKbId, setBuilderKbId] = React.useState("");
  const [builderError, setBuilderError] = React.useState("");
  const [builderLoading, setBuilderLoading] = React.useState(false);
  const [mineruGpus, setMineruGpus] = React.useState("0");
  const [workersPerGpu, setWorkersPerGpu] = React.useState(1);
  const [extractConcurrency, setExtractConcurrency] = React.useState(10);
  const [embeddingConcurrency, setEmbeddingConcurrency] = React.useState(10);
  const [maxChunkChars, setMaxChunkChars] = React.useState(1500);
  const [chunkOverlapChars, setChunkOverlapChars] = React.useState(150);
  const [queryText, setQueryText] = React.useState("");
  const [queryTrace, setQueryTrace] = React.useState<SearchTrace | null>(null);
  const [queryAnswer, setQueryAnswer] = React.useState<RagAnswerResponse | null>(null);
  const [queryLoading, setQueryLoading] = React.useState(false);
  const [queryError, setQueryError] = React.useState("");
  const [queryGraphStage, setQueryGraphStage] = React.useState<QueryGraphStage>("seed");

  const loadCatalog = React.useCallback(async () => {
    setCatalogLoading(true);
    try {
      const response = await fetch(
        apiUrl(`/database_build/catalog?milvus_db=${DEFAULT_MILVUS_DB}&user_id=${DEFAULT_USER_ID}`),
      );
      const data = await response.json();
      const libraries = (data.libraries || []) as LibraryRecord[];
      setCatalog(libraries);
      if (!selectedKbId && libraries[0]) {
        setSelectedKbId(libraries[0].kb_id);
        setBuilderKbId(libraries[0].kb_id);
      }
    } finally {
      setCatalogLoading(false);
    }
  }, [selectedKbId]);

  const loadOverview = React.useCallback(async (kbId: string) => {
    if (!kbId) return;
    setOverviewLoading(true);
    try {
      const response = await fetch(
        apiUrl(
          `/database_build/library/${encodeURIComponent(kbId)}/overview?milvus_db=${DEFAULT_MILVUS_DB}&user_id=${DEFAULT_USER_ID}`,
        ),
      );
      if (!response.ok) throw new Error("概览加载失败");
      const data = (await response.json()) as LibraryOverview;
      setOverview(data);
    } catch {
      setOverview(null);
    } finally {
      setOverviewLoading(false);
    }
  }, []);

  const loadUploads = React.useCallback(async () => {
    const response = await fetch(apiUrl("/files"));
    const data = await response.json();
    setDocuments(((data.files || []) as UploadedFile[]).map(toUploadedDocument));
  }, []);

  const loadLatestBuildTask = React.useCallback(async (kbId?: string) => {
    const query = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
    const response = await fetch(apiUrl(`/database_build/latest${query}`));
    if (!response.ok) {
      throw new Error("暂无构建任务");
    }
    return (await response.json()) as DatabaseBuildProgress;
  }, []);

  React.useEffect(() => {
    void loadCatalog();
    void loadUploads();
  }, [loadCatalog, loadUploads]);

  React.useEffect(() => {
    if (selectedKbId) {
      void loadOverview(selectedKbId);
    }
  }, [selectedKbId, loadOverview]);

  const selectedLibrary =
    catalog.find((item) => item.kb_id === selectedKbId) || catalog[0] || null;

  const pollBuildProgress = React.useCallback(
    async (taskId: string) => {
      const response = await fetch(apiUrl(`/database_build/progress/${taskId}`));
      const data = (await response.json()) as DatabaseBuildProgress;
      setBuilderTask(data);
      if (!["completed", "failed", "partial_failed"].includes(data.status)) {
        window.setTimeout(() => {
          void pollBuildProgress(taskId);
        }, 1500);
      } else {
        void loadCatalog();
        if (builderKbId) {
          setSelectedKbId(builderKbId);
          void loadOverview(builderKbId);
        }
      }
    },
    [builderKbId, loadCatalog, loadOverview],
  );

  React.useEffect(() => {
    if (!builderKbId) return;
    let cancelled = false;

    const restoreBuildTask = async () => {
      try {
        const latestTask = await loadLatestBuildTask(builderKbId);
        if (cancelled) return;
        setBuilderTask(latestTask);
        if (!["completed", "failed", "partial_failed"].includes(latestTask.status)) {
          void pollBuildProgress(latestTask.task_id);
        }
      } catch {
        if (!cancelled) {
          setBuilderTask((current) => (current?.kb_id === builderKbId ? current : null));
        }
      }
    };

    void restoreBuildTask();
    return () => {
      cancelled = true;
    };
  }, [builderKbId, loadLatestBuildTask, pollBuildProgress]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files?.length) return;
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("files", file));
    await fetch(apiUrl("/files/upload"), { method: "POST", body: formData });
    await loadUploads();
  };

  const toggleDocument = (id: string) => {
    setDocuments((current) =>
      current.map((doc) => (doc.id === id ? { ...doc, selected: !doc.selected } : doc)),
    );
  };

  const startBuild = async () => {
    const kbId = builderKbId.trim();
    const selectedFiles = documents.filter((doc) => doc.selected).map((doc) => doc.path);

    setBuilderError("");

    if (!kbId) {
      setBuilderError("请填写知识库 ID");
      return;
    }
    if (selectedFiles.length === 0) {
      setBuilderError("请先勾选至少一个文件");
      return;
    }

    setBuilderLoading(true);
    setBuilderTask(null);
    try {
      const resp = await fetch(apiUrl("/database_build/submit"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_paths: selectedFiles,
          kb_id: kbId,
          user_id: DEFAULT_USER_ID,
          milvus_db: DEFAULT_MILVUS_DB,
          output_root: "/opt/Workspace/CRX/NextGraph/backend/storage/mineru_output",
          gpus: mineruGpus
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
          workers_per_gpu: workersPerGpu,
          extract_concurrency: extractConcurrency,
          embedding_concurrency: embeddingConcurrency,
          max_chunk_chars: maxChunkChars,
          chunk_overlap_chars: chunkOverlapChars,
        }),
      });
      if (!resp.ok) {
        const errorText = await resp.text();
        throw new Error(errorText || "启动构建失败");
      }
      const data = await resp.json();
      if (!data?.task_id) {
        throw new Error("后端没有返回 task_id");
      }
      await pollBuildProgress(data.task_id);
    } catch (e: any) {
      setBuilderError("错误: " + e.message);
    } finally {
      setBuilderLoading(false);
    }
  };

  const runTraceSearch = React.useCallback(async () => {
    if (!selectedKbId || !queryText.trim()) return;
    setQueryLoading(true);
    setQueryError("");
    setQueryAnswer(null);
    try {
      const payload = {
        query: queryText,
        user_id: DEFAULT_USER_ID,
        kb_id: selectedKbId,
        mode: "hybrid",
        top_k: 12,
        entity_top_k: 10,
        relation_top_k: 14,
        expansion_degree: 2,
      };
      const [answerResponse, traceResponse] = await Promise.all([
        fetch(apiUrl("/api/search/answer"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            answer_top_k: 6,
            passage_top_k: 6,
          }),
        }),
        fetch(apiUrl("/api/search/trace"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      ]);
      if (!answerResponse.ok || !traceResponse.ok) {
        throw new Error("查询失败");
      }
      const [answerData, traceData] = (await Promise.all([
        answerResponse.json(),
        traceResponse.json(),
      ])) as [RagAnswerResponse, SearchTrace];
      setQueryAnswer(answerData);
      setQueryTrace(traceData);
    } catch {
      setQueryAnswer(null);
      setQueryTrace(null);
      setQueryError("查询失败，请检查后端搜索服务与向量库连接。");
    } finally {
      setQueryLoading(false);
    }
  }, [queryText, selectedKbId]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <img src={appLogo} alt="NextGraph" />
          <div>
            <strong>NextGraph</strong>
            <span>Knowledge Graph Workbench</span>
          </div>
        </div>
        <nav className="tab-nav">
          <button
            className={module === "overview" ? "active" : ""}
            onClick={() => setModule("overview")}
          >
            <Database size={16} />
            知识库总览
          </button>
          <button
            className={module === "builder" ? "active" : ""}
            onClick={() => setModule("builder")}
          >
            <FolderUp size={16} />
            新建知识库
          </button>
          <button
            className={module === "query" ? "active" : ""}
            onClick={() => setModule("query")}
          >
            <MessageSquareShare size={16} />
            图谱问答
          </button>
        </nav>
        <div className="status-chip">
          <span className="dot" />
          系统就绪
          <small style={{ opacity: 0.5, marginLeft: "6px", fontSize: "11px" }}>
            ({DEFAULT_MILVUS_DB})
          </small>
        </div>
      </header>

      <main className="workspace">
        {module === "overview" && (
          <OverviewModule
            catalog={catalog}
            catalogLoading={catalogLoading}
            overview={overview}
            overviewLoading={overviewLoading}
            selectedKbId={selectedKbId}
            onSelectKb={(kbId) => {
              setSelectedKbId(kbId);
              setBuilderKbId(kbId);
            }}
            onJump={(next) => setModule(next)}
          />
        )}

        {module === "builder" && (
          <BuilderModule
            builderKbId={builderKbId}
            mineruGpus={mineruGpus}
            workersPerGpu={workersPerGpu}
            extractConcurrency={extractConcurrency}
            embeddingConcurrency={embeddingConcurrency}
            maxChunkChars={maxChunkChars}
            chunkOverlapChars={chunkOverlapChars}
            onBuilderKbIdChange={setBuilderKbId}
            onMineruGpusChange={setMineruGpus}
            onWorkersPerGpuChange={setWorkersPerGpu}
            onExtractConcurrencyChange={setExtractConcurrency}
            onEmbeddingConcurrencyChange={setEmbeddingConcurrency}
            onMaxChunkCharsChange={setMaxChunkChars}
            onChunkOverlapCharsChange={setChunkOverlapChars}
            documents={documents}
            task={builderTask}
            error={builderError}
            loading={builderLoading}
            onUpload={handleUpload}
            onToggleDocument={toggleDocument}
            onStartBuild={startBuild}
          />
        )}

        {module === "query" && (
          <QueryModule
            selectedKbId={selectedKbId}
            catalog={catalog}
            trace={queryTrace}
            answer={queryAnswer}
            loading={queryLoading}
            query={queryText}
            queryError={queryError}
            graphStage={queryGraphStage}
            onSelectKb={setSelectedKbId}
            onQueryChange={setQueryText}
            onGraphStageChange={setQueryGraphStage}
            onSearch={runTraceSearch}
          />
        )}
      </main>
    </div>
  );
}

function OverviewModule({
  catalog,
  catalogLoading,
  overview,
  overviewLoading,
  selectedKbId,
  onSelectKb,
  onJump,
}: {
  catalog: LibraryRecord[];
  catalogLoading: boolean;
  overview: LibraryOverview | null;
  overviewLoading: boolean;
  selectedKbId: string;
  onSelectKb: (kbId: string) => void;
  onJump: (module: ModuleKey) => void;
}) {
  const [graphMode, setGraphMode] = React.useState<"2D" | "3D">("2D");
  const [overviewGraphRef, overviewGraphSize] = useElementSize<HTMLDivElement>();
  const overviewGraphInstanceRef = React.useRef<any>(null);
  const totals = catalog.reduce(
    (acc, item) => ({
      libraries: acc.libraries + 1,
      files: acc.files + item.file_count,
      entities: acc.entities + item.entity_count,
      relations: acc.relations + item.relation_count,
    }),
    { libraries: 0, files: 0, entities: 0, relations: 0 },
  );

  const graph = React.useMemo(() => {
    const raw = overview?.milvus_preview?.graph_preview || { nodes: [], links: [] };
    return prepareGraphData(raw.nodes, raw.links, {
      densityMode: "balanced",
      preferredNodeLimit: 18,
      preferredLinkLimit: 16,
    });
  }, [overview]);

  React.useEffect(() => {
    if (!overviewGraphInstanceRef.current || graphMode !== "2D" || graph.nodes.length === 0) return;
    configureGraphForces(overviewGraphInstanceRef.current, "balanced");
    overviewGraphInstanceRef.current.__selectedRelationId = null;
    const timer = window.setTimeout(() => {
      overviewGraphInstanceRef.current?.zoomToFit?.(600, 80);
      overviewGraphInstanceRef.current?.centerAt?.();
    }, 200);
    return () => window.clearTimeout(timer);
  }, [graph.links.length, graph.nodes.length, graphMode]);

  return (
    <section className="module-grid">
      <div className="hero-panel">
        <div>
          <span className="eyebrow">Knowledge Base</span>
          <h1>知识资产全景洞察</h1>
          <p>实时监控多个知识库的数据规模、图谱体量与连接状态，从宏观视角掌控您的图谱生态。</p>
        </div>
        <div className="hero-actions">
          <button onClick={() => onJump("builder")}>
            <Zap size={16} />
            新建并构建
          </button>
          <button className="ghost" onClick={() => onJump("query")}>
            <Search size={16} />
            进入问答
          </button>
        </div>
      </div>

      <div className="metric-strip">
        <MetricCard title="知识库数量" value={String(totals.libraries)} />
        <MetricCard title="文件总数" value={String(totals.files)} />
        <MetricCard title="实体总数" value={String(totals.entities)} />
        <MetricCard title="关系总数" value={String(totals.relations)} />
      </div>

      <div className="overview-layout">
        <section className="library-list panel">
          <div className="panel-head">
            <h2>知识库目录</h2>
            <span>{catalog.length} 个库</span>
          </div>
          {catalogLoading ? (
            <div className="empty-panel">正在读取知识库目录…</div>
          ) : (
            <div className="library-cards">
              {catalog.map((item) => (
                <button
                  key={item.kb_id}
                  className={`library-card ${item.kb_id === selectedKbId ? "selected" : ""}`}
                  onClick={() => onSelectKb(item.kb_id)}
                >
                  <div className="card-main">
                    <div className="card-icon-area">
                      <div className="hex-icon">
                        <Database size={20} />
                      </div>
                    </div>
                    <div className="card-info">
                      <div className="card-row">
                        <strong>{item.kb_id}</strong>
                        <span className={`pill ${statusTone(item.latest_status)}`}>{item.latest_status}</span>
                      </div>
                      <span className="latest-msg">{item.latest_message}</span>
                    </div>
                  </div>
                  <div className="library-metrics">
                    <label>文件 {item.file_count}</label>
                    <label>实体 {item.entity_count}</label>
                    <label>关系 {item.relation_count}</label>
                  </div>
                  <div className="library-progress">
                    <div className="progress-track">
                      <div style={{ width: `${item.completion_ratio * 100}%` }} />
                    </div>
                    <span>最近更新 {formatTime(item.updated_at)}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="library-detail panel">
          <div className="panel-head">
            <h2>知识库画像</h2>
            <span>{overview?.kb_id || selectedKbId || "未选择"}</span>
          </div>
          <div className="library-detail-scroll">
            {overviewLoading ? (
              <div className="empty-panel">正在读取图谱概览…</div>
            ) : !overview ? (
              <div className="empty-panel">选择一个知识库查看详情。</div>
            ) : (
              <>
              <div className="detail-stats">
                <MetricCard title="文档" value={String(overview.summary.file_count)} compact />
                <MetricCard
                  title="实体"
                  value={String(overview.milvus_preview?.counts.entities ?? overview.summary.entity_count)}
                  compact
                />
                <MetricCard
                  title="关系"
                  value={String(overview.milvus_preview?.counts.relations ?? overview.summary.relation_count)}
                  compact
                />
                <MetricCard
                  title="段落证据"
                  value={String(overview.milvus_preview?.counts.passages ?? 0)}
                  compact
                />
              </div>

              <div className="detail-body">
                <div className="preview-card">
                  <div className="subhead">
                    <span>图谱预览</span>
                    <div className="graph-panel-meta">
                    <button
                      className="graph-mode-toggle"
                      onClick={() => setGraphMode(graphMode === "2D" ? "3D" : "2D")}
                    >
                      <span className={`graph-mode-badge ${graphMode === "3D" ? "mode-3d" : "mode-2d"}`}>
                        {graphMode}
                      </span>
                    </button>
                    </div>
                  </div>
                <div className="mini-graph" ref={overviewGraphRef}>
                  {overviewGraphSize.width > 0 ? (
                  graphMode === "2D" ? (
                  <ForceGraph2D
                    ref={overviewGraphInstanceRef}
                    graphData={graph}
                    backgroundColor="transparent"
                    width={overviewGraphSize.width}
                    height={320}
                    nodeRelSize={7}
                    nodeVal={(node) => (node as PreparedGraphNode).val}
                    cooldownTicks={0}
                    warmupTicks={80}
                    d3AlphaDecay={0.035}
                    d3VelocityDecay={0.26}
                    minZoom={0.3}
                    maxZoom={5}
                    nodeColor={(node) => ((node as PreparedGraphNode).color || "#5fe3c1")}
                    linkColor={() => "rgba(103, 204, 255, 0.28)"}
                    linkDirectionalParticles={1}
                    linkDirectionalParticleWidth={1.2}
                    linkDirectionalParticleSpeed={0.007}
                    onEngineStop={() => {
                      overviewGraphInstanceRef.current?.zoomToFit?.(500, 80);
                    }}
                    nodeCanvasObject={(node, ctx, globalScale) => {
                      const graphNode = node as PreparedGraphNode;
                      const label = graphNode.displayLabel;
                      const fontSize = Math.max(11 / globalScale, 4);
                      ctx.font = `${fontSize}px "Space Grotesk", sans-serif`;

                      const r = Math.max(graphNode.is_seed ? 8 : 5.5, graphNode.val / 1.8);
                      ctx.beginPath();
                      ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI, false);
                      ctx.fillStyle = graphNode.color || "#5fe3c1";
                      ctx.shadowBlur = 18 / globalScale;
                      ctx.shadowColor = graphNode.color || "#5fe3c1";
                      ctx.fill();
                      ctx.shadowBlur = 0;

                      if (!shouldRenderNodeLabel(graphNode, globalScale)) return;
                      const textWidth = ctx.measureText(label).width;
                      const bckgDimensions = [textWidth, fontSize].map((n) => n + fontSize * 0.55);

                      ctx.fillStyle = "rgba(5, 10, 20, 0.75)";
                      ctx.fillRect(
                        node.x! - bckgDimensions[0] / 2,
                        node.y! + r + 4,
                        bckgDimensions[0],
                        bckgDimensions[1],
                      );
                      ctx.textAlign = "center";
                      ctx.textBaseline = "top";
                      ctx.fillStyle = "#ffffff";
                      ctx.fillText(label, node.x!, node.y! + r + 5);
                    }}
                    nodePointerAreaPaint={(node, color, ctx) => {
                      ctx.fillStyle = color;
                      ctx.beginPath();
                      ctx.arc(node.x!, node.y!, 11, 0, 2 * Math.PI, false);
                      ctx.fill();
                    }}
                    linkCanvasObjectMode={() => "after"}
                    linkCanvasObject={(link, ctx, globalScale) => {
                      const fontSize = Math.min(9, 11 / globalScale);
                      ctx.font = `${fontSize}px Sans-Serif`;

                      const start = link.source as any;
                      const end = link.target as any;
                      if (typeof start !== "object" || typeof end !== "object") return;

                      const rel = (link as any).label || "";
                      const textPos = {
                        x: start.x + (end.x - start.x) * 0.5,
                        y: start.y + (end.y - start.y) * 0.5
                      };

                      const textAngle = Math.atan2(end.y - start.y, end.x - start.x);

                      const dist = Math.hypot(end.x - start.x, end.y - start.y);
                      if (dist < 90 || globalScale < 0.9) return;

                      ctx.save();
                      ctx.translate(textPos.x, textPos.y);
                      ctx.rotate(textAngle);
                      ctx.textAlign = "center";
                      ctx.textBaseline = "middle";
                      ctx.fillStyle = "rgba(160, 210, 255, 0.72)";
                      ctx.fillText(rel, 0, -2);
                      ctx.restore();
                    }}
                  />
                  ) : (
                  <ForceGraph3D
                    backgroundColor="rgba(0,0,0,0)"
                    graphData={graph}
                    width={overviewGraphSize.width}
                    height={320}
                    nodeRelSize={7}
                    nodeVal={(node) => (node as PreparedGraphNode).val}
                    nodeColor={(node) => ((node as PreparedGraphNode).color || "#5fe3c1")}
                    linkColor={() => "rgba(103, 204, 255, 0.4)"}
                    linkDirectionalParticles={1}
                    nodeThreeObject={(node) => {
                      const graphNode = node as PreparedGraphNode;
                      const label = shouldRenderNodeLabel(graphNode, 1.2) ? graphNode.displayLabel : "";
                      const isSeed = Boolean(graphNode.is_seed);
                      const sprite = new SpriteText(label);
                      sprite.color = graphNode.color || (isSeed ? "#f9d66b" : "#5fe3c1");
                      sprite.textHeight = isSeed ? 10 : 6.5;
                      sprite.padding = 2;
                      sprite.backgroundColor = "rgba(5, 11, 20, 0.75)";
                      sprite.borderRadius = 2;
                      return sprite;
                    }}
                    nodeThreeObjectExtend={true}
                  />
                  )
                  ) : null}
                </div>
                </div>

                <div className="preview-card">
                  <div className="subhead">
                    <span>高连接实体</span>
                    <span>样本 Top</span>
                  </div>
                  <div className="ranking-list">
                    {(overview.milvus_preview?.top_entities || []).slice(0, 6).map((item) => (
                      <div key={item.id} className="ranking-row">
                        <strong>{item.name}</strong>
                        <div>
                          <div className="mini-bar">
                            <div style={{ width: `${Math.min(item.relation_count * 10, 100)}%` }} />
                          </div>
                          <span>{item.relation_count} 关系</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {overview.summary.source_documents?.length ? (
                <div className="source-grid">
                  {overview.summary.source_documents.map((doc) => (
                    <div key={doc.path} className="source-card">
                      <strong>{doc.name}</strong>
                      <span>{doc.state}</span>
                      <small>
                        chunks {doc.summary?.chunks || 0} / entities {doc.summary?.entities || 0} / relations{" "}
                        {doc.summary?.relations || 0}
                      </small>
                    </div>
                  ))}
                </div>
              ) : null}
              </>
            )}
          </div>
        </section>
      </div>
    </section>
  );
}

function BuilderModule({
  builderKbId,
  mineruGpus,
  workersPerGpu,
  extractConcurrency,
  embeddingConcurrency,
  maxChunkChars,
  chunkOverlapChars,
  onMineruGpusChange,
  onWorkersPerGpuChange,
  onExtractConcurrencyChange,
  onEmbeddingConcurrencyChange,
  onMaxChunkCharsChange,
  onChunkOverlapCharsChange,
  onBuilderKbIdChange,
  documents,
  task,
  error,
  loading,
  onUpload,
  onToggleDocument,
  onStartBuild,
}: {
  builderKbId: string;
  mineruGpus: string;
  workersPerGpu: number;
  extractConcurrency: number;
  embeddingConcurrency: number;
  maxChunkChars: number;
  chunkOverlapChars: number;
  onBuilderKbIdChange: (value: string) => void;
  onMineruGpusChange: (value: string) => void;
  onWorkersPerGpuChange: (value: number) => void;
  onExtractConcurrencyChange: (value: number) => void;
  onEmbeddingConcurrencyChange: (value: number) => void;
  onMaxChunkCharsChange: (value: number) => void;
  onChunkOverlapCharsChange: (value: number) => void;
  documents: UploadedDocument[];
  task: DatabaseBuildProgress | null;
  error: string;
  loading: boolean;
  onUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onToggleDocument: (id: string) => void;
  onStartBuild: () => void;
}) {
  return (
    <section className="builder-layout">
      <div className="hero-panel">
        <div>
          <span className="eyebrow">Data Pipeline</span>
          <h1>自动构建知识图谱</h1>
          <p>上传非结构化业务文档，系统将自动执行解析、实体抽取与向量化，为您无缝构建结构化图谱网络。</p>
        </div>
        <div className="hero-actions">
          <label className="upload-button">
            <Upload size={16} />
            上传文件
            <input type="file" multiple onChange={onUpload} />
          </label>
          <button onClick={onStartBuild} disabled={loading}>
            {loading ? <RefreshCw size={16} className="spin" /> : <Sparkles size={16} />}
            {loading ? "正在启动..." : "开始构建"}
          </button>
        </div>
      </div>

      <div className="builder-top">
        <section className="panel builder-form">
          <div className="panel-head">
            <h2>知识库配置</h2>
            <span>{DEFAULT_MILVUS_DB}</span>
          </div>
          <label className="field">
            知识库 ID
            <input
              value={builderKbId}
              onChange={(event) => onBuilderKbIdChange(event.target.value)}
              placeholder="例如：kb_demo"
            />
          </label>
          <div className="builder-note">
            <span>当前会写入同一个 Milvus 数据库 `{DEFAULT_MILVUS_DB}`，通过 `kb_id` 做隔离展示与检索。</span>
          </div>
          <details className="advanced-settings">
            <summary>高级参数</summary>
            <div className="advanced-settings-grid">
              <label className="field compact-field">
                MinerU GPU
                <input
                  value={mineruGpus}
                  onChange={(event) => onMineruGpusChange(event.target.value)}
                  placeholder="例如：0,1"
                />
              </label>
              <label className="field compact-field">
                每卡 Worker 数
                <input
                  type="number"
                  min={1}
                  max={16}
                  value={workersPerGpu}
                  onChange={(event) => onWorkersPerGpuChange(Number(event.target.value) || 1)}
                />
              </label>
              <label className="field compact-field">
                LLM 抽取并发
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={extractConcurrency}
                  onChange={(event) => onExtractConcurrencyChange(Number(event.target.value) || 1)}
                />
              </label>
              <label className="field compact-field">
                Embedding 并发
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={embeddingConcurrency}
                  onChange={(event) => onEmbeddingConcurrencyChange(Number(event.target.value) || 1)}
                />
              </label>
              <label className="field compact-field">
                分块长度
                <input
                  type="number"
                  min={200}
                  max={12000}
                  value={maxChunkChars}
                  onChange={(event) => onMaxChunkCharsChange(Number(event.target.value) || 200)}
                />
              </label>
              <label className="field compact-field">
                分块重叠
                <input
                  type="number"
                  min={0}
                  max={4000}
                  value={chunkOverlapChars}
                  onChange={(event) => onChunkOverlapCharsChange(Number(event.target.value) || 0)}
                />
              </label>
            </div>
          </details>
          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="panel build-overview">
          <div className="panel-head">
            <h2>当前构建进度</h2>
            <span>{task?.status || "idle"}</span>
          </div>
          <div className="build-kpis">
            <MetricCard title="总进度" value={`${Math.round(task?.progress_percent || 0)}%`} compact />
            <MetricCard title="文件数" value={String(task?.files.length || 0)} compact />
            <MetricCard
              title="已完成"
              value={String(task?.files.filter((file) => file.stage === "completed").length || 0)}
              compact
            />
            <MetricCard
              title="失败"
              value={String(task?.files.filter((file) => file.stage === "failed").length || 0)}
              compact
            />
          </div>
          <div className="build-message">{task?.message || "选择文件后开始构建。"}</div>
        </section>
      </div>

      <section className="panel file-panel">
        <div className="panel-head">
          <h2>文件处理流水线</h2>
          <span>{documents.length} 个待选文件</span>
        </div>
        <div className="file-table">
          {documents.map((doc) => {
            const progress = task?.files.find((item) => item.path === doc.path);
            return (
              <div key={doc.id} className="file-row">
                <label className="check-cell">
                  <input
                    type="checkbox"
                    checked={doc.selected}
                    onChange={() => onToggleDocument(doc.id)}
                  />
                  <span />
                </label>
                <div className="file-meta">
                  <strong>{doc.name}</strong>
                  <span>
                    {doc.type} · {doc.size} · {doc.uploadedAt}
                  </span>
                </div>
                <div className="pipeline-cell">
                  <StageRail label="解析" percent={progress?.parse_progress || 0} tone="cyan" />
                  <StageRail label="向量化" percent={progress?.vector_progress || 0} tone="gold" />
                  <StageRail label="总进度" percent={progress?.progress || 0} tone="green" />
                </div>
                <div className="file-state">
                  <span className={`pill ${statusTone(progress?.stage || "queued")}`}>
                    {progress?.stage || "queued"}
                  </span>
                  <small>{progress?.state || "等待构建"}</small>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </section>
  );
}

function QueryModule({
  selectedKbId,
  catalog,
  trace,
  answer,
  loading,
  query,
  queryError,
  graphStage,
  onSelectKb,
  onQueryChange,
  onGraphStageChange,
  onSearch,
}: {
  selectedKbId: string;
  catalog: LibraryRecord[];
  trace: SearchTrace | null;
  answer: RagAnswerResponse | null;
  loading: boolean;
  query: string;
  queryError: string;
  graphStage: QueryGraphStage;
  onSelectKb: (kbId: string) => void;
  onQueryChange: (value: string) => void;
  onGraphStageChange: (value: QueryGraphStage) => void;
  onSearch: () => void;
}) {
  const [selectedRelationId, setSelectedRelationId] = React.useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = React.useState<string | null>(null);
  const [hoveredLinkId, setHoveredLinkId] = React.useState<string | null>(null);
  const [graphScale, setGraphScale] = React.useState(100);
  const [seedExploreDepth, setSeedExploreDepth] = React.useState<2 | 3 | 4>(3);
  const [seedShuffleNonce, setSeedShuffleNonce] = React.useState(0);
  const [seedStageGraph, setSeedStageGraph] = React.useState<EntityNeighborhoodResponse["graph"]>({
    nodes: [],
    links: [],
  });
  const [selectedSeedEntityId, setSelectedSeedEntityId] = React.useState<string | null>(null);
  const [seedStageCenterId, setSeedStageCenterId] = React.useState<string | null>(null);
  const [seedStageLoading, setSeedStageLoading] = React.useState(false);
  const answerSupport = React.useMemo(
    () => answer?.grounded_passages?.[0]?.passage?.trim() || trace?.trace.grounded_passages?.[0]?.passage?.trim() || "",
    [answer, trace],
  );
  const graphSource = React.useMemo(() => {
    if (!trace) return { nodes: [], links: [] };
    if (graphStage === "seed") {
      return seedStageGraph;
    }
    return getTraceGraphByStage(trace, graphStage);
  }, [graphStage, seedStageGraph, trace]);
  const graphData = React.useMemo(() => {
    if (!trace) {
      return {
        nodes: [],
        links: [],
        stats: { totalNodes: 0, totalLinks: 0, visibleNodes: 0, visibleLinks: 0 },
        nodeMap: new Map<string, PreparedGraphNode>(),
      };
    }
    return prepareGraphData(
      graphSource.nodes.map((node) => ({
        ...node,
        label: getReadableNodeLabel(node),
        color: node.is_seed ? "#ffd166" : "#64f0cd",
      })),
      graphSource.links.map((link) => ({
        ...link,
        color: link.id === selectedRelationId ? colorFromSeed(link.id) : "rgba(103, 204, 255, 0.32)",
      })),
      {
        selectedRelationId,
        densityMode: "balanced",
        preferredNodeLimit: graphStage === "seed" ? 36 : 20,
        preferredLinkLimit: graphStage === "seed" ? 48 : 22,
      },
    );
  }, [graphSource, graphStage, selectedRelationId, trace]);

  const topologyGraph = React.useMemo(
    () => buildTopologyGraph(graphData, selectedRelationId),
    [graphData, selectedRelationId],
  );
  const centerNodeId = React.useMemo(
    () =>
      (graphStage === "seed" ? seedStageCenterId : null) ||
      topologyGraph.nodes.find((node) => node.is_seed)?.id ||
      topologyGraph.nodes[0]?.id ||
      null,
    [graphStage, seedStageCenterId, topologyGraph.nodes],
  );

  const [graphRef, graphSize] = useElementSize<HTMLDivElement>();
  const graphCanvasRef = React.useRef<GraphCanvasHandle | null>(null);
  const evidenceRefs = React.useRef<Record<string, HTMLDivElement | null>>({});
  const highlightTerms = React.useMemo(() => {
    if (!trace) return [];
    const queryTerms = extractHighlightTerms(trace.search.query);
    const relationTerms = trace.trace.result_relations.flatMap((item) => [
      item.subject_name,
      item.object_name,
      item.relation,
    ]);
    return Array.from(new Set([...queryTerms, ...relationTerms]));
  }, [trace]);

  const evidenceMatches = React.useMemo(() => {
    if (!trace || !selectedRelationId) return new Set<string>();
    return new Set(
      trace.trace.grounded_passages
        .filter((item) => item.matched_relation_ids?.includes(selectedRelationId))
        .map((item) => item.id),
    );
  }, [graphSource, selectedRelationId, trace]);

  React.useEffect(() => {
    setSelectedRelationId(null);
    setHoveredNodeId(null);
    setHoveredLinkId(null);
  }, [graphStage, trace]);

  React.useEffect(() => {
    if (!trace) {
      setSelectedSeedEntityId(null);
      return;
    }
    const firstSeedId = trace.trace.seed_entities[0]?.id || null;
    setSelectedSeedEntityId((current) =>
      current && trace.trace.seed_entities.some((entity) => entity.id === current) ? current : firstSeedId,
    );
  }, [trace]);

  React.useEffect(() => {
    setSeedShuffleNonce(0);
  }, [seedExploreDepth, trace]);

  React.useEffect(() => {
    if (!trace || graphStage !== "seed") {
      setSeedStageLoading(false);
      return;
    }

    const seedEntityIds = trace.trace.seed_entities.map((entity) => entity.id).filter(Boolean);
    if (seedEntityIds.length === 0) {
      setSeedStageGraph({ nodes: [], links: [] });
      setSeedStageCenterId(null);
      setSeedStageLoading(false);
      return;
    }

    let cancelled = false;
    const loadSeedGraph = async () => {
      setSeedStageLoading(true);
      try {
        const response = await fetch(apiUrl("/api/search/entity-neighborhood"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: DEFAULT_USER_ID,
            kb_id: selectedKbId,
            seed_entity_ids: seedEntityIds,
            center_entity_id: selectedSeedEntityId || seedEntityIds[0],
            depth: seedExploreDepth,
            relation_limit: 10,
            shuffle_seed: seedShuffleNonce,
          }),
        });
        if (!response.ok) throw new Error("实体阶段关系图加载失败");
        const data = (await response.json()) as EntityNeighborhoodResponse;
        if (cancelled) return;
        setSeedStageGraph(data.graph || { nodes: [], links: [] });
        setSeedStageCenterId(data.center_entity_id || selectedSeedEntityId || seedEntityIds[0] || null);
      } catch {
        if (cancelled) return;
        setSeedStageGraph({ nodes: [], links: [] });
        setSeedStageCenterId(selectedSeedEntityId || seedEntityIds[0] || null);
      } finally {
        if (!cancelled) setSeedStageLoading(false);
      }
    };

    void loadSeedGraph();
    return () => {
      cancelled = true;
    };
  }, [graphStage, seedExploreDepth, seedShuffleNonce, selectedKbId, selectedSeedEntityId, trace]);

  const graphStageHint = React.useMemo(() => {
    if (!trace) return "";
    if (graphStage === "seed") {
      return "这里展示问题里识别出的实体，你可以切换实体并查看它周围的关联关系。";
    }
    return "这里展示系统最终保留下来的关键关系，右侧证据会对应这些结果。";
  }, [graphStage, trace]);

  React.useEffect(() => {
    if (evidenceMatches.size === 0) return;
    const firstMatch = Array.from(evidenceMatches)[0];
    evidenceRefs.current[firstMatch]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [evidenceMatches]);

  return (
    <section className="query-layout">
      <div className="hero-panel">
        <div>
          <span className="eyebrow">Retrieval & Reasoning</span>
          <h1>语义检索与图谱推理</h1>
          <p>通过自然语言发起查询，深度透视系统召回、图谱扩展的完整推理链路，精准溯源原文依据。</p>
        </div>
      </div>

      <div className="query-top">
        <section className="panel query-form">
          <div className="panel-head">
            <h2>问题输入</h2>
            <span>{selectedKbId || "未选择"}</span>
          </div>
          <label className="field">
            检索知识库
            <select value={selectedKbId} onChange={(event) => onSelectKb(event.target.value)}>
              {catalog.map((item) => (
                <option key={item.kb_id} value={item.kb_id}>
                  {item.kb_id}
                </option>
              ))}
            </select>
          </label>
          <div className="query-box">
            <textarea
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="例如：冰箱温度调节后多久可以稳定？"
            />
            <button onClick={onSearch} disabled={loading || !selectedKbId}>
              {loading ? <RefreshCw size={16} className="spin" /> : <Send size={16} />}
              开始推理
            </button>
          </div>
          {trace ? (
            <div className="query-answer-card">
              <div className="answer-card-head">
                <Bot size={18} />
                <strong>问题回答</strong>
              </div>
              <p>
                <HighlightText
                  text={
                    answer?.answer ||
                    "知识库中还没有生成正式回答，请先检查后端 /api/search/answer 是否可用。"
                  }
                  terms={highlightTerms}
                />
              </p>
              {answerSupport ? (
                <small className="answer-support">
                  <HighlightText text={answerSupport} terms={highlightTerms} />
                </small>
              ) : null}
            </div>
          ) : null}
          {queryError ? <div className="error-banner">{queryError}</div> : null}
        </section>

        <section className="panel reasoning-feed">
          <div className="panel-head">
            <h2>推理步骤</h2>
            <span>{trace?.trace.steps.length || 0} 步</span>
          </div>
          {trace ? (
            <div className="step-list">
              {trace.trace.steps.map((step, index) => (
                <div key={step.id} className="step-card">
                  <div className="step-index">{index + 1}</div>
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.summary}</p>
                    {step.highlights?.length ? (
                      <div className="tag-row">
                        {step.highlights.map((item) => (
                          <span key={item}>{item}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-panel">输入问题后，这里会展示完整的检索编排过程。</div>
          )}
        </section>
      </div>

      <div className="reasoning-layout">
        <section className="panel graph-panel">
          <div className="panel-head">
            <div className="graph-panel-title">
              <h2>实体拓扑结构画布</h2>
              <span className="graph-ai-pill">
                <Sparkles size={13} />
                AI 模型实时生成
              </span>
            </div>
            <div className="graph-panel-meta">
              <span>过滤后节点：{topologyGraph.stats.visibleNodes} / {topologyGraph.stats.totalNodes}</span>
              {graphStageHint ? <span>{graphStageHint}</span> : null}
            </div>
          </div>
          <div className="graph-legend-strip">
            <span><i className="legend-dot center" /> 黄色：当前中心实体</span>
            <span><i className="legend-dot seed" /> 绿色：命中实体</span>
            <span><i className="legend-dot related" /> 蓝色：关联实体</span>
          </div>
          {graphStage === "seed" && trace?.trace.seed_entities.length ? (
            <div className="graph-seed-entity-strip">
              {trace.trace.seed_entities.map((entity) => (
                <button
                  key={entity.id}
                  type="button"
                  className={selectedSeedEntityId === entity.id ? "active" : ""}
                  onClick={() => setSelectedSeedEntityId(entity.id)}
                >
                  {entity.name}
                </button>
              ))}
            </div>
          ) : null}
          <div className="reasoning-graph" ref={graphRef}>
            {trace && topologyGraph.nodes.length > 0 && graphSize.width > 0 ? (
                <TopologyCanvas
                  ref={graphCanvasRef}
                  width={graphSize.width}
                  height={Math.max(graphSize.height, 560)}
                  nodes={topologyGraph.nodes}
                  links={topologyGraph.links}
                  centerNodeId={centerNodeId}
                  dimUnfocused={graphStage !== "seed"}
                  selectedRelationId={selectedRelationId}
                  hoveredNodeId={hoveredNodeId}
                  hoveredLinkId={hoveredLinkId}
                  onHoveredNodeChange={setHoveredNodeId}
                  onHoveredLinkChange={setHoveredLinkId}
                  onSelectedRelationChange={setSelectedRelationId}
                  onScaleChange={setGraphScale}
                />
            ) : (
              <div className="empty-panel dark">
                <Network size={28} />
                {!trace
                  ? "等待检索结果"
                  : graphStage === "seed" && seedStageLoading
                    ? "实体阶段正在按数据库中的实体关系构图..."
                    : "本次检索暂未生成可展示的关系图谱"}
              </div>
            )}
            <div className="graph-control-dock">
              <div className="graph-hop-switch">
                {[
                  { id: "seed", label: "命中实体" },
                  { id: "result", label: "使用数据" },
                ].map((stage) => (
                  <button
                    key={stage.id}
                    type="button"
                    className={graphStage === stage.id ? "active" : ""}
                    onClick={() => onGraphStageChange(stage.id as QueryGraphStage)}
                  >
                    {stage.label}
                  </button>
                ))}
              </div>
              {graphStage === "seed" ? (
                <div className="graph-seed-controls">
                  <div className="graph-seed-depth">
                    {[2, 3, 4].map((depth) => (
                      <button
                        key={depth}
                        type="button"
                        className={seedExploreDepth === depth ? "active" : ""}
                        onClick={() => setSeedExploreDepth(depth as 2 | 3 | 4)}
                      >
                        {depth} 轮
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="graph-randomize"
                    onClick={() => setSeedShuffleNonce((current) => current + 1)}
                  >
                    <Sparkles size={14} />
                    换一组
                  </button>
                </div>
              ) : null}
              <div className="graph-control-status">
                <button
                  type="button"
                  className="control-main"
                  onClick={() => {
                    graphCanvasRef.current?.resetView();
                  }}
                >
                  <RotateCcw size={16} />
                  重力布局
                </button>
                <span>比例：{graphScale}%</span>
              </div>
            </div>
          </div>
        </section>

        <section className="panel answer-panel">
          <div className="panel-head">
            <h2>结果与证据</h2>
            <span>{trace?.search.results.length || 0} 条</span>
          </div>
          {trace ? (
            <div className="answer-panel-scroll">
              <section className="answer-section answer-section-primary">
                <div className="answer-section-head">
                  <div className="answer-section-title">
                    <Bot size={18} />
                    <strong>最终答案</strong>
                  </div>
                  <span className="answer-section-meta">{trace.search.kb_id}</span>
                </div>
                <div className="answer-summary">
                  <div>
                    <p>
                      {answer?.answer || "知识库中还没有生成正式回答，请先检查后端 /api/search/answer 是否可用。"}
                    </p>
                    <small className="answer-section-meta">
                      命中 {trace.search.results.length} 条关系，回查 {trace.search.grounded_passages.length} 段证据。
                    </small>
                    {answer?.retrieval_query ? (
                      <small className="answer-retrieval-query">
                        检索改写：<HighlightText text={answer.retrieval_query} terms={highlightTerms} />
                      </small>
                    ) : null}
                  </div>
                </div>
              </section>

              <section className="answer-section">
                <div className="answer-section-head">
                  <div className="answer-section-title">
                    <BrainCircuit size={18} />
                    <strong>关系摘要</strong>
                  </div>
                  <span className="answer-section-meta">用于解释命中的图谱关系</span>
                </div>
                <div className="relation-list">
                  {trace.trace.result_relations.map((item, index) => (
                    <button
                      key={item.id}
                      type="button"
                      className={`relation-card interactive-card ${
                        item.id === selectedRelationId ? "selected" : ""
                      }`}
                      style={
                        item.id === selectedRelationId
                          ? ({ "--relation-accent": colorFromSeed(item.id) } as React.CSSProperties)
                          : undefined
                      }
                      onClick={() => setSelectedRelationId(item.id)}
                    >
                      <div className="content-head">
                        <div className="relation-card-title">
                          <span className="relation-rank">#{index + 1}</span>
                          <strong>
                            <HighlightText
                              text={`${item.subject_name} → ${item.object_name}`}
                              terms={[item.subject_name, item.object_name]}
                            />
                          </strong>
                        </div>
                        <span className="relation-badge">
                          <HighlightText text={item.relation} terms={[item.relation]} />
                        </span>
                      </div>
                      <div className="relation-triple-line">
                        <HighlightText
                          text={`${item.subject_name} -- ${item.relation} --> ${item.object_name}`}
                          terms={[item.subject_name, item.object_name, item.relation]}
                        />
                      </div>
                      <div className="content-scroll">
                        <p>
                          <HighlightText
                            text={item.describe || item.passage}
                            terms={[item.subject_name, item.object_name, item.relation, item.describe || "", ...highlightTerms]}
                          />
                        </p>
                      </div>
                      <div className="relation-score-line">相关度 {item.score.toFixed(3)}</div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="answer-section">
                <div className="answer-section-head">
                  <div className="answer-section-title">
                    <Cpu size={18} />
                    <strong>证据原文</strong>
                  </div>
                  <span className="answer-section-meta">用于支撑最终回答</span>
                </div>
                <div className="evidence-list">
                  {trace.trace.grounded_passages.map((item, index) => (
                    <div
                      key={item.id}
                      ref={(node) => {
                        evidenceRefs.current[item.id] = node;
                      }}
                      className={`evidence-card ${
                        evidenceMatches.has(item.id) ? "matched" : selectedRelationId ? "dimmed" : ""
                      }`}
                    >
                      <div className="evidence-head">
                        <label>{item.docment_id || item.id}</label>
                        <span className="evidence-index">证据 {index + 1}</span>
                      </div>
                      {item.matched_relation_ids?.length ? (
                        <div className="tag-row evidence-tags">
                          {item.matched_relation_ids.map((relationId) => {
                            const relation = trace.trace.result_relations.find((entry) => entry.id === relationId);
                            const label = relation
                              ? `${relation.subject_name} → ${relation.relation} → ${relation.object_name}`
                              : relationId;
                            return (
                              <button
                                key={relationId}
                                type="button"
                                className={`evidence-tag ${relationId === selectedRelationId ? "active" : ""}`}
                                style={
                                  relationId === selectedRelationId
                                    ? ({ "--relation-accent": colorFromSeed(relationId) } as React.CSSProperties)
                                    : undefined
                                }
                                onClick={() => setSelectedRelationId(relationId)}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>
                      ) : null}
                      <div className="content-scroll evidence-scroll">
                        <p>
                          <HighlightText text={item.passage} terms={highlightTerms} />
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          ) : (
            <div className="empty-panel">还没有查询结果。</div>
          )}
        </section>
      </div>
    </section>
  );
}

function MetricCard({
  title,
  value,
  compact = false,
}: {
  title: string;
  value: string;
  compact?: boolean;
}) {
  return (
    <div className={`metric-card ${compact ? "compact" : ""}`}>
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StageRail({
  label,
  percent,
  tone,
}: {
  label: string;
  percent: number;
  tone: "cyan" | "gold" | "green";
}) {
  return (
    <div className="stage-rail">
      <div className="stage-head">
        <span>{label}</span>
        <span>{Math.round(percent)}%</span>
      </div>
      <div className={`stage-track ${tone}`}>
        <div style={{ width: `${Math.max(0, Math.min(percent, 100))}%` }} />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
