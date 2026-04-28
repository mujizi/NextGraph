import React from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  ArrowLeft,
  BookOpen,
  Check,
  Database,
  Edit3,
  Filter,
  FolderOpen,
  House,
  MessageCircle,
  MoreHorizontal,
  Plus,
  Search,
  Send,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import appLogo from "./assets/app-logo.png";
import "./styles.css";

type Page = "home" | "knowledge" | "chat" | "search";
type Status = "已解析" | "待解析" | "部分失败";

type KnowledgeBase = {
  id: string;
  name: string;
  date: string;
  owner: string;
  files: number;
  status: Status;
  tone: "green" | "blue" | "purple" | "amber";
};

type ChatItem = {
  id: string;
  title: string;
  knowledgeId: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolLabel?: string;
  materials?: ChatMaterial[];
  catalogGraph?: CatalogGraph;
};

type ChatMaterial = {
  id: string;
  sourceType: "local" | "global";
  knowledgeName: string;
  fileId: string;
  fileName: string;
  paragraphId: string;
  pageIdx: number;
  bbox: number[];
  text: string;
  imageUrl: string;
};

type CatalogGraph = {
  books: CatalogGraphBook[];
};

type CatalogGraphBook = {
  fileId: string;
  fileName: string;
  selectedCatalogIds: string[];
  catalogs: CatalogGraphNode[];
};

type CatalogGraphNode = {
  catalogId: string;
  title: string;
  path: string[];
  depth: number;
  selected: boolean;
};

type GraphNodeData = {
  label: string;
  eyebrow?: string;
  selected?: boolean;
  icon?: "book";
  tone?: "book" | "focus" | "context" | "leaf";
  size?: number;
};

type KnowledgeFile = {
  id: string;
  checked: boolean;
  type: string;
  name: string;
  note: string;
  uploadedAt: string;
  parseStatus: "未解析" | "排队中" | "解析中" | "失败" | "完成";
  graphStatus: "未抽取" | "排队中" | "抽取中" | "失败" | "成功";
  parseProgress: number;
  graphProgress: number;
  storedPath?: string;
};

type UploadedFileResponse = {
  name: string;
  relativePath: string;
  extension: string;
  size: number;
  storedPath: string;
};

type UploadResponse = {
  knowledgeName: string;
  storedDir: string;
  saved: UploadedFileResponse[];
  skipped: string[];
};

type DeleteResponse = {
  deleted: string[];
  missing: string[];
};

type ParseFileStatus = "queued" | "parsing" | "parsed" | "failed";

type ParseBatchFile = {
  storedPath: string;
  name: string;
  status: ParseFileStatus;
  progress: number;
  error?: string | null;
  outputPath?: string;
};

type ParseBatchResponse = {
  id: string;
  status: "queued" | "running" | "done" | "partial_failed";
  total: number;
  completed: number;
  failed: number;
  finished: number;
  running: number;
  queued: number;
  progress: number;
  files: ParseBatchFile[];
};

type GraphFileStatus =
  | "queued"
  | "validating"
  | "chunking"
  | "llm_extracting"
  | "catalog_merging"
  | "embedding"
  | "writing"
  | "done"
  | "failed"
  | "skipped";

type GraphBatchFile = {
  storedPath: string;
  name: string;
  status: GraphFileStatus;
  progress: number;
  message?: string;
  error?: string | null;
  chunksTotal?: number;
  chunksDone?: number;
  questionsIndexed?: number;
  outputPath?: string;
};

type GraphBatchResponse = {
  id: string;
  status: "queued" | "running" | "done" | "partial_failed";
  total: number;
  completed: number;
  failed: number;
  finished: number;
  running: number;
  queued: number;
  progress: number;
  files: GraphBatchFile[];
};

type SearchParagraph = {
  paragraphId: string;
  text: string;
  pageIdx?: number;
  bbox?: number[];
  order?: number;
};

type SearchCatalogNode = {
  catalogId?: string;
  title: string;
  summary: string;
  paragraphIds?: string[];
  children?: SearchCatalogNode[];
};

type LocalSearchResult = {
  book: {
    fileId: string;
    fileName: string;
  };
  question: {
    questionId?: string;
    question: string;
    answerHint: string;
    score: number;
    catalogIds?: string[];
  };
  paragraphs: SearchParagraph[];
};

type GlobalSearchBook = {
  fileId: string;
  fileName: string;
  graphDir: string;
  catalog: SearchCatalogNode[];
  selectedCatalogIds: string[];
  selectedCatalogs: {
    catalogId: string;
    title: string;
    summary: string;
    path: string[];
  }[];
  reason: string;
  paragraphs: SearchParagraph[];
};

type KnowledgeSearchResponse = {
  query: string;
  knowledgeName: string;
  mode: "local" | "global" | "hybrid";
  bookCount?: number;
  message?: string;
  local: {
    topK: number;
    strategy: string;
    results: LocalSearchResult[];
  };
  global: {
    topK: number;
    strategy: string;
    books: GlobalSearchBook[];
  };
};

const initialKnowledgeBases: KnowledgeBase[] = [
  {
    id: "kb-yingpu",
    name: "影谱项目库",
    date: "2026-04-27",
    owner: "Mengna",
    files: 3,
    status: "已解析",
    tone: "green",
  },
  {
    id: "kb-script",
    name: "电影剧本文档",
    date: "2026-04-25",
    owner: "Mengna",
    files: 0,
    status: "部分失败",
    tone: "blue",
  },
];

const initialChats: ChatItem[] = [
  { id: "chat-1", title: "一个真正有力量的主角", knowledgeId: "kb-yingpu" },
  { id: "chat-2", title: "为什么观众对电影角色共情", knowledgeId: "kb-yingpu" },
  { id: "chat-3", title: "第一章电影剧本写作基础", knowledgeId: "kb-script" },
];

const allowedExtensions = new Set(["pdf", "txt", "md", "xlsx", "docx"]);
const acceptedFileTypes = ".pdf,.txt,.md,.xlsx,.docx";

const initialFiles: KnowledgeFile[] = [
  {
    id: "file-1",
    checked: true,
    type: "PDF",
    name: "电影剧本写作基础与角色弧光.pdf",
    note: "等待 AI 图谱抽取",
    uploadedAt: "21/04/2026 09:59:20",
    parseStatus: "完成",
    graphStatus: "未抽取",
    parseProgress: 100,
    graphProgress: 0,
  },
  {
    id: "file-2",
    checked: false,
    type: "DOCX",
    name: "第一章电影剧本结构.docx",
    note: "文本解析中",
    uploadedAt: "21/04/2026 10:12:03",
    parseStatus: "未解析",
    graphStatus: "未抽取",
    parseProgress: 63,
    graphProgress: 0,
  },
  {
    id: "file-3",
    checked: false,
    type: "MD",
    name: "角色动机与冲突设计.md",
    note: "等待中",
    uploadedAt: "21/04/2026 10:40:11",
    parseStatus: "未解析",
    graphStatus: "未抽取",
    parseProgress: 0,
    graphProgress: 0,
  },
];

const initialKnowledgeFiles: Record<string, KnowledgeFile[]> = {
  "kb-yingpu": initialFiles,
  "kb-script": [],
};

const storageKeys = {
  knowledgeBases: "nextgraph:knowledge-bases",
  knowledgeFiles: "nextgraph:knowledge-files",
  chats: "nextgraph:chats",
  chatMessages: "nextgraph:chat-messages",
  uiState: "nextgraph:ui-state",
} as const;

type PersistedUiState = {
  page?: Page;
  selectedKnowledgeId?: string | null;
  selectedChatKnowledgeId?: string | null;
  selectedSearchKnowledgeId?: string | null;
  activeChatId?: string;
};

function isPage(value: unknown): value is Page {
  return value === "home" || value === "knowledge" || value === "chat" || value === "search";
}

function readStoredValue<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;

  try {
    const rawValue = window.localStorage.getItem(key);
    return rawValue ? (JSON.parse(rawValue) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeStoredValue<T>(key: string, value: T) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Keep the app usable if browser storage is unavailable or full.
  }
}

const chatHistoryRounds = Number.parseInt(
  ((import.meta as unknown as { env?: Record<string, string> }).env?.VITE_CHAT_HISTORY_ROUNDS ?? "10"),
  10,
);

function trimChatHistory(messages: ChatMessage[]) {
  return messages.slice(-Math.max(1, chatHistoryRounds) * 2);
}

function parseSseChunk(chunk: string) {
  const events: { event: string; data: string }[] = [];
  for (const block of chunk.split("\n\n")) {
    if (!block.trim()) continue;
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    events.push({ event, data: dataLines.join("\n") });
  }
  return events;
}

function getFileExtension(fileName: string) {
  return fileName.split(".").pop()?.toLowerCase() ?? "";
}

function getUploadRelativePath(file: File) {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

function formatUploadDate(date: Date) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatKnowledgeDate(date: Date) {
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function toParseStatusLabel(status: ParseFileStatus): KnowledgeFile["parseStatus"] {
  if (status === "parsed") return "完成";
  if (status === "failed") return "失败";
  if (status === "parsing") return "解析中";
  return "排队中";
}

function toParseNote(status: ParseFileStatus, error?: string | null) {
  if (status === "parsed") return "解析完成";
  if (status === "failed") return error ? `解析失败：${error.slice(0, 36)}` : "解析失败";
  if (status === "parsing") return "MinerU 解析中";
  return "等待 MinerU 解析";
}

function toGraphStatusLabel(status: GraphFileStatus): KnowledgeFile["graphStatus"] {
  if (status === "done") return "成功";
  if (status === "failed" || status === "skipped") return "失败";
  if (status === "queued") return "排队中";
  return "抽取中";
}

function toGraphNote(status: GraphFileStatus, message?: string, error?: string | null) {
  if (status === "done") return "抽取图完成";
  if (status === "failed" || status === "skipped") {
    return error ? `抽取失败：${error.slice(0, 36)}` : "抽取失败";
  }
  return message || "AI 抽取图处理中";
}

function App() {
  const [page, setPage] = React.useState<Page>(() => {
    const storedUiState = readStoredValue<PersistedUiState>(storageKeys.uiState, {});
    return isPage(storedUiState.page) ? storedUiState.page : "home";
  });
  const [knowledgeBases, setKnowledgeBases] = React.useState<KnowledgeBase[]>(
    () => readStoredValue(storageKeys.knowledgeBases, initialKnowledgeBases),
  );
  const [creatingKnowledge, setCreatingKnowledge] = React.useState(false);
  const [selectedKnowledgeId, setSelectedKnowledgeId] = React.useState<string | null>(() => {
    const storedUiState = readStoredValue<PersistedUiState>(storageKeys.uiState, {});
    return storedUiState.selectedKnowledgeId ?? null;
  });
  const [selectedChatKnowledgeId, setSelectedChatKnowledgeId] = React.useState<string | null>(() => {
    const storedUiState = readStoredValue<PersistedUiState>(storageKeys.uiState, {});
    return storedUiState.selectedChatKnowledgeId ?? null;
  });
  const [selectedSearchKnowledgeId, setSelectedSearchKnowledgeId] = React.useState<string | null>(
    () => {
      const storedUiState = readStoredValue<PersistedUiState>(storageKeys.uiState, {});
      return storedUiState.selectedSearchKnowledgeId ?? null;
    },
  );
  const [chats, setChats] = React.useState<ChatItem[]>(() =>
    readStoredValue(storageKeys.chats, initialChats),
  );
  const [chatMessages, setChatMessages] = React.useState<Record<string, ChatMessage[]>>(() =>
    readStoredValue(storageKeys.chatMessages, {}),
  );
  const [activeChatId, setActiveChatId] = React.useState(() => {
    const storedUiState = readStoredValue<PersistedUiState>(storageKeys.uiState, {});
    return storedUiState.activeChatId ?? initialChats[0].id;
  });
  const [knowledgeFiles, setKnowledgeFiles] =
    React.useState<Record<string, KnowledgeFile[]>>(() =>
      readStoredValue(storageKeys.knowledgeFiles, initialKnowledgeFiles),
    );

  React.useEffect(() => {
    writeStoredValue(storageKeys.knowledgeBases, knowledgeBases);
  }, [knowledgeBases]);

  React.useEffect(() => {
    writeStoredValue(storageKeys.knowledgeFiles, knowledgeFiles);
  }, [knowledgeFiles]);

  React.useEffect(() => {
    writeStoredValue(storageKeys.chats, chats);
  }, [chats]);

  React.useEffect(() => {
    writeStoredValue(storageKeys.chatMessages, chatMessages);
  }, [chatMessages]);

  React.useEffect(() => {
    writeStoredValue(storageKeys.uiState, {
      page,
      selectedKnowledgeId,
      selectedChatKnowledgeId,
      selectedSearchKnowledgeId,
      activeChatId,
    });
  }, [page, selectedKnowledgeId, selectedChatKnowledgeId, selectedSearchKnowledgeId, activeChatId]);

  const openPage = (nextPage: Page) => {
    if (nextPage === "knowledge") {
      setSelectedKnowledgeId(null);
    }
    if (nextPage === "chat") {
      setSelectedChatKnowledgeId(null);
    }
    if (nextPage === "search") {
      setSelectedSearchKnowledgeId(null);
    }
    setPage(nextPage);
  };

  const createKnowledgeBase = (name: string) => {
    const nextIndex = knowledgeBases.length + 1;
    const newKnowledge: KnowledgeBase = {
      id: `kb-${Date.now()}`,
      name: name.trim() || `新知识库 ${nextIndex}`,
      date: formatKnowledgeDate(new Date()),
      owner: "Mengna",
      files: 0,
      status: "待解析",
      tone: nextIndex % 2 === 0 ? "purple" : "amber",
    };

    setKnowledgeBases((current) => [...current, newKnowledge]);
    setKnowledgeFiles((current) => ({ ...current, [newKnowledge.id]: [] }));
  };

  const updateKnowledgeFiles = (knowledgeId: string, nextFiles: KnowledgeFile[]) => {
    setKnowledgeFiles((current) => ({ ...current, [knowledgeId]: nextFiles }));
    setKnowledgeBases((current) =>
      current.map((knowledge) =>
        knowledge.id === knowledgeId ? { ...knowledge, files: nextFiles.length } : knowledge,
      ),
    );
  };

  const renameKnowledgeBase = (id: string, name: string) => {
    setKnowledgeBases((current) =>
      current.map((knowledge) =>
        knowledge.id === id ? { ...knowledge, name: name.trim() || knowledge.name } : knowledge,
      ),
    );
  };

  const openKnowledgeBase = (id: string) => {
    setSelectedKnowledgeId(id);
    setPage("knowledge");
  };

  const openConversationKnowledge = (targetPage: "chat" | "search", knowledgeId: string) => {
    if (targetPage === "chat") {
      setSelectedChatKnowledgeId(knowledgeId);
      const firstChat = chats.find((item) => item.knowledgeId === knowledgeId) ?? chats[0];
      if (firstChat) setActiveChatId(firstChat.id);
    } else {
      setSelectedSearchKnowledgeId(knowledgeId);
    }
  };

  return (
    <div className="app">
      <TopBar current={page} onChange={openPage} />
      {page === "home" && (
        <HomePage
          knowledgeBases={knowledgeBases}
          onCreateKnowledge={() => setCreatingKnowledge(true)}
          onOpenKnowledge={openKnowledgeBase}
          onRenameKnowledge={renameKnowledgeBase}
        />
      )}
      {page === "knowledge" && (
        <KnowledgePage
          knowledgeBases={knowledgeBases}
          knowledgeFiles={knowledgeFiles}
          selectedKnowledgeId={selectedKnowledgeId}
          onCreateKnowledge={() => setCreatingKnowledge(true)}
          onOpenKnowledge={openKnowledgeBase}
          onRenameKnowledge={renameKnowledgeBase}
          onBackToList={() => setSelectedKnowledgeId(null)}
          onFilesChange={updateKnowledgeFiles}
        />
      )}
      {page === "chat" && (
        selectedChatKnowledgeId ? (
          <ConversationPage
            label="聊天"
            description="围绕知识库进行连续问答"
            items={chats}
            activeId={activeChatId}
            knowledgeBases={knowledgeBases}
            selectedKnowledgeId={selectedChatKnowledgeId}
            onActiveChange={setActiveChatId}
            onItemsChange={setChats}
            chatMessages={chatMessages}
            onChatMessagesChange={setChatMessages}
            onBackToKnowledgeList={() => setSelectedChatKnowledgeId(null)}
          />
        ) : (
          <ConversationKnowledgePicker
            title="全部聊天"
            knowledgeBases={knowledgeBases}
            onCreateKnowledge={() => setCreatingKnowledge(true)}
            onOpenKnowledge={(id) => openConversationKnowledge("chat", id)}
            onRenameKnowledge={renameKnowledgeBase}
          />
        )
      )}
      {page === "search" && (
        selectedSearchKnowledgeId ? (
          <SearchWorkspace
            knowledgeBases={knowledgeBases}
            selectedKnowledgeId={selectedSearchKnowledgeId}
            onBackToKnowledgeList={() => setSelectedSearchKnowledgeId(null)}
          />
        ) : (
          <ConversationKnowledgePicker
            title="全部搜索"
            knowledgeBases={knowledgeBases}
            onCreateKnowledge={() => setCreatingKnowledge(true)}
            onOpenKnowledge={(id) => openConversationKnowledge("search", id)}
            onRenameKnowledge={renameKnowledgeBase}
          />
        )
      )}
      {creatingKnowledge && (
        <CreateKnowledgeModal
          onCancel={() => setCreatingKnowledge(false)}
          onCreate={(name) => {
            createKnowledgeBase(name);
            setCreatingKnowledge(false);
          }}
        />
      )}
    </div>
  );
}

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand-lockup">
      <div className={compact ? "logo-mark compact" : "logo-mark"}>
        <img src={appLogo} alt="" aria-hidden="true" />
      </div>
    </div>
  );
}

function TopBar({
  current,
  onChange,
}: {
  current: Page;
  onChange: (page: Page) => void;
}) {
  const items: Array<[Page, string, React.ReactNode]> = [
    ["home", "首页", <House size={18} />],
    ["knowledge", "知识库", <BookOpen size={18} />],
    ["chat", "聊天", <MessageCircle size={18} />],
    ["search", "搜索", <Search size={18} />],
  ];

  return (
    <header className="topbar">
      <Logo compact />
      <nav className="pill-nav" aria-label="主导航">
        {items.map(([key, label, icon]) => (
          <button
            key={key}
            aria-label={label}
            title={label}
            className={current === key ? "active" : ""}
            onClick={() => onChange(key)}
          >
            {icon}
          </button>
        ))}
      </nav>
      <div className="topbar-actions">
        <button className="icon-button" aria-label="全局搜索">
          <Search size={17} />
        </button>
        <div className="avatar">M</div>
      </div>
    </header>
  );
}

function HomePage({
  knowledgeBases,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
}: {
  knowledgeBases: KnowledgeBase[];
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
}) {
  return (
    <main className="home page-shell">
      <section className="home-hero">
        <div className="hero-copy">
          <h1>
            欢迎来到 <span>影谱知识库</span>
          </h1>
          <p>管理知识库、解析文件、抽取图谱并开始智能问答</p>
        </div>
        <button className="primary-action" onClick={onCreateKnowledge}>
          <Plus size={18} />
          新建知识库
        </button>
      </section>

      <KnowledgeGrid
        title="知识库"
        knowledgeBases={knowledgeBases}
        onCreateKnowledge={onCreateKnowledge}
        onOpenKnowledge={onOpenKnowledge}
        onRenameKnowledge={onRenameKnowledge}
        showCreateAction={false}
      />
    </main>
  );
}

function KnowledgeGrid({
  title,
  knowledgeBases,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
  showCreateAction = true,
}: {
  title: string;
  knowledgeBases: KnowledgeBase[];
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
  showCreateAction?: boolean;
}) {
  return (
    <section className="home-section">
      <div className="section-head">
        <div className="section-title">
          <h2>{title}</h2>
          <FolderOpen size={24} />
        </div>
        {showCreateAction && (
          <button className="secondary-action" onClick={onCreateKnowledge}>
            <Plus size={17} />
            新建知识库
          </button>
        )}
      </div>
      <div className="card-grid">
        {knowledgeBases.map((knowledge) => (
          <KnowledgeCard
            key={knowledge.id}
            knowledge={knowledge}
            onOpen={() => onOpenKnowledge(knowledge.id)}
            onRename={(name) => onRenameKnowledge(knowledge.id, name)}
          />
        ))}
      </div>
    </section>
  );
}

function KnowledgeCard({
  knowledge,
  onOpen,
  onRename,
}: {
  knowledge: KnowledgeBase;
  onOpen: () => void;
  onRename: (name: string) => void;
}) {
  const [editing, setEditing] = React.useState(false);
  const [draft, setDraft] = React.useState(knowledge.name);

  React.useEffect(() => {
    setDraft(knowledge.name);
  }, [knowledge.name]);

  const save = () => {
    onRename(draft);
    setEditing(false);
  };

  return (
    <article className="project-card knowledge-card">
      <button className="card-open-area" onClick={onOpen} aria-label={`打开 ${knowledge.name}`}>
        <div className={`letter-icon ${knowledge.tone}`}>{knowledge.name[0] || "知"}</div>
        <MoreHorizontal size={18} className="card-more" />
        <h3>{knowledge.name}</h3>
        <p>
          {knowledge.date} · {knowledge.owner} · {knowledge.files} files
        </p>
        <span className={`status ${knowledge.status === "已解析" ? "success" : "warn"}`}>
          {knowledge.status}
        </span>
      </button>

      {editing ? (
        <div className="rename-row">
          <input
            value={draft}
            autoFocus
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") save();
              if (event.key === "Escape") setEditing(false);
            }}
          />
          <button aria-label="保存名称" onClick={save}>
            <Check size={15} />
          </button>
          <button aria-label="取消重命名" onClick={() => setEditing(false)}>
            <X size={15} />
          </button>
        </div>
      ) : (
        <button className="rename-button" onClick={() => setEditing(true)}>
          <Edit3 size={14} />
          重命名
        </button>
      )}
    </article>
  );
}

function CreateKnowledgeModal({
  onCancel,
  onCreate,
}: {
  onCancel: () => void;
  onCreate: (name: string) => void;
}) {
  const [name, setName] = React.useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    onCreate(name);
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <form className="modal-card" onSubmit={submit} role="dialog" aria-modal="true">
        <div className="modal-head">
          <h2>新建知识库</h2>
          <button type="button" aria-label="关闭" onClick={onCancel}>
            <X size={17} />
          </button>
        </div>
        <label className="modal-field">
          <span>知识库名称</span>
          <input
            value={name}
            autoFocus
            placeholder="请输入知识库名称"
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <div className="modal-actions">
          <button type="button" className="outline-button" onClick={onCancel}>
            取消
          </button>
          <button type="submit" className="primary-small">
            创建
          </button>
        </div>
      </form>
    </div>
  );
}

function KnowledgePage({
  knowledgeBases,
  knowledgeFiles,
  selectedKnowledgeId,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
  onBackToList,
  onFilesChange,
}: {
  knowledgeBases: KnowledgeBase[];
  knowledgeFiles: Record<string, KnowledgeFile[]>;
  selectedKnowledgeId: string | null;
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
  onBackToList: () => void;
  onFilesChange: (knowledgeId: string, files: KnowledgeFile[]) => void;
}) {
  const selectedKnowledge = knowledgeBases.find((knowledge) => knowledge.id === selectedKnowledgeId);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const [uploadMenuOpen, setUploadMenuOpen] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const [uploadMessage, setUploadMessage] = React.useState("");
  const [activeParseBatchId, setActiveParseBatchId] = React.useState<string | null>(null);
  const [parseBatch, setParseBatch] = React.useState<ParseBatchResponse | null>(null);
  const [activeGraphBatchId, setActiveGraphBatchId] = React.useState<string | null>(null);
  const [graphBatch, setGraphBatch] = React.useState<GraphBatchResponse | null>(null);

  React.useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  });

  React.useEffect(() => {
    if (!activeParseBatchId || !selectedKnowledge) return;

    let cancelled = false;
    let timer: number | undefined;

    const applyBatch = (batch: ParseBatchResponse) => {
      setParseBatch(batch);
      const nextFiles = knowledgeFiles[selectedKnowledge.id] ?? [];
      const stateByPath = new Map(batch.files.map((file) => [file.storedPath, file]));

      onFilesChange(
        selectedKnowledge.id,
        nextFiles.map((file) => {
          if (!file.storedPath) return file;
          const parseState = stateByPath.get(file.storedPath);
          if (!parseState) return file;

          return {
            ...file,
            note: toParseNote(parseState.status, parseState.error),
            parseStatus: toParseStatusLabel(parseState.status),
            parseProgress: parseState.status === "parsed" ? 100 : 0,
          };
        }),
      );
    };

    const pollBatch = async () => {
      try {
        const response = await fetch(`/api/knowledge/parse/${activeParseBatchId}`);
        if (!response.ok) throw new Error("parse status failed");
        const batch = (await response.json()) as ParseBatchResponse;
        if (cancelled) return;

        applyBatch(batch);

        if (batch.status === "done" || batch.status === "partial_failed") {
          setActiveParseBatchId(null);
          setUploadMessage(
            `解析完成：成功 ${batch.completed} 个，失败 ${batch.failed} 个。`,
          );
          return;
        }
      } catch {
        if (!cancelled) {
          setUploadMessage("解析状态刷新失败，请确认后端服务已启动。");
        }
      }

      if (!cancelled) {
        timer = window.setTimeout(pollBatch, 1500);
      }
    };

    pollBatch();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [activeParseBatchId, selectedKnowledgeId]);

  React.useEffect(() => {
    if (!activeGraphBatchId || !selectedKnowledge) return;

    let cancelled = false;
    let timer: number | undefined;

    const applyBatch = (batch: GraphBatchResponse) => {
      setGraphBatch(batch);
      const nextFiles = knowledgeFiles[selectedKnowledge.id] ?? [];
      const stateByPath = new Map(batch.files.map((file) => [file.storedPath, file]));

      onFilesChange(
        selectedKnowledge.id,
        nextFiles.map((file) => {
          if (!file.storedPath) return file;
          const graphState = stateByPath.get(file.storedPath);
          if (!graphState) return file;

          return {
            ...file,
            note: toGraphNote(graphState.status, graphState.message, graphState.error),
            graphStatus: toGraphStatusLabel(graphState.status),
            graphProgress: graphState.progress,
          };
        }),
      );
    };

    const pollBatch = async () => {
      try {
        const response = await fetch(`/api/knowledge/graph-extract/${activeGraphBatchId}`);
        if (!response.ok) throw new Error("graph status failed");
        const batch = (await response.json()) as GraphBatchResponse;
        if (cancelled) return;

        applyBatch(batch);

        if (batch.status === "done" || batch.status === "partial_failed") {
          setActiveGraphBatchId(null);
          setUploadMessage(`抽取图完成：成功 ${batch.completed} 个，失败 ${batch.failed} 个。`);
          return;
        }
      } catch {
        if (!cancelled) {
          setUploadMessage("抽取图状态刷新失败，请确认后端服务已启动。");
        }
      }

      if (!cancelled) {
        timer = window.setTimeout(pollBatch, 1500);
      }
    };

    pollBatch();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [activeGraphBatchId, selectedKnowledgeId]);

  if (!selectedKnowledge) {
    return (
      <main className="home page-shell">
        <KnowledgeGrid
          title="全部知识库"
          knowledgeBases={knowledgeBases}
          onCreateKnowledge={onCreateKnowledge}
          onOpenKnowledge={onOpenKnowledge}
          onRenameKnowledge={onRenameKnowledge}
        />
      </main>
    );
  }

  const tableFiles = knowledgeFiles[selectedKnowledge.id] ?? [];
  const selectedCount = tableFiles.filter((file) => file.checked).length;
  const parseRunning = activeParseBatchId !== null;
  const graphRunning = activeGraphBatchId !== null;
  const graphReadySelectedCount = tableFiles.filter(
    (file) => file.checked && file.parseStatus === "完成" && Boolean(file.storedPath),
  ).length;
  const parseSummary = parseBatch
    ? {
        total: parseBatch.total,
        completed: parseBatch.completed,
        failed: parseBatch.failed,
        running: parseBatch.running,
        queued: parseBatch.queued,
        progress: parseBatch.progress,
      }
    : {
        total: selectedCount,
        completed: 0,
        failed: 0,
        running: 0,
        queued: 0,
        progress: 0,
      };
  const graphSummary = graphBatch
    ? {
        total: graphBatch.total,
        completed: graphBatch.completed,
        failed: graphBatch.failed,
        running: graphBatch.running,
        queued: graphBatch.queued,
        progress: graphBatch.progress,
      }
    : {
        total: graphReadySelectedCount,
        completed: 0,
        failed: 0,
        running: 0,
        queued: 0,
        progress: 0,
      };

  const updateFiles = (nextFiles: KnowledgeFile[]) => {
    onFilesChange(selectedKnowledge.id, nextFiles);
  };

  const toggleAllFiles = () => {
    const shouldCheck = selectedCount !== tableFiles.length;
    updateFiles(tableFiles.map((file) => ({ ...file, checked: shouldCheck })));
  };

  const toggleFile = (id: string) => {
    updateFiles(
      tableFiles.map((file) => (file.id === id ? { ...file, checked: !file.checked } : file)),
    );
  };

  const uploadSelectedFiles = async (fileList: FileList | null) => {
    const selectedFiles = Array.from(fileList ?? []);
    if (!selectedFiles.length) return;

    const supportedFiles = selectedFiles.filter((file) =>
      allowedExtensions.has(getFileExtension(getUploadRelativePath(file))),
    );

    if (!supportedFiles.length) {
      setUploadMessage("仅支持 PDF、TXT、MD、XLSX、DOCX 文件。");
      return;
    }

    const formData = new FormData();
    formData.append("knowledge_name", selectedKnowledge.name);
    supportedFiles.forEach((file) => {
      formData.append("files", file);
      formData.append("relative_paths", getUploadRelativePath(file));
    });

    setUploading(true);
    setUploadMessage("");

    try {
      const response = await fetch("/api/knowledge/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("upload failed");
      }

      const result = (await response.json()) as UploadResponse;
      const uploadedAt = formatUploadDate(new Date());
      const uploadedFiles = result.saved.map((file) => ({
        id: `${file.storedPath}-${Date.now()}`,
        checked: false,
        type: file.extension,
        name: file.relativePath,
        note: "等待解析",
        uploadedAt,
        parseStatus: "未解析" as const,
        graphStatus: "未抽取" as const,
        parseProgress: 0,
        graphProgress: 0,
        storedPath: file.storedPath,
      }));

      updateFiles([...uploadedFiles, ...tableFiles]);
      setUploadMessage(
        `已上传 ${result.saved.length} 个文件到 ${result.storedDir}${
          result.skipped.length ? `，跳过 ${result.skipped.length} 个不支持文件` : ""
        }。`,
      );
    } catch {
      setUploadMessage("上传失败，请确认后端服务已启动。");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (folderInputRef.current) folderInputRef.current.value = "";
    }
  };

  const parseSelectedFiles = async () => {
    if (!selectedCount || parseRunning) return;

    const selectedFiles = tableFiles.filter(
      (file): file is KnowledgeFile & { storedPath: string } =>
        file.checked && Boolean(file.storedPath),
    );
    if (!selectedFiles.length) {
      setUploadMessage("请选择已上传到后端的文件。");
      return;
    }

    updateFiles(
      tableFiles.map((file) =>
        file.checked
          ? {
              ...file,
              note: "等待 MinerU 解析",
              parseStatus: "排队中",
              parseProgress: 0,
            }
          : file,
      ),
    );
    setUploadMessage("");

    try {
      const response = await fetch("/api/knowledge/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          knowledgeName: selectedKnowledge.name,
          files: selectedFiles.map((file) => ({
            storedPath: file.storedPath,
            name: file.name,
          })),
        }),
      });

      if (!response.ok) {
        throw new Error("parse failed");
      }

      const batch = (await response.json()) as ParseBatchResponse;
      setParseBatch(batch);
      setActiveParseBatchId(batch.id);
      setUploadMessage(`已提交 ${batch.total} 个文件到 MinerU 解析队列。`);
    } catch {
      updateFiles(
        tableFiles.map((file) =>
          file.checked
            ? {
                ...file,
                note: "解析提交失败",
                parseStatus: "失败",
                parseProgress: 0,
              }
            : file,
        ),
      );
      setUploadMessage("解析提交失败，请确认后端和 MinerU Router 已启动。");
    }
  };

  const extractSelectedGraphs = async () => {
    if (!graphReadySelectedCount || graphRunning) return;

    const selectedFiles = tableFiles.filter(
      (file): file is KnowledgeFile & { storedPath: string } =>
        file.checked && file.parseStatus === "完成" && Boolean(file.storedPath),
    );

    updateFiles(
      tableFiles.map((file) =>
        file.checked && file.parseStatus === "完成"
          ? {
              ...file,
              note: "等待 AI 抽取图",
              graphStatus: "排队中",
              graphProgress: 0,
            }
          : file,
      ),
    );
    setUploadMessage("");

    try {
      const response = await fetch("/api/knowledge/graph-extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          knowledgeName: selectedKnowledge.name,
          files: selectedFiles.map((file) => ({
            storedPath: file.storedPath,
            name: file.name,
          })),
          options: {
            llmConcurrency: 50,
            embeddingConcurrency: 50,
            maxChunkTokens: 3000,
            llmMaxRetries: 3,
            catalogMergeMode: "rule",
            forceRebuild: false,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("graph extract failed");
      }

      const batch = (await response.json()) as GraphBatchResponse;
      setGraphBatch(batch);
      setActiveGraphBatchId(batch.id);
      setUploadMessage(`已提交 ${batch.total} 个文件到 AI 抽取图队列。`);
    } catch {
      updateFiles(
        tableFiles.map((file) =>
          file.checked && file.parseStatus === "完成"
            ? {
                ...file,
                note: "抽取图提交失败",
                graphStatus: "失败",
                graphProgress: 0,
              }
            : file,
        ),
      );
      setUploadMessage("抽取图提交失败，请确认后端服务已启动。");
    }
  };

  const deleteSelectedFiles = async () => {
    const selectedFiles = tableFiles.filter((file) => file.checked);
    if (!selectedFiles.length) return;

    const storedPaths = selectedFiles.flatMap((file) => (file.storedPath ? [file.storedPath] : []));

    try {
      if (storedPaths.length) {
        const response = await fetch("/api/knowledge/files", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ storedPaths }),
        });

        if (!response.ok) {
          throw new Error("delete failed");
        }

        const result = (await response.json()) as DeleteResponse;
        setUploadMessage(
          `已删除 ${selectedFiles.length} 个文件${
            result.deleted.length ? `，目录文件 ${result.deleted.length} 个已移除` : ""
          }。`,
        );
      } else {
        setUploadMessage(`已删除 ${selectedFiles.length} 个文件。`);
      }

      updateFiles(tableFiles.filter((file) => !file.checked));
    } catch {
      setUploadMessage("删除失败，请确认后端服务已启动。");
    }
  };

  return (
    <main className="workbench-page">
      <section className="content-panel">
        <div className="knowledge-header">
          <div>
            <button className="back-link" onClick={onBackToList}>
              <ArrowLeft size={16} />
              全部知识库
            </button>
            <div className="title-row">
              <h1>文件列表</h1>
              <span className="tag">{selectedKnowledge.name}</span>
            </div>
            <p>解析成功后才能问答哦。</p>
          </div>
          <div className="toolbar">
            <button className="outline-button">
              <Filter size={16} />
              筛选
            </button>
            <label className="toolbar-search">
              <Search size={16} />
              <input placeholder="搜索" />
            </label>
            <div className="upload-control">
              <button
                className="square-primary"
                aria-label="上传文件或文件夹"
                aria-expanded={uploadMenuOpen}
                onClick={() => setUploadMenuOpen((open) => !open)}
                disabled={uploading}
              >
                <Upload size={17} />
              </button>
              {uploadMenuOpen && (
                <div className="upload-menu">
                  <button
                    type="button"
                    onClick={() => {
                      setUploadMenuOpen(false);
                      fileInputRef.current?.click();
                    }}
                  >
                    上传文件
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setUploadMenuOpen(false);
                      folderInputRef.current?.click();
                    }}
                  >
                    上传文件夹
                  </button>
                </div>
              )}
              <input
                ref={fileInputRef}
                className="visually-hidden"
                type="file"
                multiple
                accept={acceptedFileTypes}
                onChange={(event) => uploadSelectedFiles(event.target.files)}
              />
              <input
                ref={folderInputRef}
                className="visually-hidden"
                type="file"
                multiple
                accept={acceptedFileTypes}
                onChange={(event) => uploadSelectedFiles(event.target.files)}
              />
            </div>
          </div>
        </div>

        <div className="batch-bar">
          <strong>已选择 {selectedCount} 个文件</strong>
          <button onClick={parseSelectedFiles} disabled={!selectedCount || parseRunning}>
            {parseRunning ? "解析中" : "解析文件"}
          </button>
          <div className="batch-parse-progress" aria-label="批量解析完成率">
            <span>
              总数 {parseSummary.total} · 完成 {parseSummary.completed} · 失败{" "}
              {parseSummary.failed} · 解析中 {parseSummary.running} · 排队{" "}
              {parseSummary.queued}
            </span>
            <span className="progress-track">
              <span style={{ width: `${parseSummary.progress}%` }} />
            </span>
            <b>{parseSummary.progress}%</b>
          </div>
          <div className="batch-parse-progress" aria-label="批量抽取图完成率">
            <span>
              抽取 {graphSummary.total} · 完成 {graphSummary.completed} · 失败{" "}
              {graphSummary.failed} · 抽取中 {graphSummary.running} · 排队{" "}
              {graphSummary.queued}
            </span>
            <span className="progress-track">
              <span style={{ width: `${graphSummary.progress}%` }} />
            </span>
            <b>{graphSummary.progress}%</b>
          </div>
          <button
            className="active"
            onClick={extractSelectedGraphs}
            disabled={!graphReadySelectedCount || graphRunning}
            title={graphReadySelectedCount ? "开始 AI 抽取图" : "请先选择解析完成的文件"}
          >
            {graphRunning ? "抽取中" : "AI 抽取图"}
          </button>
          <button className="danger" onClick={deleteSelectedFiles}>
            删除
          </button>
        </div>

        {uploadMessage && <div className="upload-status">{uploadMessage}</div>}

        <FileTable files={tableFiles} onToggle={toggleFile} onToggleAll={toggleAllFiles} />
      </section>
    </main>
  );
}

function ConversationKnowledgePicker({
  title,
  knowledgeBases,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
}: {
  title: string;
  knowledgeBases: KnowledgeBase[];
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
}) {
  return (
    <main className="home page-shell">
      <KnowledgeGrid
        title={title}
        knowledgeBases={knowledgeBases}
        onCreateKnowledge={onCreateKnowledge}
        onOpenKnowledge={onOpenKnowledge}
        onRenameKnowledge={onRenameKnowledge}
      />
    </main>
  );
}

function SearchWorkspace({
  knowledgeBases,
  selectedKnowledgeId,
  onBackToKnowledgeList,
}: {
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeId: string;
  onBackToKnowledgeList: () => void;
}) {
  const selectedKnowledge =
    knowledgeBases.find((knowledge) => knowledge.id === selectedKnowledgeId) ?? knowledgeBases[0];
  const [query, setQuery] = React.useState("");
  const [localTopK, setLocalTopK] = React.useState(5);
  const [globalTopK, setGlobalTopK] = React.useState(5);
  const [searching, setSearching] = React.useState(false);
  const [searchError, setSearchError] = React.useState("");
  const [searchResult, setSearchResult] = React.useState<KnowledgeSearchResponse | null>(null);
  const [activeGlobalBookId, setActiveGlobalBookId] = React.useState<string | null>(null);

  const activeGlobalBook =
    searchResult?.global.books.find((book) => book.fileId === activeGlobalBookId) ??
    searchResult?.global.books[0] ??
    null;

  React.useEffect(() => {
    if (!searchResult?.global.books.length) {
      setActiveGlobalBookId(null);
      return;
    }
    setActiveGlobalBookId((current) =>
      current && searchResult.global.books.some((book) => book.fileId === current)
        ? current
        : searchResult.global.books[0].fileId,
    );
  }, [searchResult]);

  const submitSearch = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || !selectedKnowledge || searching) return;

    setSearching(true);
    setSearchError("");

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmedQuery,
          knowledgeName: selectedKnowledge.name,
          mode: "hybrid",
          localTopK,
          globalTopK,
        }),
      });

      if (!response.ok) {
        throw new Error("search failed");
      }

      setSearchResult((await response.json()) as KnowledgeSearchResponse);
    } catch {
      setSearchError("搜索失败，请确认后端服务已启动，且该知识库已完成 AI 抽取图。");
    } finally {
      setSearching(false);
    }
  };

  return (
    <main className="search-page">
      <div className="search-workspace">
        <button className="back-link" onClick={onBackToKnowledgeList}>
          <ArrowLeft size={16} />
          全部搜索
        </button>
        <div className="search-title">
          <span className={`letter-icon small ${selectedKnowledge?.tone ?? "green"}`}>
            {selectedKnowledge?.name[0] ?? "知"}
          </span>
          <div>
            <h1>搜索</h1>
            <p>{selectedKnowledge?.name ?? "暂无知识库"}</p>
          </div>
        </div>
        <form
          className="search-form"
          onSubmit={(event) => {
            event.preventDefault();
            submitSearch();
          }}
        >
          <label className="big-search">
            <Search size={22} />
            <input
              value={query}
              placeholder="搜索知识库内容..."
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="search-options">
            <label>
              Local TopK
              <input
                type="number"
                min={1}
                max={20}
                value={localTopK}
                onChange={(event) => setLocalTopK(Number(event.target.value) || 5)}
              />
            </label>
            <label>
              Global TopK
              <input
                type="number"
                min={1}
                max={20}
                value={globalTopK}
                onChange={(event) => setGlobalTopK(Number(event.target.value) || 5)}
              />
            </label>
            <button className="secondary-action" disabled={!query.trim() || searching}>
              {searching ? "检索中" : "搜索"}
            </button>
          </div>
        </form>

        {searchError && <div className="search-error">{searchError}</div>}
        {searchResult?.message && <div className="search-empty">{searchResult.message}</div>}

        {searchResult && !searchResult.message && (
          <div className="search-results">
            <section className="result-section">
              <div className="result-section-head">
                <div>
                  <h2>Local</h2>
                  <p>
                    相似问题 {searchResult.local.results.length}/{searchResult.local.topK} ·{" "}
                    {searchResult.local.strategy}
                  </p>
                </div>
              </div>
              <div className="local-result-list">
                {searchResult.local.results.length ? (
                  searchResult.local.results.map((item, index) => (
                    <article className="local-result" key={`${item.book.fileId}-${item.question.questionId ?? index}`}>
                      <div className="result-meta">
                        <span>{item.book.fileName}</span>
                        <b>{item.question.score.toFixed(3)}</b>
                      </div>
                      <h3>{item.question.question}</h3>
                      <p>{item.question.answerHint}</p>
                      <ParagraphList paragraphs={item.paragraphs} />
                    </article>
                  ))
                ) : (
                  <div className="search-empty">暂无相似问题。</div>
                )}
              </div>
            </section>

            <section className="result-section">
              <div className="result-section-head">
                <div>
                  <h2>Global</h2>
                  <p>
                    相关书籍 {searchResult.global.books.length} · {searchResult.global.strategy}
                  </p>
                </div>
                {searchResult.global.books.length > 0 && (
                  <select
                    value={activeGlobalBook?.fileId ?? ""}
                    onChange={(event) => setActiveGlobalBookId(event.target.value)}
                  >
                    {searchResult.global.books.map((book) => (
                      <option value={book.fileId} key={book.fileId}>
                        {book.fileName}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {activeGlobalBook ? (
                <div className="global-result-grid">
                  <div className="catalog-panel">
                    <div className="book-name">{activeGlobalBook.fileName}</div>
                    <CatalogTree
                      nodes={activeGlobalBook.catalog}
                      selectedIds={new Set(activeGlobalBook.selectedCatalogIds)}
                    />
                  </div>
                  <div className="catalog-detail">
                    <h3>选中子目录</h3>
                    {activeGlobalBook.selectedCatalogs.map((node) => (
                      <div className="selected-catalog" key={node.catalogId}>
                        <strong>{node.path.join(" / ")}</strong>
                        <p>{node.summary}</p>
                      </div>
                    ))}
                    <ParagraphList paragraphs={activeGlobalBook.paragraphs} />
                  </div>
                </div>
              ) : (
                <div className="search-empty">暂无相关目录。</div>
              )}
            </section>
          </div>
        )}
      </div>
    </main>
  );
}

function ParagraphList({ paragraphs }: { paragraphs: SearchParagraph[] }) {
  if (!paragraphs.length) return null;
  return (
    <div className="paragraph-list">
      {paragraphs.map((paragraph) => (
        <blockquote key={paragraph.paragraphId}>
          <span>
            {paragraph.paragraphId}
            {typeof paragraph.pageIdx === "number" ? ` · p.${paragraph.pageIdx + 1}` : ""}
          </span>
          {paragraph.text}
        </blockquote>
      ))}
    </div>
  );
}

function CatalogTree({
  nodes,
  selectedIds,
}: {
  nodes: SearchCatalogNode[];
  selectedIds: Set<string>;
}) {
  if (!nodes.length) return <div className="search-empty">暂无目录。</div>;
  return (
    <ul className="catalog-tree">
      {nodes.map((node, index) => {
        const selected = Boolean(node.catalogId && selectedIds.has(node.catalogId));
        return (
          <li className={selected ? "selected" : ""} key={node.catalogId ?? `${node.title}-${index}`}>
            <div>
              <span>{node.title}</span>
              {node.summary && <small>{node.summary}</small>}
            </div>
            {node.children && node.children.length > 0 && (
              <CatalogTree nodes={node.children} selectedIds={selectedIds} />
            )}
          </li>
        );
      })}
    </ul>
  );
}

function FileTable({
  files,
  onToggle,
  onToggleAll,
}: {
  files: KnowledgeFile[];
  onToggle: (id: string) => void;
  onToggleAll: () => void;
}) {
  const allChecked = files.length > 0 && files.every((file) => file.checked);

  return (
    <div className="file-table">
      <div className="table-row table-head">
        <div className="cell check">
          <input
            type="checkbox"
            checked={allChecked}
            aria-label="选择全部文件"
            onChange={onToggleAll}
          />
        </div>
        <div className="cell name">名称 ↕</div>
        <div className="cell date">上传日期 ↕</div>
        <div className="cell parse-status">是否解析</div>
        <div className="cell graph-status">是否抽取图</div>
        <div className="cell parse-progress">解析进度</div>
        <div className="cell graph-progress">抽取图进度</div>
      </div>
      {!files.length && (
        <div className="empty-file-row">上传文件或文件夹开始构建知识库</div>
      )}
      {files.map((file) => (
        <div className="table-row" key={file.id}>
          <div className="cell check">
            <input
              type="checkbox"
              checked={file.checked}
              aria-label={`选择 ${file.name}`}
              onChange={() => onToggle(file.id)}
            />
          </div>
          <div className="cell name">
            <span className={`file-type ${file.type.toLowerCase()}`}>{file.type}</span>
            <span className="file-name">
              <strong>{file.name}</strong>
              <small>{file.note}</small>
            </span>
          </div>
          <div className="cell date">{file.uploadedAt}</div>
          <div className="cell parse-status">
            <span className={`state-pill ${file.parseStatus === "完成" ? "success" : file.parseStatus === "失败" ? "failed" : ""}`}>
              {file.parseStatus}
            </span>
          </div>
          <div className="cell graph-status">
            <span className={`state-pill ${file.graphStatus === "成功" ? "success" : file.graphStatus === "失败" ? "failed" : ""}`}>
              {file.graphStatus}
            </span>
          </div>
          <div className="cell parse-progress">
            <span className="progress-track">
              <span style={{ width: `${file.parseProgress}%` }} />
            </span>
            <span className={file.parseProgress === 100 ? "done" : "processing"}>
              {file.parseProgress}%
            </span>
          </div>
          <div className="cell graph-progress">
            <span className="progress-track">
              <span style={{ width: `${file.graphProgress}%` }} />
            </span>
            <span className={file.graphProgress === 100 ? "done" : "processing"}>
              {file.graphProgress}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ConversationPage({
  label,
  description,
  items,
  activeId,
  knowledgeBases,
  selectedKnowledgeId,
  onActiveChange,
  onItemsChange,
  chatMessages,
  onChatMessagesChange,
  onBackToKnowledgeList,
}: {
  label: string;
  description: string;
  items: ChatItem[];
  activeId: string;
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeId: string;
  onActiveChange: (id: string) => void;
  onItemsChange: React.Dispatch<React.SetStateAction<ChatItem[]>>;
  chatMessages: Record<string, ChatMessage[]>;
  onChatMessagesChange: React.Dispatch<React.SetStateAction<Record<string, ChatMessage[]>>>;
  onBackToKnowledgeList: () => void;
}) {
  const [sending, setSending] = React.useState(false);
  const [streamError, setStreamError] = React.useState("");
  const selectedKnowledge =
    knowledgeBases.find((knowledge) => knowledge.id === selectedKnowledgeId) ?? knowledgeBases[0];
  const preferredItem =
    items.find((item) => item.id === activeId) ??
    items.find((item) => item.knowledgeId === selectedKnowledge?.id) ??
    items[0];
  const activeItem = preferredItem;
  const activeKnowledge =
    knowledgeBases.find((knowledge) => knowledge.id === activeItem?.knowledgeId) ??
    selectedKnowledge;
  const activeMessages = activeItem ? chatMessages[activeItem.id] ?? [] : [];

  const createItem = () => {
    const fallbackKnowledgeId = selectedKnowledge?.id ?? activeKnowledge?.id ?? "";
    const newItem: ChatItem = {
      id: `${label}-${Date.now()}`,
      title: `新${label} ${items.length + 1}`,
      knowledgeId: fallbackKnowledgeId,
    };

    onItemsChange((current) => [newItem, ...current]);
    onChatMessagesChange((current) => ({ ...current, [newItem.id]: [] }));
    onActiveChange(newItem.id);
  };

  const updateKnowledge = (knowledgeId: string) => {
    if (!activeItem) return;
    onItemsChange((current) =>
      current.map((item) => (item.id === activeItem.id ? { ...item, knowledgeId } : item)),
    );
  };

  const removeActive = () => {
    if (!activeItem) return;
    onItemsChange((current) => current.filter((item) => item.id !== activeItem.id));
    onChatMessagesChange((current) => {
      const nextMessages = { ...current };
      delete nextMessages[activeItem.id];
      return nextMessages;
    });
    const nextItem = items.find((item) => item.id !== activeItem.id);
    if (nextItem) onActiveChange(nextItem.id);
  };

  const sendMessage = async (content: string) => {
    if (!activeItem || sending) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
    };
    const assistantMessage: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
    };
    const history = trimChatHistory(activeMessages);

    setSending(true);
    setStreamError("");
    onChatMessagesChange((current) => ({
      ...current,
      [activeItem.id]: [...(current[activeItem.id] ?? []), userMessage, assistantMessage],
    }));
    if (activeMessages.length === 0) {
      onItemsChange((current) =>
        current.map((item) =>
          item.id === activeItem.id ? { ...item, title: content.slice(0, 24) || item.title } : item,
        ),
      );
    }

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          history: history.map((message) => ({
            role: message.role,
            content: message.content,
          })),
          knowledgeId: activeKnowledge?.id,
          knowledgeName: activeKnowledge?.name,
          historyRounds: chatHistoryRounds,
        }),
      });
      if (!response.ok || !response.body) throw new Error("聊天服务暂不可用");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const boundary = buffer.lastIndexOf("\n\n");
        if (boundary === -1) continue;
        const ready = buffer.slice(0, boundary + 2);
        buffer = buffer.slice(boundary + 2);

        for (const event of parseSseChunk(ready)) {
          if (event.event === "delta") {
            const payload = JSON.parse(event.data) as { content?: string };
            assistantContent += payload.content ?? "";
            onChatMessagesChange((current) => ({
              ...current,
              [activeItem.id]: (current[activeItem.id] ?? []).map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, content: assistantContent }
                  : message,
              ),
            }));
          }
          if (event.event === "tool") {
            const payload = JSON.parse(event.data) as { label?: string };
            if (payload.label) {
              onChatMessagesChange((current) => ({
                ...current,
                [activeItem.id]: (current[activeItem.id] ?? []).map((message) =>
                  message.id === assistantMessage.id
                    ? { ...message, toolLabel: payload.label }
                    : message,
                ),
              }));
            }
          }
          if (event.event === "materials") {
            const payload = JSON.parse(event.data) as { items?: ChatMaterial[] };
            onChatMessagesChange((current) => ({
              ...current,
              [activeItem.id]: (current[activeItem.id] ?? []).map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, materials: payload.items ?? [] }
                  : message,
              ),
            }));
          }
          if (event.event === "catalogGraph") {
            const payload = JSON.parse(event.data) as CatalogGraph;
            onChatMessagesChange((current) => ({
              ...current,
              [activeItem.id]: (current[activeItem.id] ?? []).map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, catalogGraph: payload }
                  : message,
              ),
            }));
          }
          if (event.event === "error") {
            const payload = JSON.parse(event.data) as { message?: string };
            throw new Error(payload.message || "聊天生成失败");
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "聊天生成失败";
      setStreamError(message);
      onChatMessagesChange((current) => ({
        ...current,
        [activeItem.id]: (current[activeItem.id] ?? []).map((item) =>
          item.id === assistantMessage.id
            ? { ...item, content: `生成失败：${message}` }
            : item,
        ),
      }));
    } finally {
      setSending(false);
    }
  };

  const activeMaterials = activeMessages
    .filter((message) => message.role === "assistant")
    .flatMap((message) => message.materials ?? []);
  const activeCatalogGraph = [...activeMessages]
    .reverse()
    .find((message) => message.role === "assistant" && message.catalogGraph?.books.length)?.catalogGraph;

  return (
    <main className="workbench-page">
      <section className="chat-shell">
        <aside className="chat-sidebar">
          <button className="sidebar-back" onClick={onBackToKnowledgeList}>
            <ArrowLeft size={15} />
            选择知识库
          </button>
          <div className="knowledge-chip">
            <span className={`letter-icon small ${selectedKnowledge?.tone ?? "green"}`}>
              {selectedKnowledge?.name[0] ?? "知"}
            </span>
            <strong>{selectedKnowledge?.name ?? "暂无知识库"}</strong>
            <Send size={16} />
          </div>
          <div className="session-head">
            <h2>{label}</h2>
            <span>{items.length} 项</span>
            <button aria-label={`新建${label}`} onClick={createItem}>
              <Plus size={16} />
            </button>
            <button aria-label={`删除当前${label}`} onClick={removeActive} disabled={!activeItem}>
              <Trash2 size={16} />
            </button>
          </div>
          <label className="side-search">
            <Search size={15} />
            <input placeholder={`搜索${label}`} />
          </label>
          {items.map((item) => (
            <button
              key={item.id}
              className={item.id === activeItem?.id ? "session-item active" : "session-item"}
              onClick={() => onActiveChange(item.id)}
            >
              {item.title}
            </button>
          ))}
        </aside>

        <section className="chat-main">
          <header className="chat-main-head">
            <div>
              <strong>{activeItem?.title ?? `新${label}`}</strong>
              <span>{description}</span>
            </div>
            <div className="chat-header-actions">
              <label className="knowledge-select">
                <Database size={15} />
                <select
                  value={activeKnowledge?.id ?? ""}
                  onChange={(event) => updateKnowledge(event.target.value)}
                  disabled={!activeItem}
                >
                  {knowledgeBases.map((knowledge) => (
                    <option key={knowledge.id} value={knowledge.id}>
                      {knowledge.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </header>
          <div className="message-stream">
            {activeMessages.length === 0 ? (
              <div className="empty-chat">
                <MessageCircle size={24} />
                <strong>开始一段新对话</strong>
                <span>影视相关问题会自动检索当前知识库。</span>
              </div>
            ) : (
              activeMessages.map((message) => (
                <article
                  key={message.id}
                  className={message.role === "assistant" ? "ai-message" : "user-message"}
                >
                  {message.role === "assistant" && message.toolLabel && (
                    <span className="tool-label">{message.toolLabel}</span>
                  )}
                  {message.role === "assistant" ? (
                    <MarkdownMessage content={message.content || "正在生成..."} />
                  ) : (
                    message.content
                  )}
                </article>
              ))
            )}
            {streamError && <div className="chat-error">{streamError}</div>}
          </div>
          <ChatInput disabled={!activeItem || sending} onSend={sendMessage} />
        </section>
        <MaterialPanel materials={activeMaterials} />
        <CatalogGraphPanel graph={activeCatalogGraph} />
      </section>
    </main>
  );
}

function MaterialPanel({ materials }: { materials: ChatMaterial[] }) {
  const [filter, setFilter] = React.useState<"all" | "local" | "global">("all");
  const visibleMaterials =
    filter === "all" ? materials : materials.filter((item) => item.sourceType === filter);

  return (
    <aside className="material-panel">
      <div className="material-panel-head">
        <div>
          <strong>素材</strong>
          <span>{visibleMaterials.length}/{materials.length} 条</span>
        </div>
        <label className="material-filter">
          <select value={filter} onChange={(event) => setFilter(event.target.value as "all" | "local" | "global")}>
            <option value="all">全部素材</option>
            <option value="local">Local</option>
            <option value="global">Global</option>
          </select>
        </label>
      </div>
      <div className="material-list">
        {visibleMaterials.length ? (
          visibleMaterials.map((item) => (
            <article className="material-card" key={item.id}>
              <div className="material-preview">
                <img src={item.imageUrl} alt={`${item.fileName} ${item.paragraphId}`} loading="lazy" />
              </div>
              <div className="material-file">
                <span className="material-thumb">PDF</span>
                <div>
                  <strong title={item.fileName}>{item.fileName}</strong>
                  <small>
                    {item.sourceType === "local" ? "Local" : "Global"} · p.{item.pageIdx + 1} ·{" "}
                    {item.paragraphId}
                  </small>
                </div>
              </div>
              {item.text && <p>{item.text}</p>}
            </article>
          ))
        ) : (
          <div className="material-empty">
            {materials.length ? "当前筛选下没有素材。" : "影视知识库检索后会在这里显示 PDF 原文素材。"}
          </div>
        )}
      </div>
    </aside>
  );
}

function CatalogGraphPanel({ graph }: { graph?: CatalogGraph }) {
  const books = graph?.books ?? [];
  const [activeFileId, setActiveFileId] = React.useState<string>("");
  const activeBook = books.find((book) => book.fileId === activeFileId) ?? books[0];
  const selectedCatalogs = activeBook?.catalogs.filter((node) => node.selected) ?? [];
  const contextCatalogs = activeBook?.catalogs.filter((node) => !node.selected) ?? [];
  const nodeTypes = React.useMemo(() => ({ catalogNode: GraphCatalogNode }), []);
  const { nodes, edges } = React.useMemo(
    () => buildCatalogFlow(activeBook, selectedCatalogs, contextCatalogs),
    [activeBook, contextCatalogs, selectedCatalogs],
  );

  React.useEffect(() => {
    if (!books.length) {
      setActiveFileId("");
      return;
    }
    if (!books.some((book) => book.fileId === activeFileId)) {
      setActiveFileId(books[0].fileId);
    }
  }, [activeFileId, books]);

  return (
    <aside className="catalog-graph-panel">
      <div className="material-panel-head">
        <div>
          <strong>图谱</strong>
          <span>{activeBook ? `${selectedCatalogs.length} 个命中目录` : "等待 global 检索"}</span>
        </div>
        <label className="material-filter">
          <select
            value={activeBook?.fileId ?? ""}
            onChange={(event) => setActiveFileId(event.target.value)}
            disabled={!books.length}
          >
            {books.length ? (
              books.map((book) => (
                <option key={book.fileId} value={book.fileId}>
                  {book.fileName}
                </option>
              ))
            ) : (
              <option value="">暂无书籍</option>
            )}
          </select>
        </label>
      </div>
      <div className="catalog-graph-canvas">
        {activeBook ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.22 }}
            minZoom={0.55}
            maxZoom={1.7}
            nodesDraggable
            nodesConnectable={false}
            elementsSelectable
            panOnDrag
            zoomOnScroll
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#dbe8fb" gap={24} size={1} />
            <Controls showInteractive={false} position="bottom-left" />
            <MiniMap
              pannable
              zoomable
              position="bottom-right"
              nodeStrokeWidth={2}
              nodeColor={(node) => {
                const tone = node.data?.tone;
                if (tone === "focus") return "#f1a72f";
                if (tone === "book") return "#e66a61";
                if (tone === "leaf") return "#55b7d9";
                return "#36b39a";
              }}
            />
          </ReactFlow>
        ) : (
          <div className="material-empty">Global 检索后会在这里显示图谱关系。</div>
        )}
      </div>
    </aside>
  );
}

function GraphCatalogNode({ data }: NodeProps<Node<GraphNodeData>>) {
  const size = data.size ?? (data.selected ? 74 : 54);
  return (
    <div
      className={`flow-catalog-node ${data.tone ?? "context"}${data.selected ? " selected" : ""}`}
      style={{ width: size, height: size }}
      title={data.eyebrow ? `${data.eyebrow} / ${data.label}` : data.label}
    >
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      {data.icon === "book" && <BookOpen size={14} />}
      <strong>{data.label}</strong>
    </div>
  );
}

function buildCatalogFlow(
  activeBook: CatalogGraphBook | undefined,
  selectedCatalogs: CatalogGraphNode[],
  contextCatalogs: CatalogGraphNode[],
): { nodes: Node<GraphNodeData>[]; edges: Edge[] } {
  if (!activeBook) {
    return { nodes: [], edges: [] };
  }

  const focusNodes = selectedCatalogs.slice(0, 3);
  const nearbyNodes = contextCatalogs.slice(0, 12);
  const centerX = 360;
  const centerY = 230;
  const nodes: Node<GraphNodeData>[] = [
    {
      id: "book",
      type: "catalogNode",
      position: { x: 64, y: centerY - 40 },
      data: {
        label: activeBook.fileName,
        eyebrow: "文件",
        icon: "book",
        tone: "book",
        size: 78,
      },
    },
  ];
  const edges: Edge[] = [];

  focusNodes.forEach((node, index) => {
    const nodeId = `focus-${node.catalogId}`;
    const offsetY = (index - (focusNodes.length - 1) / 2) * 92;
    nodes.push({
      id: nodeId,
      type: "catalogNode",
      position: { x: centerX - 38, y: centerY + offsetY - 38 },
      data: {
        label: node.title,
        eyebrow: node.path.slice(0, -1).join(" / ") || "命中目录",
        selected: true,
        tone: "focus",
        size: 76,
      },
    });
    edges.push({
      id: `book-${nodeId}`,
      source: "book",
      target: nodeId,
      type: "default",
      animated: true,
      label: "命中",
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
      style: { stroke: "#7597d8", strokeWidth: 1.8 },
      labelStyle: { fill: "#6d7f9d", fontSize: 10, fontWeight: 700 },
      labelBgStyle: { fill: "#ffffff", fillOpacity: 0.82 },
    });
  });

  nearbyNodes.forEach((node, index) => {
    const angle = -Math.PI * 0.9 + (index / Math.max(nearbyNodes.length - 1, 1)) * Math.PI * 1.8;
    const radius = index % 3 === 0 ? 190 : index % 3 === 1 ? 245 : 145;
    const size = index % 4 === 0 ? 58 : 48;
    const nodeId = `context-${node.catalogId}`;
    const target = focusNodes[index % Math.max(focusNodes.length, 1)];
    nodes.push({
      id: nodeId,
      type: "catalogNode",
      position: {
        x: centerX + Math.cos(angle) * radius - size / 2,
        y: centerY + Math.sin(angle) * radius - size / 2,
      },
      data: {
        label: node.title,
        eyebrow: node.path.slice(0, -1).join(" / ") || "周围目录",
        tone: index % 4 === 0 ? "leaf" : "context",
        size,
      },
    });
    if (target) {
      edges.push({
        id: `${nodeId}-focus-${target.catalogId}`,
        source: nodeId,
        target: `focus-${target.catalogId}`,
        type: "default",
        label: "相关",
        style: { stroke: "#b7c5d9", strokeWidth: 1.2 },
        labelStyle: { fill: "#7b8ba6", fontSize: 9, fontWeight: 700 },
        labelBgStyle: { fill: "#ffffff", fillOpacity: 0.72 },
      });
    }
  });

  return { nodes, edges };
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-message">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function ChatInput({
  disabled,
  onSend,
}: {
  disabled: boolean;
  onSend: (content: string) => void;
}) {
  const [value, setValue] = React.useState("");
  const submit = () => {
    const content = value.trim();
    if (!content || disabled) return;
    setValue("");
    onSend(content);
  };

  return (
    <div className="chat-input">
      <textarea
        placeholder="请输入消息..."
        value={value}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
      />
      <div className="input-actions">
        <button className="send-button" aria-label="发送" disabled={disabled || !value.trim()} onClick={submit}>
          <Send size={17} />
        </button>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
