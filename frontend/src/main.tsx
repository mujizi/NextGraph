import React from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  Bot,
  BookOpen,
  Check,
  CircleCheck,
  Clock,
  Database,
  Edit3,
  FileText,
  Filter,
  FolderOpen,
  FolderUp,
  House,
  Link2,
  MessageCircle,
  MoreHorizontal,
  Network,
  Paperclip,
  Plus,
  Play,
  RefreshCw,
  Search,
  Send,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import appLogo from "./assets/app-logo.png";
import "./styles.css";

type Page = "home" | "knowledge" | "chat" | "search";
type Status = "已解析" | "待解析" | "部分失败";
type BuildStatus = "idle" | "uploading" | "building" | "completed" | "failed";

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

type UploadedDocument = {
  id: string;
  name: string;
  path: string;
  type: string;
  uploadedAt: string;
  size: string;
  parser: string;
  selected: boolean;
  progress: number;
  state: string;
};

type BuildParams = {
  kbId: string;
  userId: string;
  milvusDb: string;
  outputRoot: string;
  gpus: string;
  workersPerGpu: number;
  method: "auto" | "txt" | "ocr";
  backend: string;
  lang: string;
  startPage: string;
  endPage: string;
  formula: boolean;
  table: boolean;
  vectorConcurrency: number;
  maxChunkChars: number;
  chunkOverlapChars: number;
  embeddingModel: string;
};

type DatabaseBuildProgress = {
  task_id: string;
  status: string;
  stage: string;
  message: string;
  milvus_db?: string;
  progress_percent: number;
  files: Array<{
    path: string;
    stage: string;
    state: string;
    progress: number;
    summary?: {
      chunks?: number;
      entities?: number;
      relations?: number;
      failed_embeddings?: number;
    };
  }>;
};

type BuildFileProgress = DatabaseBuildProgress["files"][number];

type BuiltLibrary = {
  task_id: string;
  kb_id: string;
  user_id: string;
  milvus_db: string;
  status: string;
  stage: string;
  message: string;
  progress_percent: number;
  file_count: number;
  completed: number;
  failed: number;
  updated_at: number;
};

type LibraryCollectionView = {
  collection: string;
  matched_count: number | null;
  sample_count: number;
  samples: Array<Record<string, unknown>>;
};

type LibraryInspection = {
  kb_id: string;
  user_id: string;
  milvus_db: string;
  collections: Record<"entities" | "relations" | "passages", LibraryCollectionView>;
};

const initialKnowledgeBases: KnowledgeBase[] = [
  {
    id: "kb-yingpu",
    name: "影谱项目库",
    date: "2026-04-27",
    owner: "Mengna",
    files: 128,
    status: "已解析",
    tone: "green",
  },
  {
    id: "kb-script",
    name: "电影剧本文档",
    date: "2026-04-25",
    owner: "Mengna",
    files: 36,
    status: "部分失败",
    tone: "blue",
  },
];

const initialChats: ChatItem[] = [
  { id: "chat-1", title: "一个真正有力量的主角", knowledgeId: "kb-yingpu" },
  { id: "chat-2", title: "为什么观众对电影角色共情", knowledgeId: "kb-yingpu" },
  { id: "chat-3", title: "第一章电影剧本写作基础", knowledgeId: "kb-script" },
];

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const initialDocuments: UploadedDocument[] = [
  {
    id: "sample-1",
    type: "PDF",
    name: "电影剧本写作基础与角色弧光.pdf",
    path: "/opt/Workspace/CRX/NextGraph/backend/storage/samples/电影剧本写作基础与角色弧光.pdf",
    uploadedAt: "21/04/2026 09:59:20",
    size: "12.4 MB",
    parser: "Book",
    selected: true,
    progress: 100,
    state: "完成",
  },
  {
    id: "sample-2",
    type: "DOC",
    name: "第一章电影剧本结构.docx",
    path: "/opt/Workspace/CRX/NextGraph/backend/storage/samples/第一章电影剧本结构.docx",
    uploadedAt: "21/04/2026 10:12:03",
    size: "4.8 MB",
    parser: "Text",
    selected: false,
    progress: 63,
    state: "63%",
  },
  {
    id: "sample-3",
    type: "MD",
    name: "角色动机与冲突设计.md",
    path: "/opt/Workspace/CRX/NextGraph/backend/storage/samples/角色动机与冲突设计.md",
    uploadedAt: "21/04/2026 10:40:11",
    size: "128 KB",
    parser: "Markdown",
    selected: false,
    progress: 0,
    state: "等待中",
  },
];

function formatUploadedAt(value: number) {
  return new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });
}

function formatFileSize(size: number) {
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

function toUploadedDocument(file: {
  id: string;
  name: string;
  path: string;
  size: number;
  uploaded_at: number;
  extension: string;
}): UploadedDocument {
  const extension = file.extension.toUpperCase();
  return {
    id: file.id,
    name: file.name,
    path: file.path,
    type: extension,
    uploadedAt: formatUploadedAt(file.uploaded_at),
    size: formatFileSize(file.size),
    parser: extension === "MD" ? "Markdown" : extension === "PDF" ? "Book" : "Text",
    selected: false,
    progress: 0,
    state: "待构建",
  };
}

function App() {
  const [page, setPage] = React.useState<Page>("home");
  const [knowledgeBases, setKnowledgeBases] = React.useState<KnowledgeBase[]>(
    initialKnowledgeBases,
  );
  const [creatingKnowledge, setCreatingKnowledge] = React.useState(false);
  const [selectedKnowledgeId, setSelectedKnowledgeId] = React.useState<string | null>(null);
  const [selectedChatKnowledgeId, setSelectedChatKnowledgeId] = React.useState<string | null>(null);
  const [selectedSearchKnowledgeId, setSelectedSearchKnowledgeId] = React.useState<string | null>(
    null,
  );
  const [uploadedDocuments, setUploadedDocuments] =
    React.useState<UploadedDocument[]>(initialDocuments);
  const [chats, setChats] = React.useState<ChatItem[]>(initialChats);
  const [activeChatId, setActiveChatId] = React.useState(initialChats[0].id);

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
      date: "2026-04-27",
      owner: "Mengna",
      files: 0,
      status: "待解析",
      tone: nextIndex % 2 === 0 ? "purple" : "amber",
    };

    setKnowledgeBases((current) => [...current, newKnowledge]);
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
          chats={chats}
          onCreateKnowledge={() => setCreatingKnowledge(true)}
          onOpenKnowledge={openKnowledgeBase}
          onOpenChat={(id) => openConversationKnowledge("chat", id)}
          onRenameKnowledge={renameKnowledgeBase}
        />
      )}
      {page === "knowledge" && (
        <KnowledgePage
          knowledgeBases={knowledgeBases}
          selectedKnowledgeId={selectedKnowledgeId}
          documents={uploadedDocuments}
          onDocumentsChange={setUploadedDocuments}
          onCreateKnowledge={() => setCreatingKnowledge(true)}
          onOpenKnowledge={openKnowledgeBase}
          onRenameKnowledge={renameKnowledgeBase}
          onBackToList={() => setSelectedKnowledgeId(null)}
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
  chats,
  onCreateKnowledge,
  onOpenKnowledge,
  onOpenChat,
  onRenameKnowledge,
}: {
  knowledgeBases: KnowledgeBase[];
  chats: ChatItem[];
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onOpenChat: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
}) {
  const parsedCount = knowledgeBases.filter((knowledge) => knowledge.status === "已解析").length;
  const fileCount = knowledgeBases.reduce((total, knowledge) => total + knowledge.files, 0);

  return (
    <main className="home page-shell">
      <section className="home-hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <Network size={16} />
            NextGraph Knowledge Studio
          </span>
          <h1>
            欢迎来到 <span>影谱知识库</span>
          </h1>
          <p>管理知识库、解析文件、抽取图谱并开始智能问答</p>
        </div>
        <div className="hero-stats" aria-label="知识库概览">
          <MetricCard label="知识库" value={knowledgeBases.length} hint={`${parsedCount} 个已解析`} />
          <MetricCard label="文档" value={fileCount} hint="可用于构建与检索" />
          <MetricCard label="会话" value={chats.length} hint="知识库问答记录" />
        </div>
        <button className="primary-action" onClick={onCreateKnowledge}>
          <Plus size={18} />
          新建知识库
        </button>
      </section>

      <WorkflowStrip />

      <KnowledgeGrid
        title="知识库"
        knowledgeBases={knowledgeBases}
        onCreateKnowledge={onCreateKnowledge}
        onOpenKnowledge={onOpenKnowledge}
        onRenameKnowledge={onRenameKnowledge}
        showCreateAction={false}
      />

      <ChatLaunchSection
        chats={chats}
        knowledgeBases={knowledgeBases}
        onOpenChat={onOpenChat}
      />
    </main>
  );
}

function MetricCard({ label, value, hint }: { label: string; value: number; hint: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function WorkflowStrip() {
  const steps = [
    ["上传文件", "PDF、Word、Markdown、TXT 等多格式接入", <Upload size={17} />],
    ["构建知识库", "解析、分块、向量化并写入 Milvus", <Database size={17} />],
    ["智能使用", "在 Chat 和搜索里复用知识库能力", <Bot size={17} />],
  ];

  return (
    <section className="workflow-strip" aria-label="知识库构建流程">
      {steps.map(([title, description, icon]) => (
        <article className="workflow-step" key={String(title)}>
          <span>{icon}</span>
          <div>
            <strong>{title}</strong>
            <p>{description}</p>
          </div>
        </article>
      ))}
    </section>
  );
}

function ChatLaunchSection({
  chats,
  knowledgeBases,
  onOpenChat,
}: {
  chats: ChatItem[];
  knowledgeBases: KnowledgeBase[];
  onOpenChat: (id: string) => void;
}) {
  return (
    <section className="home-section">
      <div className="section-head">
        <div className="section-title">
          <h2>最近聊天</h2>
          <MessageCircle size={23} />
        </div>
      </div>
      <div className="chat-launch-grid">
        {chats.slice(0, 3).map((chat) => {
          const knowledge = knowledgeBases.find((item) => item.id === chat.knowledgeId);
          return (
            <button
              className="chat-launch-card"
              key={chat.id}
              onClick={() => onOpenChat(chat.knowledgeId)}
            >
              <span className={`letter-icon small ${knowledge?.tone ?? "green"}`}>
                {knowledge?.name[0] ?? "知"}
              </span>
              <div>
                <strong>{chat.title}</strong>
                <p>{knowledge?.name ?? "未绑定知识库"}</p>
              </div>
              <MessageCircle size={18} />
            </button>
          );
        })}
      </div>
    </section>
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
        <div className="card-topline">
          <div className={`letter-icon ${knowledge.tone}`}>{knowledge.name[0] || "知"}</div>
          <span className={`status ${knowledge.status === "已解析" ? "success" : "warn"}`}>
            {knowledge.status}
          </span>
        </div>
        <MoreHorizontal size={18} className="card-more" />
        <h3>{knowledge.name}</h3>
        <p>
          {knowledge.date} · {knowledge.owner} · {knowledge.files} files
        </p>
        <div className="card-signal">
          <span><CircleCheck size={14} /> 解析</span>
          <span><Clock size={14} /> 最近更新</span>
        </div>
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
  selectedKnowledgeId,
  documents,
  onDocumentsChange,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
  onBackToList,
}: {
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeId: string | null;
  documents: UploadedDocument[];
  onDocumentsChange: React.Dispatch<React.SetStateAction<UploadedDocument[]>>;
  onCreateKnowledge: () => void;
  onOpenKnowledge: (id: string) => void;
  onRenameKnowledge: (id: string, name: string) => void;
  onBackToList: () => void;
}) {
  const selectedKnowledge = knowledgeBases.find((knowledge) => knowledge.id === selectedKnowledgeId);

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

  return (
    <main className="workbench-page">
      <BuildWorkspace
        knowledge={selectedKnowledge}
        documents={documents}
        onDocumentsChange={onDocumentsChange}
        onBackToList={onBackToList}
      />
    </main>
  );
}

function BuildWorkspace({
  knowledge,
  documents,
  onDocumentsChange,
  onBackToList,
}: {
  knowledge: KnowledgeBase;
  documents: UploadedDocument[];
  onDocumentsChange: React.Dispatch<React.SetStateAction<UploadedDocument[]>>;
  onBackToList: () => void;
}) {
  const [activeTab, setActiveTab] = React.useState<"upload" | "build">("upload");
  const [status, setStatus] = React.useState<BuildStatus>("idle");
  const [message, setMessage] = React.useState("选择服务器文件并配置参数后即可开始构建。");
  const [taskId, setTaskId] = React.useState<string | null>(null);
  const [builtLibraries, setBuiltLibraries] = React.useState<BuiltLibrary[]>([]);
  const [currentBuildFiles, setCurrentBuildFiles] = React.useState<BuildFileProgress[]>([]);
  const [params, setParams] = React.useState<BuildParams>({
    kbId: knowledge.id,
    userId: "admin_user",
    milvusDb: "crx",
    outputRoot: "/opt/Workspace/CRX/NextGraph/backend/storage/mineru_output",
    gpus: "0",
    workersPerGpu: 1,
    method: "auto",
    backend: "pipeline",
    lang: "ch",
    startPage: "",
    endPage: "",
    formula: true,
    table: true,
    vectorConcurrency: 10,
    maxChunkChars: 1500,
    chunkOverlapChars: 150,
    embeddingModel: "text-embedding-3-small",
  });

  const selectedCount = documents.filter((file) => file.selected).length;
  const taskStorageKey = `nextgraph-build-task:${params.kbId || knowledge.id}`;

  const loadBuiltLibraries = React.useCallback(() => {
    fetch(`${API_BASE}/database_build/libraries?kb_id=${encodeURIComponent(params.kbId || knowledge.id)}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then((data: { libraries: BuiltLibrary[] }) => setBuiltLibraries(data.libraries ?? []))
      .catch(() => setBuiltLibraries([]));
  }, [knowledge.id, params.kbId]);

  const applyBuildProgress = React.useCallback(
    (data: DatabaseBuildProgress) => {
      setCurrentBuildFiles(data.files);
      setTaskId(data.task_id);
      setStatus(
        data.status === "completed"
          ? "completed"
          : data.status === "failed" || data.status === "partial_failed"
            ? "failed"
            : "building",
      );
      setMessage(
        `${data.message} 目标库 ${data.milvus_db ?? params.milvusDb}，整体 ${Math.round(
          data.progress_percent,
        )}%`,
      );
      window.localStorage.setItem(taskStorageKey, data.task_id);
      if (["completed", "failed", "partial_failed"].includes(data.status)) {
        loadBuiltLibraries();
      }
    },
    [loadBuiltLibraries, params.milvusDb, taskStorageKey],
  );

  const openBuildTask = (nextTaskId: string) => {
    fetch(`${API_BASE}/database_build/progress/${nextTaskId}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then((data: DatabaseBuildProgress) => {
        applyBuildProgress(data);
        window.localStorage.setItem(taskStorageKey, data.task_id);
      })
      .catch(() => setMessage("无法读取该构建任务进度。"));
  };

  React.useEffect(() => {
    fetch(`${API_BASE}/files`)
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then((data: { files: Array<Parameters<typeof toUploadedDocument>[0]> }) => {
        if (data.files?.length) {
          onDocumentsChange((current) => [
            ...data.files.map(toUploadedDocument),
            ...current.filter((item) => item.id.startsWith("sample-")),
          ]);
        }
      })
      .catch(() => {
        setMessage("暂时没有读取到服务器文件，仍可先查看页面流程。");
      });
  }, [onDocumentsChange]);

  React.useEffect(() => {
    loadBuiltLibraries();
  }, [loadBuiltLibraries]);

  React.useEffect(() => {
    const savedTaskId = window.localStorage.getItem(taskStorageKey);
    const endpoint = savedTaskId
      ? `${API_BASE}/database_build/progress/${savedTaskId}`
      : `${API_BASE}/database_build/latest?kb_id=${encodeURIComponent(params.kbId || knowledge.id)}`;

    fetch(endpoint)
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then((data: DatabaseBuildProgress) => {
        applyBuildProgress(data);
        if (["completed", "failed", "partial_failed"].includes(data.status)) {
          setStatus(data.status === "completed" ? "completed" : "failed");
        }
      })
      .catch(() => undefined);
  }, [applyBuildProgress, knowledge.id, params.kbId, taskStorageKey]);

  React.useEffect(() => {
    if (!taskId || status !== "building") return undefined;

    const timer = window.setInterval(() => {
      fetch(`${API_BASE}/database_build/progress/${taskId}`)
        .then((response) => (response.ok ? response.json() : Promise.reject(response)))
        .then((data: DatabaseBuildProgress) => {
          applyBuildProgress(data);
          if (["completed", "failed", "partial_failed"].includes(data.status)) {
            setStatus(data.status === "completed" ? "completed" : "failed");
            setMessage(data.status === "completed" ? "数据库构建完成，已写入 Milvus。" : "构建结束，但存在失败文档。");
          }
        })
        .catch(() => {
          setStatus("failed");
          setMessage("无法获取构建进度，请检查后端服务。");
        });
    }, 1200);

    return () => window.clearInterval(timer);
  }, [applyBuildProgress, status, taskId]);

  const uploadFiles = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files ?? []);
    if (!selectedFiles.length) return;

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));
    setStatus("uploading");
    setMessage(`正在上传 ${selectedFiles.length} 个文件...`);

    try {
      const response = await fetch(`${API_BASE}/files/upload`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { files: Array<Parameters<typeof toUploadedDocument>[0]> };
      onDocumentsChange((current) => [
        ...data.files.map((file) => ({ ...toUploadedDocument(file), selected: true })),
        ...current,
      ]);
      setActiveTab("build");
      setStatus("idle");
      setMessage("上传完成，已自动选中新文件。");
    } catch (error) {
      setStatus("failed");
      setMessage(error instanceof Error ? error.message : "上传失败，请检查后端服务。");
    } finally {
      event.target.value = "";
    }
  };

  const startBuild = async () => {
    const selectedFiles = documents.filter((file) => file.selected);
    if (!selectedFiles.length) {
      setMessage("请至少选择一个文件。");
      return;
    }

    setStatus("building");
    setMessage("已提交构建任务，正在等待进度回传...");
    setCurrentBuildFiles(
      selectedFiles.map((file) => ({
        path: file.path,
        stage: "queued",
        state: "排队中",
        progress: 0,
      })),
    );

    try {
      const response = await fetch(`${API_BASE}/database_build/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_paths: selectedFiles.map((file) => file.path),
          kb_id: params.kbId || knowledge.id,
          user_id: params.userId,
          milvus_db: params.milvusDb,
          output_root: params.outputRoot,
          gpus: params.gpus.split(",").map((gpu) => gpu.trim()).filter(Boolean),
          workers_per_gpu: params.workersPerGpu,
          method: params.method,
          lang: params.lang || null,
          backend: params.backend,
          start_page: params.startPage ? Number(params.startPage) : null,
          end_page: params.endPage ? Number(params.endPage) : null,
          formula: params.formula,
          table: params.table,
          source: params.kbId || knowledge.id,
          vector_concurrency: params.vectorConcurrency,
          max_chunk_chars: params.maxChunkChars,
          chunk_overlap_chars: params.chunkOverlapChars,
          embedding_model: params.embeddingModel,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { task_id: string };
      setTaskId(data.task_id);
      window.localStorage.setItem(taskStorageKey, data.task_id);
      loadBuiltLibraries();
    } catch (error) {
      setStatus("failed");
      setMessage(error instanceof Error ? error.message : "构建任务提交失败。");
    }
  };

  const updateSelection = (id: string, selected: boolean) => {
    onDocumentsChange((current) =>
      current.map((file) => (file.id === id ? { ...file, selected } : file)),
    );
  };

  return (
    <section className="content-panel">
      <div className="knowledge-header">
        <div>
          <button className="back-link" onClick={onBackToList}>
            <ArrowLeft size={16} />
            全部知识库
          </button>
          <div className="title-row">
            <h1>数据库构建</h1>
            <span className="tag">{knowledge.name}</span>
          </div>
          <p>{message}</p>
        </div>
        <div className="toolbar">
          <button className="soft-button" onClick={() => setActiveTab("build")}>
            <Sparkles size={16} />
            构建数据库
          </button>
          <button className="outline-button">
            <Filter size={16} />
            筛选
          </button>
          <label className="toolbar-search">
            <Search size={16} />
            <input placeholder="搜索" />
          </label>
          <button className="outline-button">
            <FolderUp size={16} />
            文件夹
          </button>
          <label className="square-primary" aria-label="新增文件" title="上传文件">
            <Upload size={17} />
            <input type="file" multiple onChange={uploadFiles} />
          </label>
        </div>
      </div>

      <div className="workspace-tabs">
        <button className={activeTab === "upload" ? "active" : ""} onClick={() => setActiveTab("upload")}>
          <Upload size={16} />
          上传文件
        </button>
        <button className={activeTab === "build" ? "active" : ""} onClick={() => setActiveTab("build")}>
          <Database size={16} />
          构建数据库
        </button>
      </div>

      {activeTab === "upload" ? (
        <UploadPanel
          status={status}
          documents={documents}
          onUpload={uploadFiles}
          onGoBuild={() => setActiveTab("build")}
        />
      ) : (
        <>
          <BuiltLibraryPanel
            libraries={builtLibraries}
            currentTaskId={taskId}
            onOpenTask={openBuildTask}
          />
          <DatabaseVisualPanel
            libraries={builtLibraries}
            currentTaskId={taskId}
            onOpenTask={openBuildTask}
          />
          <BuildControlPanel
            params={params}
            selectedCount={selectedCount}
            status={status}
            onParamsChange={setParams}
            onStartBuild={startBuild}
          />
          <FileTable
            files={documents}
            buildFiles={currentBuildFiles}
            onSelectionChange={updateSelection}
          />
        </>
      )}
    </section>
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
      <section className="picker-hero">
        <span className="eyebrow">
          <BookOpen size={16} />
          Knowledge Bases
        </span>
        <h1>{title}</h1>
        <p>选择一个知识库，继续上传文件、构建数据库，或进入问答与搜索。</p>
      </section>
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
        <label className="big-search">
          <Search size={22} />
          <input placeholder="搜索知识库内容..." />
        </label>
      </div>
    </main>
  );
}

function UploadPanel({
  status,
  documents,
  onUpload,
  onGoBuild,
}: {
  status: BuildStatus;
  documents: UploadedDocument[];
  onUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onGoBuild: () => void;
}) {
  return (
    <div className="upload-grid">
      <label className="upload-drop">
        <Upload size={30} />
        <strong>{status === "uploading" ? "正在上传..." : "选择文件上传到服务器"}</strong>
        <span>支持 PDF、Word、PPT、Excel、Markdown、TXT、JSON，可多选上传。</span>
        <input type="file" multiple onChange={onUpload} />
      </label>
      <div className="upload-summary">
        <div className="summary-card">
          <FileText size={20} />
          <span>服务器文件</span>
          <strong>{documents.length}</strong>
        </div>
        <div className="summary-card">
          <Check size={20} />
          <span>已选构建</span>
          <strong>{documents.filter((file) => file.selected).length}</strong>
        </div>
        <button className="primary-small" onClick={onGoBuild}>
          <Database size={16} />
          去构建
        </button>
      </div>
    </div>
  );
}

function BuildControlPanel({
  params,
  selectedCount,
  status,
  onParamsChange,
  onStartBuild,
}: {
  params: BuildParams;
  selectedCount: number;
  status: BuildStatus;
  onParamsChange: React.Dispatch<React.SetStateAction<BuildParams>>;
  onStartBuild: () => void;
}) {
  const disabled = status === "building" || status === "uploading";

  const update = <Key extends keyof BuildParams>(key: Key, value: BuildParams[Key]) => {
    onParamsChange((current) => ({ ...current, [key]: value }));
  };

  return (
    <div className="build-panel">
      <div className="build-panel-head">
        <div>
          <span>
            <SlidersHorizontal size={16} />
            构建参数
          </span>
          <strong>已选择 {selectedCount} 个文件</strong>
        </div>
        <button className="primary-small" onClick={onStartBuild} disabled={disabled}>
          {status === "building" ? <RefreshCw size={16} /> : <Play size={16} />}
          {status === "building" ? "构建中" : "开始构建"}
        </button>
      </div>
      <div className="param-grid">
        <label>
          <span>知识库 ID</span>
          <input value={params.kbId} onChange={(event) => update("kbId", event.target.value)} />
        </label>
        <label>
          <span>用户 ID</span>
          <input value={params.userId} onChange={(event) => update("userId", event.target.value)} />
        </label>
        <label>
          <span>Milvus DB</span>
          <input list="milvus-db-options" value={params.milvusDb} onChange={(event) => update("milvusDb", event.target.value)} />
          <datalist id="milvus-db-options">
            <option value="crx" />
            <option value="default" />
            <option value="nextgraph" />
          </datalist>
        </label>
        <label>
          <span>输出目录</span>
          <input value={params.outputRoot} onChange={(event) => update("outputRoot", event.target.value)} />
        </label>
        <label>
          <span>GPU</span>
          <input value={params.gpus} onChange={(event) => update("gpus", event.target.value)} />
        </label>
        <label>
          <span>每 GPU Worker</span>
          <input
            type="number"
            min={1}
            max={16}
            value={params.workersPerGpu}
            onChange={(event) => update("workersPerGpu", Number(event.target.value))}
          />
        </label>
        <label>
          <span>解析方式</span>
          <select value={params.method} onChange={(event) => update("method", event.target.value as BuildParams["method"])}>
            <option value="auto">auto</option>
            <option value="txt">txt</option>
            <option value="ocr">ocr</option>
          </select>
        </label>
        <label>
          <span>后端</span>
          <input list="mineru-backend-options" value={params.backend} onChange={(event) => update("backend", event.target.value)} />
          <datalist id="mineru-backend-options">
            <option value="pipeline" />
          </datalist>
        </label>
        <label>
          <span>语言</span>
          <input list="mineru-lang-options" value={params.lang} onChange={(event) => update("lang", event.target.value)} />
          <datalist id="mineru-lang-options">
            <option value="ch" />
            <option value="en" />
            <option value="ja" />
            <option value="ko" />
          </datalist>
        </label>
        <label>
          <span>起始页</span>
          <input value={params.startPage} onChange={(event) => update("startPage", event.target.value)} />
        </label>
        <label>
          <span>结束页</span>
          <input value={params.endPage} onChange={(event) => update("endPage", event.target.value)} />
        </label>
        <label>
          <span>向量并发</span>
          <input
            type="number"
            min={1}
            max={64}
            value={params.vectorConcurrency}
            onChange={(event) => update("vectorConcurrency", Number(event.target.value))}
          />
        </label>
        <label>
          <span>分块字符</span>
          <input
            type="number"
            min={200}
            max={12000}
            value={params.maxChunkChars}
            onChange={(event) => update("maxChunkChars", Number(event.target.value))}
          />
        </label>
        <label>
          <span>分块重叠</span>
          <input
            type="number"
            min={0}
            max={4000}
            value={params.chunkOverlapChars}
            onChange={(event) => update("chunkOverlapChars", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Embedding</span>
          <input
            value={params.embeddingModel}
            onChange={(event) => update("embeddingModel", event.target.value)}
          />
        </label>
        <label className="toggle-field">
          <input
            type="checkbox"
            checked={params.formula}
            onChange={(event) => update("formula", event.target.checked)}
          />
          <span>公式解析</span>
        </label>
        <label className="toggle-field">
          <input
            type="checkbox"
            checked={params.table}
            onChange={(event) => update("table", event.target.checked)}
          />
          <span>表格解析</span>
        </label>
      </div>
    </div>
  );
}

function BuiltLibraryPanel({
  libraries,
  currentTaskId,
  onOpenTask,
}: {
  libraries: BuiltLibrary[];
  currentTaskId: string | null;
  onOpenTask: (taskId: string) => void;
}) {
  if (!libraries.length) {
    return (
      <div className="built-library-panel empty">
        <Database size={17} />
        <span>当前知识库还没有构建记录</span>
      </div>
    );
  }

  return (
    <div className="built-library-panel">
      <div className="built-library-head">
        <span>
          <Database size={16} />
          已构建记录
        </span>
        <strong>{libraries.length}</strong>
      </div>
      <div className="built-library-list">
        {libraries.slice(0, 4).map((item) => (
          <button
            className={item.task_id === currentTaskId ? "built-library active" : "built-library"}
            key={item.task_id}
            onClick={() => onOpenTask(item.task_id)}
          >
            <div>
              <strong>{item.milvus_db}</strong>
              <span>{item.completed}/{item.file_count} 文件 · {Math.round(item.progress_percent)}%</span>
            </div>
            <span className={item.status === "completed" ? "done" : "processing"}>
              {item.status === "completed" ? "已入库" : item.status === "running" ? "构建中" : "有失败"}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function DatabaseVisualPanel({
  libraries,
  currentTaskId,
  onOpenTask,
}: {
  libraries: BuiltLibrary[];
  currentTaskId: string | null;
  onOpenTask: (taskId: string) => void;
}) {
  const selectedLibrary =
    libraries.find((item) => item.task_id === currentTaskId) ??
    libraries.find((item) => item.status === "completed") ??
    libraries[0];
  const [inspection, setInspection] = React.useState<LibraryInspection | null>(null);
  const [queryText, setQueryText] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  const loadInspection = React.useCallback(() => {
    if (!selectedLibrary) return;
    setLoading(true);
    setError("");
    const params = new URLSearchParams({
      user_id: selectedLibrary.user_id,
      milvus_db: selectedLibrary.milvus_db,
      limit: "8",
    });
    fetch(`${API_BASE}/database_build/library/${encodeURIComponent(selectedLibrary.kb_id)}?${params}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(response)))
      .then((data: LibraryInspection) => setInspection(data))
      .catch(() => {
        setInspection(null);
        setError("无法读取 Milvus 数据，请确认后端能访问目标数据库。");
      })
      .finally(() => setLoading(false));
  }, [selectedLibrary]);

  React.useEffect(() => {
    loadInspection();
  }, [loadInspection]);

  if (!selectedLibrary) {
    return (
      <div className="db-visual-panel empty">
        <Database size={17} />
        <span>暂无可访问的构建数据库</span>
      </div>
    );
  }

  const collections = inspection?.collections;
  const lowerQuery = queryText.trim().toLowerCase();
  const filterRows = (rows: Array<Record<string, unknown>>) => {
    if (!lowerQuery) return rows;
    return rows.filter((row) => JSON.stringify(row).toLowerCase().includes(lowerQuery));
  };

  return (
    <div className="db-visual-panel">
      <div className="db-visual-head">
        <div>
          <span>
            <Database size={16} />
            数据库可视化
          </span>
          <strong>
            {selectedLibrary.milvus_db} / {selectedLibrary.kb_id}
          </strong>
        </div>
        <div className="db-visual-actions">
          <label className="db-visual-search">
            <Search size={15} />
            <input
              value={queryText}
              placeholder="过滤样本"
              onChange={(event) => setQueryText(event.target.value)}
            />
          </label>
          <button className="outline-button" onClick={() => onOpenTask(selectedLibrary.task_id)}>
            查看进度
          </button>
          <button className="primary-small" onClick={loadInspection} disabled={loading}>
            <RefreshCw size={15} />
            刷新
          </button>
        </div>
      </div>

      {error ? <div className="db-visual-error">{error}</div> : null}

      {collections ? (
        <div className="db-collection-grid">
          {(["entities", "relations", "passages"] as const).map((key) => (
            <CollectionPreview
              key={key}
              title={key === "entities" ? "实体" : key === "relations" ? "关系" : "段落"}
              data={collections[key]}
              rows={filterRows(collections[key].samples)}
            />
          ))}
        </div>
      ) : (
        <div className="db-visual-loading">{loading ? "正在读取数据库..." : "暂无可展示数据"}</div>
      )}
    </div>
  );
}

function CollectionPreview({
  title,
  data,
  rows,
}: {
  title: string;
  data: LibraryCollectionView;
  rows: Array<Record<string, unknown>>;
}) {
  return (
    <section className="collection-preview">
      <div className="collection-preview-head">
        <div>
          <strong>{title}</strong>
          <span>{data.collection}</span>
        </div>
        <b>{data.matched_count ?? data.sample_count}</b>
      </div>
      <div className="collection-samples">
        {rows.length ? (
          rows.slice(0, 5).map((row, index) => <SampleRow key={`${data.collection}-${index}`} row={row} />)
        ) : (
          <span className="empty-sample">没有匹配样本</span>
        )}
      </div>
    </section>
  );
}

function SampleRow({ row }: { row: Record<string, unknown> }) {
  const primary =
    String(row.name ?? row.relation ?? row.passage ?? row.id ?? "").slice(0, 120) || "未命名记录";
  const secondary = String(row.id ?? row.docment_id ?? "").slice(0, 80);

  return (
    <article className="sample-row">
      <strong>{primary}</strong>
      {secondary ? <span>{secondary}</span> : null}
    </article>
  );
}

function FileTable({
  files,
  buildFiles,
  onSelectionChange,
}: {
  files: UploadedDocument[];
  buildFiles: BuildFileProgress[];
  onSelectionChange: (id: string, selected: boolean) => void;
}) {
  const progressByPath = React.useMemo(
    () => new Map(buildFiles.map((item) => [item.path, item])),
    [buildFiles],
  );

  return (
    <div className="file-table">
      <div className="table-row table-head">
        <div className="cell check">选择</div>
        <div className="cell name">名称 ↕</div>
        <div className="cell date">上传日期 ↕</div>
        <div className="cell enable">启用</div>
        <div className="cell chunks">大小</div>
        <div className="cell meta">路径</div>
        <div className="cell parser">解析</div>
        <div className="cell actions">动作</div>
      </div>
      {files.map((file) => {
        const buildFile = progressByPath.get(file.path);
        const progress = Math.round(buildFile?.progress ?? 0);
        const state = buildFile
          ? buildFile.stage === "completed"
            ? `已入库：${buildFile.summary?.entities ?? 0} 实体 / ${
                buildFile.summary?.relations ?? 0
              } 关系 / ${buildFile.summary?.chunks ?? 0} 块`
            : buildFile.stage === "failed"
              ? `失败：${buildFile.state}`
              : buildFile.state
          : "未加入当前构建";
        return (
          <div className="table-row" key={file.id}>
            <div className="cell check">
              <input
                type="checkbox"
                checked={file.selected}
                onChange={(event) => onSelectionChange(file.id, event.target.checked)}
              />
            </div>
            <div className="cell name">
              <span className={`file-type ${file.type.toLowerCase()}`}>{file.type}</span>
              <span className="file-name">
                <strong>{file.name}</strong>
                <small>{state}</small>
              </span>
            </div>
            <div className="cell date">{file.uploadedAt}</div>
            <div className="cell enable">
              <span className="switch">
                <span />
              </span>
            </div>
            <div className="cell chunks">{file.size}</div>
            <div className="cell meta" title={file.path}>{file.path}</div>
            <div className="cell parser">
              <span className="parser-tag">{file.parser}</span>
            </div>
            <div className="cell actions">
              <span className="progress-track">
                <span style={{ width: `${progress}%` }} />
              </span>
              <span className={progress === 100 ? "done" : "processing"}>{state}</span>
              <MoreHorizontal size={18} />
            </div>
          </div>
        );
      })}
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
  onBackToKnowledgeList: () => void;
}) {
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

  const createItem = () => {
    const fallbackKnowledgeId = selectedKnowledge?.id ?? activeKnowledge?.id ?? "";
    const newItem: ChatItem = {
      id: `${label}-${Date.now()}`,
      title: `新${label} ${items.length + 1}`,
      knowledgeId: fallbackKnowledgeId,
    };

    onItemsChange((current) => [newItem, ...current]);
    onActiveChange(newItem.id);
  };

  const updateKnowledge = (knowledgeId: string) => {
    onItemsChange((current) =>
      current.map((item) => (item.id === activeItem.id ? { ...item, knowledgeId } : item)),
    );
  };

  const removeActive = () => {
    onItemsChange((current) => current.filter((item) => item.id !== activeItem.id));
    const nextItem = items.find((item) => item.id !== activeItem.id);
    if (nextItem) onActiveChange(nextItem.id);
  };

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
              <button className="primary-small" onClick={createItem}>
                <Plus size={16} />
                新建
              </button>
            </div>
          </header>
          <div className="message-stream">
            <article className="ai-message">
              <p>
                当前{label}已连接到「{activeKnowledge?.name ?? "暂无知识库"}」。你可以在右上角切换知识库，
                新建后也会保留独立的知识库选择。
              </p>
              <span>
                <Link2 size={14} />
                来源：产品需求.md、电影剧本写作基础.pdf
              </span>
            </article>
            <article className="user-message">提炼主角弧光的关键步骤</article>
            <div className="retrieving">
              <span />
              正在检索知识库...
            </div>
          </div>
          <ChatInput />
        </section>
      </section>
    </main>
  );
}

function ChatInput() {
  return (
    <div className="chat-input">
      <textarea placeholder="请输入消息..." />
      <div className="input-actions">
        <button aria-label="附件">
          <Paperclip size={17} />
        </button>
        <button className="thinking">Thinking</button>
        <button className="send-button" aria-label="发送">
          <Send size={17} />
        </button>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
