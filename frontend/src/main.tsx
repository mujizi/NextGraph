import React from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  BookOpen,
  Check,
  Database,
  Edit3,
  Filter,
  FolderOpen,
  FolderUp,
  House,
  Link2,
  MessageCircle,
  MoreHorizontal,
  Paperclip,
  Plus,
  Search,
  Send,
  Sparkles,
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

const files = [
  {
    checked: true,
    type: "PDF",
    name: "电影剧本写作基础与角色弧光.pdf",
    note: "等待 AI 图谱抽取",
    uploadedAt: "21/04/2026 09:59:20",
    chunks: "1354",
    metadata: "0 fields",
    parser: "Book",
    progress: 100,
    state: "完成",
  },
  {
    checked: false,
    type: "DOC",
    name: "第一章电影剧本结构.docx",
    note: "文本解析中",
    uploadedAt: "21/04/2026 10:12:03",
    chunks: "842",
    metadata: "3 fields",
    parser: "Text",
    progress: 63,
    state: "63%",
  },
  {
    checked: false,
    type: "MD",
    name: "角色动机与冲突设计.md",
    note: "等待中",
    uploadedAt: "21/04/2026 10:40:11",
    chunks: "-",
    metadata: "0 fields",
    parser: "Markdown",
    progress: 0,
    state: "等待中",
  },
];

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
          onCreateKnowledge={() => setCreatingKnowledge(true)}
          onOpenKnowledge={openKnowledgeBase}
          onRenameKnowledge={renameKnowledgeBase}
        />
      )}
      {page === "knowledge" && (
        <KnowledgePage
          knowledgeBases={knowledgeBases}
          selectedKnowledgeId={selectedKnowledgeId}
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
  selectedKnowledgeId,
  onCreateKnowledge,
  onOpenKnowledge,
  onRenameKnowledge,
  onBackToList,
}: {
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeId: string | null;
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
            <button className="soft-button">
              <Sparkles size={16} />
              AI 抽取图
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
            <button className="square-primary" aria-label="新增文件">
              <Upload size={17} />
            </button>
          </div>
        </div>

        <div className="batch-bar">
          <strong>已选择 8 个文件</strong>
          <button>解析文件</button>
          <button className="active">AI 抽取图</button>
          <span>启用</span>
          <span>停用</span>
          <span className="danger">删除</span>
        </div>

        <FileTable />
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

function FileTable() {
  return (
    <div className="file-table">
      <div className="table-row table-head">
        <div className="cell check">□</div>
        <div className="cell name">名称 ↕</div>
        <div className="cell date">上传日期 ↕</div>
        <div className="cell enable">启用</div>
        <div className="cell chunks">分块数</div>
        <div className="cell meta">元数据</div>
        <div className="cell parser">解析</div>
        <div className="cell actions">动作</div>
      </div>
      {files.map((file) => (
        <div className="table-row" key={file.name}>
          <div className="cell check">{file.checked ? "☑" : "□"}</div>
          <div className="cell name">
            <span className={`file-type ${file.type.toLowerCase()}`}>{file.type}</span>
            <span className="file-name">
              <strong>{file.name}</strong>
              <small>{file.note}</small>
            </span>
          </div>
          <div className="cell date">{file.uploadedAt}</div>
          <div className="cell enable">
            <span className="switch">
              <span />
            </span>
          </div>
          <div className="cell chunks">{file.chunks}</div>
          <div className="cell meta">{file.metadata}</div>
          <div className="cell parser">
            <span className="parser-tag">{file.parser}</span>
          </div>
          <div className="cell actions">
            <span className="progress-track">
              <span style={{ width: `${file.progress}%` }} />
            </span>
            <span className={file.progress === 100 ? "done" : "processing"}>{file.state}</span>
            <MoreHorizontal size={18} />
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
