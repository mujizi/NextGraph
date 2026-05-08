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

// 🌟 后端地址配置
const BACKEND_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

// 🌟 文件类型徽标颜色映射字典
export const getFileBadgeStyle = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase() || "unknown";
  const styles: Record<string, { bg: string, color: string }> = {
    pdf: { bg: '#ffebee', color: '#f44336' },
    doc: { bg: '#e3f2fd', color: '#2196f3' },
    docx: { bg: '#e3f2fd', color: '#2196f3' },
    md: { bg: '#e8f5e9', color: '#4caf50' },
    json: { bg: '#fff3e0', color: '#ff9800' },
    txt: { bg: '#f3e5f5', color: '#9c27b0' },
    csv: { bg: '#e0f7fa', color: '#00bcd4' },
    xlsx: { bg: '#e8f5e9', color: '#4caf50' },
  };
  return { ext: ext.toUpperCase(), style: styles[ext] || { bg: '#f5f5f5', color: '#607d8b' } };
};

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

  // ================= 🌟 状态管理 =================
  const [pendingFiles, setPendingFiles] = React.useState<File[]>([]);
  const [pendingPaths, setPendingPaths] = React.useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = React.useState("");

  const [showConflictModal, setShowConflictModal] = React.useState(false);
  const [conflictQueue, setConflictQueue] = React.useState<string[]>([]);
  const [currentConflictIndex, setCurrentConflictIndex] = React.useState(0);
  const [applyToAll, setApplyToAll] = React.useState(false);
  const [resolutions, setResolutions] = React.useState<Record<string, "replace" | "rename" | "skip">>({});

  const [serverFiles, setServerFiles] = React.useState<any[]>([]);
  const [selectedForDelete, setSelectedForDelete] = React.useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = React.useState(false);

  const [showDeleteModal, setShowDeleteModal] = React.useState(false);

  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const fetchServerFiles = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/delete/list`);
      const data = await res.json();
      if (data.code === 200) {
        setServerFiles(data.files);
        setSelectedForDelete(new Set()); 
      }
    } catch (err) {
      console.error("获取服务器文件列表失败", err);
    }
  };

  React.useEffect(() => {
    if (page === "knowledge" && selectedKnowledgeId) fetchServerFiles();
  }, [page, selectedKnowledgeId]);

  const startUpload = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    type WebkitFile = File & { webkitRelativePath?: string };
    const filesArray = Array.from(fileList) as WebkitFile[];
    const pathsArray = filesArray.map((f) => f.webkitRelativePath || f.name);

    setUploadStatus("正在预检冲突...");
    try {
      const res = await fetch(`${BACKEND_URL}/api/upload/check_conflicts`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paths: pathsArray }),
      });
      if (!res.ok) throw new Error(`check_conflicts failed: ${res.status}`);
      const data = await res.json();

      if (data.conflicts && data.conflicts.length > 0) {
        setConflictQueue(data.conflicts);
        setCurrentConflictIndex(0);
        setApplyToAll(false);
        setResolutions({});
        setPendingFiles(filesArray);
        setPendingPaths(pathsArray);
        setShowConflictModal(true);
      } else {
        executeUpload(filesArray, pathsArray);
      }
    } catch (e) {
      alert("后端服务未响应，请检查 backend/app/main.py 是否运行");
      setUploadStatus("");
    }
  };

  const handleResolve = (action: "replace" | "rename" | "skip") => {
    if (applyToAll) {
      const newRes = { ...resolutions };
      for (let i = currentConflictIndex; i < conflictQueue.length; i++) newRes[conflictQueue[i]] = action;
      setShowConflictModal(false);
      processFinalUpload(newRes);
    } else {
      const currentPath = conflictQueue[currentConflictIndex];
      const newRes = { ...resolutions, [currentPath]: action };
      if (currentConflictIndex + 1 < conflictQueue.length) {
        setResolutions(newRes);
        setCurrentConflictIndex(currentConflictIndex + 1);
      } else {
        setShowConflictModal(false);
        processFinalUpload(newRes);
      }
    }
  };

  const cancelUpload = () => {
    setShowConflictModal(false);
    setConflictQueue([]);
    setPendingFiles([]);
    setPendingPaths([]);
    setUploadStatus("已取消上传");
    setTimeout(() => setUploadStatus(""), 3000);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const processFinalUpload = (finalResolutions: Record<string, "replace" | "rename" | "skip">) => {
    let finalFiles: File[] = [];
    let finalPaths: string[] = [];
    pendingFiles.forEach((file, i) => {
      const path = pendingPaths[i];
      const decision = finalResolutions[path];
      if (decision === "skip") return; 
      if (decision === "rename") {
        const dotIdx = path.lastIndexOf('.');
        const randomSuffix = Math.floor(Math.random() * 10000);
        const newPath = dotIdx > 0 ? `${path.substring(0, dotIdx)}(1)_${randomSuffix}${path.substring(dotIdx)}` : `${path}(1)_${randomSuffix}`;
        finalFiles.push(file);
        finalPaths.push(newPath);
      } else {
        finalFiles.push(file);
        finalPaths.push(path);
      }
    });
    if (finalFiles.length > 0) executeUpload(finalFiles, finalPaths);
    else { setUploadStatus("所有冲突已被跳过，无新文件上传"); setTimeout(() => setUploadStatus(""), 3000); }
  };

  const executeUpload = async (files: File[], paths: string[]) => {
    setUploadStatus("文件上传中...");
    const formData = new FormData();
    files.forEach((f, i) => { formData.append("files", f); formData.append("target_paths", paths[i]); });
    try {
      const res = await fetch(`${BACKEND_URL}/api/upload/do_upload`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`do_upload failed: ${res.status}`);
      setUploadStatus("上传成功"); setTimeout(() => setUploadStatus(""), 3000);
      fetchServerFiles(); 
    } catch (e) {
      setUploadStatus("上传失败");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (folderInputRef.current) folderInputRef.current.value = "";
    }
  };

  // 共享的文件夹分组逻辑
  const groupedFiles = React.useMemo(() => {
    const groups: Record<string, any[]> = {};
    serverFiles.forEach(file => {
      const parts = file.path.split('/');
      const folder = parts.length > 1 ? parts.slice(0, -1).join('/') : "根目录";
      if (!groups[folder]) groups[folder] = [];
      groups[folder].push(file);
    });
    return groups;
  }, [serverFiles]);

  const confirmDelete = async () => {
    if (selectedForDelete.size === 0) return;
    setIsDeleting(true);
    const pathsToDelete = new Set(selectedForDelete);
    Object.entries(groupedFiles).forEach(([folder, filesInFolder]) => {
      if (folder !== "根目录" && filesInFolder.length > 0 && filesInFolder.every((f: any) => selectedForDelete.has(f.path))) {
        pathsToDelete.add(folder); 
      }
    });

    try {
      const res = await fetch(`${BACKEND_URL}/api/delete/batch`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paths: Array.from(pathsToDelete) }),
      });
      if (res.ok) {
        alert("删除成功！");
        setShowDeleteModal(false);
        fetchServerFiles();
      } else { alert("部分删除失败"); }
    } catch (e) { alert("删除请求出错"); } finally { setIsDeleting(false); }
  };

  const openPage = (nextPage: Page) => {
    if (nextPage === "knowledge") setSelectedKnowledgeId(null);
    if (nextPage === "chat") setSelectedChatKnowledgeId(null);
    if (nextPage === "search") setSelectedSearchKnowledgeId(null);
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
          onUploadFolder={() => folderInputRef.current?.click()}
          onUploadFile={() => fileInputRef.current?.click()}
          onOpenDeleteModal={() => setShowDeleteModal(true)}
          statusMsg={uploadStatus}
          serverFiles={serverFiles}
          selectedForDelete={selectedForDelete}
          setSelectedForDelete={setSelectedForDelete}
          onConfirmDelete={confirmDelete}
          isDeleting={isDeleting}
          groupedFiles={groupedFiles}
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

      {/* ================= 全局组件与弹窗 ================= */}
      <input type="file" ref={folderInputRef} hidden /* @ts-ignore */ webkitdirectory="" directory="" multiple onChange={(e) => startUpload(e.target.files)} />
      <input type="file" ref={fileInputRef} hidden multiple onChange={(e) => startUpload(e.target.files)} />

      {showConflictModal && conflictQueue.length > 0 && (
        <div className="modal-backdrop" role="presentation" style={{ zIndex: 9999 }}>
          <div className="modal-card" style={{ maxWidth: '420px' }}>
            <div className="modal-head" style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
              <h2 style={{ fontSize: '16px', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={18} color="#f39c12" /> 上传冲突 ({currentConflictIndex + 1}/{conflictQueue.length})
              </h2>
              <button type="button" aria-label="关闭" onClick={cancelUpload}><X size={17} /></button>
            </div>
            <div style={{ padding: '20px' }}>
              <p style={{ fontSize: '14px', color: '#333', marginBottom: '10px', wordBreak: 'break-all' }}>目标位置已存在同名文件：<br/><strong style={{ color: '#e74c3c' }}>{conflictQueue[currentConflictIndex]}</strong></p>
              {conflictQueue.length > 1 && currentConflictIndex < conflictQueue.length - 1 && (
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#555', marginTop: '15px', cursor: 'pointer', background: '#f9f9f9', padding: '8px', borderRadius: '4px' }}>
                  <input type="checkbox" checked={applyToAll} onChange={(e) => setApplyToAll(e.target.checked)} />
                  为剩余的 <strong>{conflictQueue.length - currentConflictIndex - 1}</strong> 个冲突执行相同操作
                </label>
              )}
            </div>
            <div className="modal-actions" style={{ flexDirection: 'column', gap: '8px', padding: '0 20px 20px 20px' }}>
              <button className="primary-small" style={{ width: '100%', background: '#3498db' }} onClick={() => handleResolve('rename')}>保留两者 (自动重命名)</button>
              <button className="primary-small" style={{ width: '100%', background: '#e74c3c' }} onClick={() => handleResolve('replace')}>直接替换</button>
              <button className="outline-button" style={{ width: '100%' }} onClick={() => handleResolve('skip')}>跳过此文件</button>
              <button onClick={cancelUpload} style={{ width: '100%', background: 'transparent', border: 'none', color: '#999', fontSize: '13px', marginTop: '5px', cursor: 'pointer', textDecoration: 'underline' }}>停止本次全部上传</button>
            </div>
          </div>
        </div>
      )}

      {/* 🌟 独立的带折叠功能的弹窗 */}
      {showDeleteModal && (
        <div className="modal-backdrop" role="presentation" style={{ zIndex: 9999 }}>
          <div className="modal-card" style={{ width: '650px', maxWidth: '95%', maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}>
            <div className="modal-head" style={{ padding: '20px', borderBottom: '1px solid #eee' }}>
              <h2 style={{ fontSize: '16px', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Database size={18} /> 服务器文件管理
              </h2>
              <button type="button" aria-label="关闭" onClick={() => setShowDeleteModal(false)}><X size={17} /></button>
            </div>
            
            <div style={{ padding: '0', overflowY: 'auto', flex: 1, backgroundColor: '#fff' }}>
              <ModalFileTable 
                serverFiles={serverFiles} 
                groupedFiles={groupedFiles}
                selectedForDelete={selectedForDelete} 
                setSelectedForDelete={setSelectedForDelete} 
              />
            </div>

            <div style={{ padding: '15px 20px', borderTop: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fafafa', borderRadius: '0 0 8px 8px' }}>
              <span style={{ fontSize: '13px', color: '#666' }}>已选择 <strong>{selectedForDelete.size}</strong> 个文件</span>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button type="button" className="outline-button" onClick={() => setShowDeleteModal(false)}>取消</button>
                <button 
                  type="button" className="primary-small" onClick={confirmDelete} disabled={selectedForDelete.size === 0 || isDeleting}
                  style={{ backgroundColor: selectedForDelete.size > 0 ? '#e74c3c' : '#ccc', cursor: selectedForDelete.size > 0 ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  <Trash2 size={14} /> {isDeleting ? "删除中..." : "确认删除"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// === 新增：专用于弹窗内渲染折叠文件的子组件 ===
function ModalFileTable({ serverFiles, groupedFiles, selectedForDelete, setSelectedForDelete }: any) {
  const [expandedFolders, setExpandedFolders] = React.useState<Set<string>>(new Set());

  React.useEffect(() => {
    if (serverFiles) {
      const folders = new Set<string>();
      serverFiles.forEach((f: any) => {
        const parts = f.path.split('/');
        if (parts.length > 1) folders.add(parts.slice(0, -1).join('/'));
      });
      setExpandedFolders(folders);
    }
  }, [serverFiles]);

  if (!serverFiles || serverFiles.length === 0) {
    return <p style={{ color: '#999', textAlign: 'center', padding: '40px 0' }}>服务器上暂无文件</p>;
  }

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
      <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
        <tr>
          <th style={{ padding: '12px 16px', width: '40px' }}>
            <input 
              type="checkbox" 
              checked={selectedForDelete.size === serverFiles.length && serverFiles.length > 0}
              onChange={() => {
                if (selectedForDelete.size === serverFiles.length) setSelectedForDelete(new Set());
                else setSelectedForDelete(new Set(serverFiles.map((f:any) => f.path)));
              }}
            />
          </th>
          <th style={{ padding: '12px 8px' }}>目录 / 文件名</th>
          <th style={{ padding: '12px 8px', color: '#666', width: '25%' }}>服务器路径</th>
          <th style={{ padding: '12px 16px', color: '#666', width: '80px', textAlign: 'right' }}>大小</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(groupedFiles).map(([folder, filesInFolder]: [string, any]) => {
          const isExpanded = expandedFolders.has(folder);
          const allSelected = filesInFolder.length > 0 && filesInFolder.every((f: any) => selectedForDelete.has(f.path));
          const someSelected = filesInFolder.some((f: any) => selectedForDelete.has(f.path));

          return (
            <React.Fragment key={folder}>
              <tr style={{ backgroundColor: '#f8fafd', borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '10px 16px' }}>
                  <input 
                    type="checkbox" 
                    checked={allSelected}
                    ref={el => { if (el) el.indeterminate = someSelected && !allSelected; }}
                    onChange={() => {
                      const newSet = new Set(selectedForDelete);
                      if (allSelected) filesInFolder.forEach((f: any) => newSet.delete(f.path)); 
                      else filesInFolder.forEach((f: any) => newSet.add(f.path)); 
                      setSelectedForDelete(newSet);
                    }}
                  />
                </td>
                <td colSpan={3} style={{ padding: '10px 16px', paddingLeft: '8px', cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => {
                      const newExp = new Set(expandedFolders);
                      if (isExpanded) newExp.delete(folder); else newExp.add(folder);
                      setExpandedFolders(newExp);
                    }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: '#888', fontSize: '10px', display: 'inline-block', width: '12px', textAlign: 'center' }}>
                      {isExpanded ? '▼' : '▶'}
                    </span>
                    <FolderOpen size={16} color="#3498db" fill={isExpanded ? "#eaf2f8" : "transparent"} />
                    <strong style={{ color: '#2c3e50' }}>{folder}</strong>
                    <span style={{ color: '#95a5a6', fontSize: '12px', fontWeight: 'normal' }}>({filesInFolder.length} 个文件)</span>
                  </div>
                </td>
              </tr>
              
              {isExpanded && filesInFolder.map((file: any) => {
                const { ext, style } = getFileBadgeStyle(file.filename);
                return (
                  <tr key={file.path} style={{ borderBottom: '1px solid #f5f5f5', backgroundColor: selectedForDelete.has(file.path) ? '#f4f9ff' : 'transparent' }}>
                    <td style={{ padding: '10px 16px', paddingLeft: '38px' }}>
                      <input 
                        type="checkbox" 
                        checked={selectedForDelete.has(file.path)}
                        onChange={() => {
                          const newSet = new Set(selectedForDelete);
                          if (newSet.has(file.path)) newSet.delete(file.path);
                          else newSet.add(file.path);
                          setSelectedForDelete(newSet);
                        }}
                      />
                    </td>
                    <td style={{ padding: '10px 8px', fontWeight: '500', display: 'flex', alignItems: 'center' }}>
                      <span style={{ display: 'inline-block', padding: '2px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 'bold', marginRight: '8px', backgroundColor: style.bg, color: style.color }}>
                        {ext}
                      </span>
                      {file.filename}
                    </td>
                    <td style={{ padding: '10px 8px', color: '#888', fontSize: '12px', wordBreak: 'break-all' }}>{file.path}</td>
                    <td style={{ padding: '10px 16px', color: '#888', fontSize: '12px', textAlign: 'right' }}>{file.size_kb} KB</td>
                  </tr>
                )
              })}
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
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
  onUploadFolder,
  onUploadFile,
  onOpenDeleteModal, // 🌟 接收并绑定这个函数
  statusMsg,
  serverFiles,
  selectedForDelete,
  setSelectedForDelete,
  onConfirmDelete,
  isDeleting,
  groupedFiles
}: any) {
  const selectedKnowledge = knowledgeBases.find((knowledge: any) => knowledge.id === selectedKnowledgeId);

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
            {statusMsg && <span style={{ fontSize: '13px', color: '#3498db', fontWeight: '500' }}>{statusMsg}</span>}
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
            <button className="outline-button" onClick={onUploadFolder}>
              <FolderUp size={16} />
              文件夹
            </button>
            <button className="square-primary" aria-label="新增文件" onClick={onUploadFile}>
              <Upload size={17} />
            </button>
            
            {/* ================= 🌟 终于找回来的弹窗按钮 ================= */}
            <button className="outline-button" onClick={onOpenDeleteModal} style={{ color: '#e74c3c', borderColor: '#fadbd8', marginLeft: '8px' }}>
              <Trash2 size={16} />管理文件
            </button>
          </div>
        </div>

        <div className="batch-bar" style={{ opacity: selectedForDelete.size > 0 ? 1 : 0, pointerEvents: selectedForDelete.size > 0 ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
          <strong>已选择 {selectedForDelete.size} 个文件</strong>
          <button>解析文件</button>
          <button className="active">AI 抽取图</button>
          <span>启用</span>
          <span>停用</span>
          <span className="danger" onClick={onConfirmDelete} style={{ cursor: isDeleting ? 'wait' : 'pointer', opacity: isDeleting ? 0.5 : 1 }}>
            {isDeleting ? "删除中..." : "删除"}
          </span>
        </div>

        <FileTable 
          serverFiles={serverFiles} 
          groupedFiles={groupedFiles}
          selectedForDelete={selectedForDelete} 
          setSelectedForDelete={setSelectedForDelete} 
        />
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

// 主表格 (带折叠与彩标)
function FileTable({ serverFiles, groupedFiles, selectedForDelete, setSelectedForDelete }: any) {
  const [expandedFolders, setExpandedFolders] = React.useState<Set<string>>(new Set());

  React.useEffect(() => {
    if (serverFiles) {
      const folders = new Set<string>();
      serverFiles.forEach((f: any) => {
        const parts = f.path.split('/');
        if (parts.length > 1) folders.add(parts.slice(0, -1).join('/'));
      });
      setExpandedFolders(folders);
    }
  }, [serverFiles]);

  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) setSelectedForDelete(new Set(serverFiles.map((f: any) => f.path)));
    else setSelectedForDelete(new Set());
  };

  const toggleSelect = (path: string) => {
    const newSet = new Set(selectedForDelete);
    if (newSet.has(path)) newSet.delete(path);
    else newSet.add(path);
    setSelectedForDelete(newSet);
  };

  return (
    <div className="file-table">
      <div className="table-row table-head">
        <div className="cell check">
          <input type="checkbox" checked={serverFiles?.length > 0 && selectedForDelete.size === serverFiles.length} onChange={handleSelectAll} />
        </div>
        <div className="cell name">目录 / 文件名 ↕</div>
        <div className="cell date">大小 ↕</div>
        <div className="cell enable">启用</div>
        <div className="cell chunks">分块数</div>
        <div className="cell meta">元数据</div>
        <div className="cell parser">解析</div>
        <div className="cell actions">动作</div>
      </div>

      {!serverFiles || serverFiles.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>暂无文件，请在上方上传</div>
      ) : (
        Object.entries(groupedFiles).map(([folder, filesInFolder]: [string, any]) => {
          const isExpanded = expandedFolders.has(folder);
          const allSelected = filesInFolder.length > 0 && filesInFolder.every((f: any) => selectedForDelete.has(f.path));
          const someSelected = filesInFolder.some((f: any) => selectedForDelete.has(f.path));

          return (
            <React.Fragment key={folder}>
              <div className="table-row" style={{ backgroundColor: '#f8fafd', borderBottom: '1px solid #eee' }}>
                <div className="cell check">
                  <input 
                    type="checkbox" 
                    checked={allSelected}
                    ref={el => { if (el) el.indeterminate = someSelected && !allSelected; }}
                    onChange={() => {
                      const newSet = new Set(selectedForDelete);
                      if (allSelected) filesInFolder.forEach((f: any) => newSet.delete(f.path)); 
                      else filesInFolder.forEach((f: any) => newSet.add(f.path)); 
                      setSelectedForDelete(newSet);
                    }}
                  />
                </div>
                <div 
                  className="cell name" 
                  style={{ cursor: 'pointer', userSelect: 'none', display: 'flex', alignItems: 'center', gap: '8px' }}
                  onClick={() => {
                    const newExp = new Set(expandedFolders);
                    if (isExpanded) newExp.delete(folder); else newExp.add(folder);
                    setExpandedFolders(newExp);
                  }}
                >
                  <span style={{ color: '#888', fontSize: '10px', display: 'inline-block', width: '12px', textAlign: 'center' }}>
                    {isExpanded ? '▼' : '▶'}
                  </span>
                  <FolderOpen size={16} color="#3498db" fill={isExpanded ? "#eaf2f8" : "transparent"} />
                  <strong style={{ color: '#2c3e50' }}>{folder}</strong>
                  <span style={{ color: '#95a5a6', fontSize: '12px', fontWeight: 'normal' }}>({filesInFolder.length} 个文件)</span>
                </div>
                <div className="cell date"></div>
                <div className="cell enable"></div>
                <div className="cell chunks"></div>
                <div className="cell meta"></div>
                <div className="cell parser"></div>
                <div className="cell actions"></div>
              </div>

              {isExpanded && filesInFolder.map((file: any) => {
                const { ext, style } = getFileBadgeStyle(file.filename);
                const isSelected = selectedForDelete.has(file.path);
                
                return (
                  <div className="table-row" key={file.path} style={{ backgroundColor: isSelected ? '#f4f9ff' : '' }}>
                    <div className="cell check" style={{ paddingLeft: '32px' }}>
                      <input type="checkbox" checked={isSelected} onChange={() => toggleSelect(file.path)} />
                    </div>
                    <div className="cell name" style={{ paddingLeft: '24px' }}>
                      <span className="file-type" style={{ backgroundColor: style.bg, color: style.color, border: 'none', fontWeight: 'bold' }}>
                        {ext}
                      </span>
                      <span className="file-name">
                        <strong>{file.filename}</strong>
                        <small>{file.path}</small>
                      </span>
                    </div>
                    <div className="cell date">{file.size_kb} KB</div>
                    <div className="cell enable"><span className="switch"><span /></span></div>
                    <div className="cell chunks">-</div>
                    <div className="cell meta">0 fields</div>
                    <div className="cell parser"><span className="parser-tag">Auto</span></div>
                    <div className="cell actions">
                      <span className="progress-track"><span style={{ width: `100%` }} /></span>
                      <span className="done">已同步</span>
                      <MoreHorizontal size={18} />
                    </div>
                  </div>
                );
              })}
            </React.Fragment>
          );
        })
      )}
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

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}