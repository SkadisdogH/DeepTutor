"use client";

/**
 * SideChatPanel — 右侧可展开的「临时提问」抽屉。
 *
 * 与主对话并存的临时问答区:用户在右侧抽屉里问「主对话里某段是什么意思」,
 * 每次提问自动把当前主会话作为 history_references 带上,模型通过引用清单
 * 按需读取主对话全文;抽屉自己的问答存成一个真实会话,但会话带
 * ``preferences.temporary`` 标记,不会出现在左侧历史列表里(后端过滤),
 * 面板内通过自己的小列表管理,可随时「新开」。
 *
 * 实现要点(与 QuizFollowup 相同的独立 runner 模式):
 *  - 面板常驻挂载(off-screen translate-x-full),关闭只是视觉隐藏——流式
 *    进行中关掉面板,事件照常处理,重开即见最新,无需断线重连恢复。
 *  - 用独立 UnifiedWSClient 走 /api/v1/ws 发 start_turn(capability=chat),
 *    与主对话的 WS 互不干扰。
 *  - 会话 id 与消息缓存在 localStorage(按主会话隔离),重开页面后按 id
 *    GET 会话详情恢复;404(会话已被删除)则静默重置为全新对话。
 *  - 「加入对话」复用主页面已有的 ``dt:add-to-conversation`` 窗口事件,
 *    把回答预填进主对话输入框。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  MessageCircleQuestion,
  Plus,
  Send,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import MarkdownRenderer from "@/components/common/MarkdownRenderer";
import OverlayScrollbar from "@/components/chat/home/OverlayScrollbar";
import { deleteSession, getSession } from "@/lib/session-api";
import { shouldAppendEventContent } from "@/lib/stream";
import {
  type ChatMessage,
  type StreamEvent,
  UnifiedWSClient,
} from "@/lib/unified-ws";

interface SideChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface SideChatPanelProps {
  open: boolean;
  onClose: () => void;
  /** 当前主对话 session id —— 提问时作为引用带上,让模型能读到主对话全文。 */
  mainSessionId: string | null;
  /** 主对话标题(仅用于面板内的提示文案)。 */
  mainSessionTitle: string;
}

const ANIM_MS = 220;
const STORAGE_PREFIX = "dt:sidechat:";
const MAX_CONNECT_ATTEMPTS = 10;
const CONNECT_DELAY_MS = 200;

function readStash(storageKey: string | null): {
  sessionId: string | null;
  messages: SideChatMessage[];
} {
  if (!storageKey) return { sessionId: null, messages: [] };
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return { sessionId: null, messages: [] };
    const parsed = JSON.parse(raw) as {
      sessionId?: unknown;
      messages?: unknown;
    };
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.filter(
          (m): m is SideChatMessage =>
            !!m &&
            (m.role === "user" || m.role === "assistant") &&
            typeof m.content === "string",
        )
      : [];
    return {
      sessionId:
        typeof parsed.sessionId === "string" && parsed.sessionId.length > 0
          ? parsed.sessionId
          : null,
      messages,
    };
  } catch {
    return { sessionId: null, messages: [] };
  }
}

export default function SideChatPanel({
  open,
  onClose,
  mainSessionId,
  mainSessionTitle,
}: SideChatPanelProps) {
  const { t, i18n } = useTranslation();

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<SideChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [addedIndex, setAddedIndex] = useState<number | null>(null);

  const runnerRef = useRef<{ client: UnifiedWSClient } | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);
  const atBottomRef = useRef(true);
  const confirmClearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const addedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const messagesRef = useRef<SideChatMessage[]>([]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // 清理挂起的确认/反馈定时器。
  useEffect(
    () => () => {
      if (confirmClearTimer.current) clearTimeout(confirmClearTimer.current);
      if (addedTimer.current) clearTimeout(addedTimer.current);
    },
    [],
  );

  const storageKey = useMemo(
    () => (mainSessionId ? `${STORAGE_PREFIX}${mainSessionId}` : null),
    [mainSessionId],
  );

  const persistStash = useCallback(() => {
    if (!storageKey) return;
    try {
      window.localStorage.setItem(
        storageKey,
        JSON.stringify({
          sessionId: sessionIdRef.current,
          messages: messagesRef.current,
        }),
      );
    } catch {
      /* quota / privacy mode — cache is best-effort only */
    }
  }, [storageKey]);

  const clearStash = useCallback(() => {
    if (!storageKey) return;
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      /* ignore */
    }
  }, [storageKey]);

  // 切换主会话 → 重置面板(不同主对话的临时问答互相独立)。
  useEffect(() => {
    setSessionId(null);
    setMessages([]);
    setInput("");
    setIsStreaming(false);
    setError(null);
    setConfirmClear(false);
    runnerRef.current?.client.disconnect();
    runnerRef.current = null;
  }, [mainSessionId]);

  // 打开时恢复:先上缓存(秒开),后台按 id 拉服务端权威消息刷新。
  useEffect(() => {
    if (!open) return;
    const stash = readStash(storageKey);
    setSessionId(stash.sessionId);
    setMessages(stash.messages);
    setError(null);
    setConfirmClear(false);
    setAddedIndex(null);
    if (stash.sessionId) {
      getSession(stash.sessionId)
        .then((detail) => {
          const msgs: SideChatMessage[] = (detail.messages ?? [])
            .filter(
              (m) => m.role === "user" || m.role === "assistant",
            )
            .map((m) => ({
              role: m.role as "user" | "assistant",
              content: m.content,
            }));
          setMessages(msgs);
        })
        .catch(() => {
          // 404 / 服务不可达:会话没了就静默开新对话,不打断用户。
          setSessionId(null);
          setMessages([]);
          clearStash();
        });
    }
  }, [open, storageKey, clearStash]);

  // ESC 关闭(仅面板可见时)。
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (open) closeBtnRef.current?.focus();
  }, [open]);

  // 流式时跟随底部;用户向上翻看时暂停跟随。
  useEffect(() => {
    if (atBottomRef.current) {
      endRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages]);

  const handleMessagesScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    atBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  }, []);

  const handleEvent = useCallback((event: StreamEvent) => {
    if (event.type === "session") {
      const meta = (event.metadata ?? {}) as { session_id?: string };
      const next = meta.session_id || event.session_id || "";
      if (next) {
        // 同步维护 ref,确保 done/error 落盘时能读到最新会话 id。
        sessionIdRef.current = next;
        setSessionId(next);
        setError(null);
      }
      return;
    }
    if (event.type === "done") {
      setIsStreaming(false);
      persistStash();
      runnerRef.current?.client.disconnect();
      runnerRef.current = null;
      return;
    }
    if (event.type === "error") {
      const terminal = Boolean(
        ((event.metadata ?? {}) as { turn_terminal?: boolean }).turn_terminal,
      );
      setError(event.content || t("Something went wrong."));
      if (terminal) {
        setIsStreaming(false);
        persistStash();
        runnerRef.current?.client.disconnect();
        runnerRef.current = null;
      }
      return;
    }
    if (event.type === "stage_start") {
      setError(null);
    }
    if (shouldAppendEventContent(event)) {
      const msgs = [...messagesRef.current];
      const last = msgs[msgs.length - 1];
      if (!last || last.role !== "assistant") {
        msgs.push({ role: "assistant", content: event.content });
      } else {
        msgs[msgs.length - 1] = {
          ...last,
          content: last.content + event.content,
        };
      }
      messagesRef.current = msgs;
      setMessages(msgs);
    }
  }, [persistStash]);

  const ensureRunner = useCallback(() => {
    const existing = runnerRef.current;
    if (existing) {
      if (!existing.client.connected) existing.client.connect();
      return existing.client;
    }
    const client = new UnifiedWSClient(
      handleEvent,
      () => {
        if (sessionIdRef.current && messagesRef.current.length > 0) {
          persistStash();
        }
      },
    );
    runnerRef.current = { client };
    client.connect();
    return client;
  }, [handleEvent, persistStash]);

  const sendThroughRunner = useCallback(
    function send(message: ChatMessage, attempt = 0) {
      const client = ensureRunner();
      if (!client.connected) {
        if (attempt >= MAX_CONNECT_ATTEMPTS) {
          setIsStreaming(false);
          setError(
            t("Couldn't reach the server. Please check your connection and retry."),
          );
          return;
        }
        window.setTimeout(
          () => send(message, attempt + 1),
          CONNECT_DELAY_MS,
        );
        return;
      }
      client.send(message);
    },
    [ensureRunner, t],
  );

  const language = i18n.language?.toLowerCase().startsWith("zh")
    ? "zh"
    : "en";

  const handleSend = useCallback(() => {
    const content = input.trim();
    if (!content || isStreaming) return;
    setInput("");
    setError(null);
    setIsStreaming(true);
    const nextMsgs: SideChatMessage[] = [
      ...messagesRef.current,
      { role: "user", content },
    ];
    messagesRef.current = nextMsgs;
    setMessages(nextMsgs);
    sendThroughRunner({
      type: "start_turn",
      content,
      tools: [],
      capability: "chat",
      knowledge_bases: [],
      session_id: sessionIdRef.current,
      attachments: [],
      language,
      config: {},
      history_references: mainSessionId ? [mainSessionId] : undefined,
      temporary: true,
    });
  }, [input, isStreaming, language, mainSessionId, sendThroughRunner]);

  const cancelStreaming = useCallback(() => {
    runnerRef.current?.client.disconnect();
    runnerRef.current = null;
    setIsStreaming(false);
    persistStash();
  }, [persistStash]);

  // 「新开」:第一次点击进入确认态,3 秒内再点才真正清空;期间点别处自动还原。
  const handleNewScratch = useCallback(() => {
    if (!confirmClear) {
      setConfirmClear(true);
      if (confirmClearTimer.current) clearTimeout(confirmClearTimer.current);
      confirmClearTimer.current = setTimeout(
        () => setConfirmClear(false),
        3000,
      );
      return;
    }
    const oldSessionId = sessionIdRef.current;
    setSessionId(null);
    setMessages([]);
    setInput("");
    setIsStreaming(false);
    setError(null);
    setConfirmClear(false);
    clearStash();
    runnerRef.current?.client.disconnect();
    runnerRef.current = null;
    if (oldSessionId) {
      void deleteSession(oldSessionId).catch(() => {});
    }
  }, [confirmClear, clearStash]);

  const handleAddToMain = useCallback((content: string, index: number) => {
    window.dispatchEvent(
      new CustomEvent("dt:add-to-conversation", { detail: content }),
    );
    setAddedIndex(index);
    if (addedTimer.current) clearTimeout(addedTimer.current);
    addedTimer.current = setTimeout(() => setAddedIndex(null), 2000);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const visible = open;
  const hasNoMessages = messages.length === 0;
  const clearing = confirmClear;

  return (
    <div
      role="dialog"
      aria-hidden={!visible}
      aria-label={t("Temporary questions")}
      className={`fixed right-0 top-0 z-[30] flex h-dvh w-full flex-col border-l border-[var(--border)] bg-[var(--card)] transition-transform ease-out md:w-[min(400px,92vw)] ${
        visible ? "translate-x-0 shadow-2xl" : "translate-x-full"
      }`}
      style={{
        willChange: "transform",
        transitionDuration: `${ANIM_MS}ms`,
        pointerEvents: visible ? "auto" : "none",
      }}
    >
      {/* 面板头 */}
      <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--card)] px-4 py-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[var(--muted)]/60 text-[var(--primary)]">
          <MessageCircleQuestion size={18} strokeWidth={1.5} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium text-[var(--foreground)]">
            {t("Temporary questions")}
          </div>
          <div className="truncate text-[10px] text-[var(--muted-foreground)]">
            {mainSessionTitle
              ? t("References: {{title}}", { title: mainSessionTitle })
              : t("References the current conversation")}
          </div>
        </div>
        <button
          type="button"
          onClick={handleNewScratch}
          title={clearing ? t("Click again to confirm") : t("Start a new scratch conversation")}
          aria-label={clearing ? t("Click again to confirm") : t("New scratch conversation")}
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors hover:bg-[var(--muted)] ${
            clearing
              ? "text-[var(--destructive)]"
              : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
          }`}
        >
          <Trash2 size={15} strokeWidth={1.8} />
        </button>
        <button
          ref={closeBtnRef}
          type="button"
          onClick={onClose}
          title={t("Close")}
          aria-label={t("Close")}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
        >
          <X size={15} strokeWidth={1.8} />
        </button>
      </div>

      <div className="relative flex min-h-0 flex-1 flex-col">
        {/* 消息区 */}
        <div
          ref={scrollRef}
          data-chat-scroll-root="true"
          onScroll={handleMessagesScroll}
          className="min-h-0 flex-1 overflow-y-auto px-4 py-4"
        >
          {hasNoMessages ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
              <MessageCircleQuestion
                size={28}
                strokeWidth={1.2}
                className="text-[var(--muted-foreground)]/60"
              />
              <p className="max-w-[240px] text-[13px] leading-relaxed text-[var(--muted-foreground)]">
                {t(
                  "Ask anything about the main conversation here — it's read as reference, and this scratch chat stays out of your history.",
                )}
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {messages.map((msg, index) =>
                msg.role === "user" ? (
                  <div key={index} className="flex justify-end">
                    <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl bg-[var(--secondary)] px-4 py-2.5 text-[14px] leading-relaxed text-[var(--foreground)] shadow-sm">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div key={index} className="flex flex-col items-start gap-1.5">
                    <div className="max-w-full">
                      <MarkdownRenderer content={msg.content} variant="compact" />
                    </div>
                    {msg.content.trim() ? (
                      <button
                        type="button"
                        onClick={() => handleAddToMain(msg.content, index)}
                        title={t("Add this answer to the main conversation")}
                        className={`inline-flex h-6 items-center gap-1 rounded-md px-2 text-[11px] transition-colors ${
                          addedIndex === index
                            ? "text-emerald-500"
                            : "text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
                        }`}
                      >
                        {addedIndex === index ? (
                          <Check size={12} strokeWidth={2} />
                        ) : (
                          <Plus size={12} strokeWidth={2} />
                        )}
                        {addedIndex === index
                          ? t("Added to main chat")
                          : t("Add to main chat")}
                      </button>
                    ) : null}
                  </div>
                ),
              )}
              {isStreaming ? (
                <div className="flex items-center gap-1.5 pl-1 text-[var(--muted-foreground)]">
                  <span className="text-[11px]">
                    {t("Generating...")}
                  </span>
                </div>
              ) : null}
              <div ref={endRef} className="h-px w-full shrink-0" />
            </div>
          )}
        </div>
        <OverlayScrollbar scrollRef={scrollRef} />

        {error ? (
          <div className="mx-4 mb-2 rounded-lg border border-[var(--destructive)]/30 bg-[var(--destructive)]/10 px-3 py-2 text-[12px] leading-relaxed text-[var(--destructive)]">
            {error}
          </div>
        ) : null}

        {/* 输入区 */}
        <div className="border-t border-[var(--border)] bg-[var(--card)] px-3 py-2.5 pb-[calc(0.625rem+env(safe-area-inset-bottom))]">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={isStreaming}
              placeholder={t("Ask about the main conversation…")}
              aria-label={t("Temporary question input")}
              className="min-h-[36px] max-h-40 flex-1 resize-none rounded-xl border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-[14px] leading-relaxed text-[var(--foreground)] shadow-sm outline-none transition focus:border-[var(--ring)] focus:ring-2 focus:ring-[var(--ring)]/20 disabled:opacity-60 placeholder:text-[var(--muted-foreground)]"
            />
            {isStreaming ? (
              <button
                type="button"
                onClick={cancelStreaming}
                title={t("Stop")}
                aria-label={t("Stop")}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[var(--muted)] text-[var(--muted-foreground)] transition-colors hover:text-[var(--foreground)]"
              >
                <Square size={14} strokeWidth={1.8} fill="currentColor" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!input.trim()}
                title={t("Send")}
                aria-label={t("Send")}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[var(--primary)] text-white transition-[background-color,transform] duration-150 active:scale-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send size={14} strokeWidth={1.8} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}