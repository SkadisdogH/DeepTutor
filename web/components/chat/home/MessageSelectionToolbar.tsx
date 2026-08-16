"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { createPortal } from "react-dom";
import { Check, Copy, MessageSquarePlus } from "lucide-react";

/**
 * 选中消息文本 → 浮动工具条（模仿旧 Obsidian 插件 / LobeHub 的“选区加入对话”）。
 *
 * 行为：
 *  - 在任意带 `data-message-selectable` 的消息气泡里用鼠标/键盘选中一段文本，
 *    选区上方会出现一个浮动小工具条；
 *  - 「复制」把选中文本复制到剪贴板；
 *  - 「加入对话」把选中文本以引用块（>）的形式回填到聊天输入框，由用户编辑后发送。
 *
 * 与 composer 的通信复用现有 `dt:visualize-prompt` 的窗口事件模式：
 *  dispatch 一个 `dt:add-to-conversation` 事件，页面监听后调用 handlePrefillComposer。
 */
export const ADD_TO_CONVERSATION_EVENT = "dt:add-to-conversation";

function blockquote(text: string): string {
  return text
    .split("\n")
    .map((line) => (line.trim() ? `> ${line}` : ">"))
    .join("\n");
}

function quoteWithSource(
  text: string,
  sourceRole: string | null,
  sourceIndex: number | null,
  t: (key: string, opts?: Record<string, unknown>) => string,
): string {
  const sourceLabel =
    sourceRole === "user" ? t("Quote source user") : t("Quote source AI");
  const whereClause =
    sourceIndex != null ? ` (${t("Quote where", { n: sourceIndex + 1 })})` : "";
  const intro = `${t("Quote intro", { source: sourceLabel, where: whereClause })}`;
  return `${intro}\n\n${blockquote(text)}`;
}

export function MessageSelectionToolbar() {
  const { t } = useTranslation();
  const [sel, setSel] = useState<{
    text: string;
    rect: DOMRect;
    sourceRole: string | null;
    sourceIndex: number | null;
  } | null>(null);
  const [copied, setCopied] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const hide = useCallback(() => {
    setSel(null);
    setCopied(false);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const collect = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) {
      hide();
      return;
    }
    const text = selection.toString().trim();
    if (!text) {
      hide();
      return;
    }
    // 只响应发生在消息气泡内的选区。
    const inBubble = (node: Node | null): boolean => {
      if (!node) return false;
      const el = node instanceof Element ? node : node.parentElement;
      return !!el?.closest?.("[data-message-selectable]");
    };
    if (!inBubble(selection.anchorNode) && !inBubble(selection.focusNode)) {
      hide();
      return;
    }
    let range: Range;
    try {
      range = selection.getRangeAt(0);
    } catch {
      hide();
      return;
    }
    // 出处：读取选区所在消息气泡的 role 与序号，方便模型定位到原消息。
    const anchor =
      selection.anchorNode instanceof Element
        ? selection.anchorNode
        : selection.anchorNode?.parentElement ?? null;
    const bubble = anchor?.closest?.("[data-message-selectable]") ?? null;
    const rawIndex = bubble?.getAttribute("data-message-index");
    const sourceIndex =
      rawIndex != null && !Number.isNaN(Number(rawIndex))
        ? Number(rawIndex)
        : null;
    setCopied(false);
    setSel({
      text,
      rect: range.getBoundingClientRect(),
      sourceRole: bubble?.getAttribute("data-message-role") ?? null,
      sourceIndex,
    });
  }, [hide]);

  useEffect(() => {
    document.addEventListener("mouseup", collect);
    document.addEventListener("keyup", collect);
    document.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    // 点击工具条以外的任意位置关闭；点击工具条自身交给 onClick 处理。
    const onDocMouseDown = (e: MouseEvent) => {
      if (barRef.current && barRef.current.contains(e.target as Node)) return;
      hide();
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => {
      document.removeEventListener("mouseup", collect);
      document.removeEventListener("keyup", collect);
      document.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
      document.removeEventListener("mousedown", onDocMouseDown);
    };
  }, [collect, hide]);


  const handleCopy = useCallback(async () => {
    if (!sel) return;
    try {
      await navigator.clipboard.writeText(sel.text);
      setCopied(true);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(hide, 1200);
    } catch {
      hide();
    }
  }, [sel, hide]);

  const handleAddToConversation = useCallback(() => {
    if (!sel) return;
    window.dispatchEvent(
      new CustomEvent<string>(ADD_TO_CONVERSATION_EVENT, {
        detail: quoteWithSource(sel.text, sel.sourceRole, sel.sourceIndex, t),
      }),
    );
    hide();
  }, [sel, hide, t]);

  // 把工具条定位在选区上方（空间不足时放到选区下方），并夹在视口内。
  // 尺寸用近似值（工具条只有“复制 / 加入对话”两个按钮），在渲染期直接算出，
  // 避免在 effect 里调用 setState。
  const BAR_W = 200;
  const BAR_H = 36;
  // 工具条与选区之间的留白：让工具条整体避开选中内容，不压住选中的文本。
  const BAR_GAP = 16;
  const pos = (() => {
    if (!sel) return null;
    const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
    const vh = typeof window !== "undefined" ? window.innerHeight : 800;
    const centerX = sel.rect.left + sel.rect.width / 2;
    let left = centerX - BAR_W / 2;
    left = Math.max(8, Math.min(left, vw - BAR_W - 8));
    let top = sel.rect.top - BAR_H - BAR_GAP;
    if (top < BAR_GAP) top = sel.rect.bottom + BAR_GAP;
    if (top + BAR_H > vh - 8) top = Math.max(8, vh - BAR_H - 8);
    return { left, top };
  })();

  if (!sel) return null;

  // 渲染到 document.body：工具条若留在消息列表里，祖先 data-chat-scroll-root 的
  // mask-image（底部渐隐）会成为 fixed 后代的包含块，把视口坐标错位到滚动容器
  // 坐标系，导致工具条压住选中的内容。portal 到 body 后 fixed 恢复相对视口定位。
  return typeof document === "undefined"
    ? null
    : createPortal(
        <div
          ref={barRef}
          role="toolbar"
          aria-label={t("Message selection actions")}
          className="fixed z-[60] flex items-center gap-0.5 rounded-lg border border-[var(--border)] bg-[var(--card)] p-1 shadow-lg"
          style={pos ?? undefined}
          onMouseDown={(e) => e.preventDefault()}
        >
          <button
            type="button"
            onClick={() => void handleCopy()}
            aria-label={copied ? t("Copied") : t("Copy")}
            className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)]/50 hover:text-[var(--foreground)]"
          >
            {copied ? (
              <Check size={14} strokeWidth={2} className="text-[var(--primary)]" />
            ) : (
              <Copy size={14} strokeWidth={1.5} />
            )}
            {copied ? t("Copied") : t("Copy")}
          </button>
          <span className="mx-0.5 h-4 w-px bg-[var(--border)]" />
          <button
            type="button"
            onClick={handleAddToConversation}
            aria-label={t("Add to conversation")}
            className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)]/50 hover:text-[var(--foreground)]"
          >
            <MessageSquarePlus size={14} strokeWidth={1.5} />
            {t("Add to conversation")}
          </button>
        </div>,
        document.body,
      );
}
