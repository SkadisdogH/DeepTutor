"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

interface OverlayScrollbarProps {
  /** 需要跟随/控制的滚动容器（data-chat-scroll-root） */
  scrollRef: React.RefObject<HTMLElement | null>;
  /** 滑块宽度（px） */
  width?: number;
  /** 滑块距右缘距离（px） */
  insetRight?: number;
  /** 滑块上下留白（px） */
  insetY?: number;
}

/**
 * 聊天/流式滚动面的自定义常驻滚动条。
 *
 * 背景：Firefox(Linux/GTK) 与 Chromium 的原生 scrollbar-color 在“按住滑块
 * 拖拽”时会按系统主题重新上色（FF153 实测按压态变浅 rgb(227,227,227)，
 * Gecko 源码 AdjustUnthemedScrollbarThumbColor 对 ACTIVE 态亮度 ×0.192），
 * 纯 CSS 无法阻止。因此 globals.css 对所有 ``data-chat-scroll-root`` 隐藏
 * 原生滚动条（scrollbar-width: none），本组件在这些滚动容器外层（必须是与
 * 滚动区等高的 relative 包裹层）渲染一个常驻深色滑块：静止 / 悬停 / 拖拽
 * 颜色恒定 #6b6b6b（DOM 元素，不受浏览器 GTK 状态影响）。
 *
 * 约定：挂载本组件的滚动容器必须带 ``data-chat-scroll-root="true"``（原生条
 * 已被 CSS 隐藏）；内容不溢出时不渲染（thumb.height = 0 → null）。
 *
 * 滑块支持鼠标拖拽：mousedown 记录起点与起始 scrollTop，mousemove 按轨道
 * 比例换算新的 scrollTop（拖拽全程滑块颜色不变）；触摸 / 触板滚动仍走原生
 * 事件，滑块经 scroll 事件跟随。
 */
export default function OverlayScrollbar({
  scrollRef,
  width = 8,
  insetRight = 2,
  insetY = 4,
}: OverlayScrollbarProps) {
  const [thumb, setThumb] = useState({ top: 0, height: 0 });
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef<{ startY: number; startScrollTop: number } | null>(
    null,
  );

  const update = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const sh = el.scrollHeight;
    const ch = el.clientHeight;
    if (sh <= ch) {
      setThumb((t) => (t.height === 0 ? t : { top: 0, height: 0 }));
      return;
    }
    // 滑块渲染在滚动容器父级（与滚动区等高的 relative 包裹层）里。可拖拽
    // “轨道”即滚动区可视高度去掉上下留白：顶部 insetY、底部 insetY。
    const trackH = ch - 2 * insetY;
    const ratio = ch / sh;
    const height = Math.max(28, Math.round(ratio * trackH));
    const maxTop = trackH - height;
    const top =
      insetY + Math.round((el.scrollTop / (sh - ch)) * Math.max(0, maxTop));
    setThumb({ top, height });
  }, [scrollRef, insetY]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    // 首次挂载在下一帧计算滑块几何（避免 effect 内同步 setState）。
    const raf = requestAnimationFrame(() => update());

    const ro = new ResizeObserver(() => update());
    ro.observe(el);

    const mo = new MutationObserver(() => update());
    mo.observe(el, { childList: true, subtree: true, characterData: true });

    el.addEventListener("scroll", update, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      mo.disconnect();
      el.removeEventListener("scroll", update);
    };
  }, [scrollRef, update]);

  // 滑块拖拽：mousedown 记录起点，全局 mousemove 按轨道比例换算 scrollTop。
  const startDrag = useCallback(
    (e: React.MouseEvent) => {
      const el = scrollRef.current;
      if (!el) return;
      const trackH = el.clientHeight - 2 * insetY;
      const height = Math.max(
        28,
        Math.round((el.clientHeight / el.scrollHeight) * trackH),
      );
      if (trackH - height <= 0) return;
      e.preventDefault();
      dragRef.current = { startY: e.clientY, startScrollTop: el.scrollTop };
      setDragging(true);
    },
    [scrollRef, insetY],
  );

  useEffect(() => {
    if (!dragging) return;
    const el = scrollRef.current;
    if (!el) return;
    const onMove = (ev: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const sh = el.scrollHeight;
      const ch = el.clientHeight;
      if (sh <= ch) return;
      const trackH = ch - 2 * insetY;
      const height = Math.max(28, Math.round((ch / sh) * trackH));
      const maxTop = trackH - height;
      if (maxTop <= 0) return;
      const dy = ev.clientY - d.startY;
      const ratio = (sh - ch) / maxTop;
      el.scrollTop = Math.max(
        0,
        Math.min(sh - ch, d.startScrollTop + dy * ratio),
      );
    };
    const onUp = () => {
      dragRef.current = null;
      setDragging(false);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging, scrollRef, insetY]);

  if (thumb.height === 0) return null;

  return (
    <div
      aria-hidden="true"
      data-overlay-scrollbar="true"
      onMouseDown={startDrag}
      style={{
        position: "absolute",
        right: insetRight,
        top: thumb.top,
        height: thumb.height,
        width,
        borderRadius: width / 2,
        background: "#6b6b6b",
        opacity: 0.85,
        cursor: dragging ? "grabbing" : "grab",
        userSelect: "none",
        touchAction: "none",
        zIndex: 40,
      }}
    />
  );
}
