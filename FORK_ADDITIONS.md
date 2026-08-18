# 本 Fork 新增内容（Fork Additions）

> 本仓库是 [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor) 的一个 **fork**，
> 在官方版本之上，面向「数学研究 agent」场景叠加了以下增强。当前工作分支：`feat/math-research`。
>
> **运行方式差异**：官方推荐 Docker/Compose；本 fork 使用本机 systemd 用户服务直接跑
> （后端 `:8001`，前端 `:3782`），以 `deeptutor start --home <dir>` 冷启动，见文末「安装与运行」。

---

## 1. 🧮 数学研究能力（核心新增）

- **`math_research` 能力**（`deeptutor/capabilities/math_research/`：`capability.py` + `loop.py` + `prompts/{en,zh}`）
  经 `deeptutor/capabilities/registry.py` 代码级注册 + `runtime/bootstrap/builtin_capabilities.py` 声明，
  **冷启动自动可用**。流水线覆盖：数学猜想 → 文献检索 → 符号/数值验证 → 研究问题拆解 → 迭代深化。

- **`math_symbolic` 工具**（`deeptutor/tools/math_symbolic.py`，进程内 SymPy 符号验证）
  为让沙箱环境稳定可用，把 `sympy>=1.12,<2.0.0` 放进了 **pyproject 核心依赖**。

- **内置 Skills**（`deeptutor/skills/builtin/`）
  - `math-conjecture` —— 猜想生成与形式化
  - `math-symbolic-check` —— SymPy 符号验证（复用 math_symbolic）
  - `math-lit-review` —— 数学文献综述

- **人设预设** `deeptutor/services/persona/presets/math-research-tutor/` ——「数学研究导师」，
  在网页 Settings 一键启用。

- **LLM 层配套**：优先走流式（适配仅流式网关，空响退化为非流式）；上下文窗口探测尊重配置值；
  LLM 诊断改用流式。

> 注：本仓库中的 `math_animator` 能力（Manim 动画，extra `.[math-animator]`）是**上游官方功能**，
> 非本 fork 新增，此处仅列举真实差异。

## 2. 📚 知识库 / RAG 增强

- **`add_to_kb` 工具**（`deeptutor/tools/add_document_tool.py`）：对话中直接把文档/链接加入知识库。
- **大批量分块上传 + 内容哈希去重 + zip 上传修复**
  （`web/lib/knowledge-api.ts`；测试 `tests/api/test_knowledge_chunked_upload.py`、`test_knowledge_hash_dedupe.py`）。
- **web_fetch fake-ip 修复**；**加粗 + 行内 LaTeX 渲染修复**（`web/lib/latex.ts`、`web/lib/iframe-html.ts`、`web/tests/latex.test.ts`）。
- **KB 索引探测记忆化缓存**（`deeptutor/services/rag/index_probe.py` + `knowledge/manager.py`）
  以文件系统快照（relpath + mtime_ns + size）为 key 缓存 `inspect_kb_versions`，避免知识库轮询 `/list`
  时反复全量解析 docstore.json（实测对 88MB 数学 KB 造成约 **40% 单核空转**），索引真实变更时快照自动失效。
- **MinerU 大 PDF 拆分解析**（`deeptutor/services/parsing/engines/mineru/cloud.py`）：
  页数探测 → 分片 → 云端解析 → 分片合并，突破单文件大小/超时限制。
- **RAG 管道稳定性修复**：graphrag / lightrag / llamaindex 导入管道修正（未提交工作区内）。

## 3. 💬 对话体验 / Web

- **SideChat 临时提问抽屉**（`web/components/chat/sidechat/SideChatPanel.tsx`）：主对话旁的右侧抽屉，
  可随时「就主对话提问」。每次提问自动把主会话作为参考交给模型；问答存入真实会话但带
  `preferences.temporary` 标记，**不出现在历史列表**（`api/routers/sessions.py` 统一过滤、
  `services/session/turn_runtime.py` 打标）。面板常驻挂载、流式中途关闭不丢事件、按主会话隔离缓存于
  localStorage；「加入主对话」可把回答预填进主对话输入框。
- **消息选择 → 引用 / 加入对话**；引用带来源角色与下标（`MessageSelectionToolbar.tsx`）；选中工具栏 portal 固定到 body。
- **自定义滚动条** `OverlayScrollbar.tsx`（classic / overlay 统一为 DOM 滑块）。
- **KaTeX 显示公式溢出容器化**，不再横向拖拽整页。
- 配套测试：`tests/api/test_sidechat_temporary.py`、`tests/api/test_index_probe_cache.py`。

## 4. 🛡️ LLM 错误透传

- **真实 provider 错误透传**，不再用误导性 fallback 掩盖（`tests/services/llm/test_openai_compat_fallback.py`、
  `tests/utils/test_user_facing_errors.py`）。

---

## 安装与运行（针对本 fork）

前置：**Python 3.11–3.13**、**Node.js 22 LTS**。

```bash
# 1) 后端（可编辑安装，开发调试推荐）
python -m pip install --upgrade pip
python -m pip install -e .          # sympy 已含在核心依赖；动画需求再加 -e ".[math-animator]"

# 2) 前端
cd web && npm ci --legacy-peer-deps && npm run build && cd ..

# 3) 初始化 + 启动（--home 指向运行时数据根，初始化自动建骨架）
deeptutor init --home <你的运行时目录>
deeptutor start --home <你的运行时目录>
```

- 浏览器打开前端地址，在 **Settings 里配置自己的模型 / embedding Key**（本 fork 开发时主模型经中转站
  `cf.api.fan` 仅支持流式；embedding 用硅基流动 `BAAI/bge-m3` 1024 维 —— 均可自行更换）。
- 本机部署 = 用户级 systemd 服务 `deeptutor.service`；本地 runtime 目录 `/home/skadi/code/mathAgent/runtime`。
- 运行时数据（API Key、知识库、记忆、聊天历史）**不入库**，clone 后自行重建，属预期行为。

## 与官方同步

```bash
git fetch upstream -p
git merge upstream/main        # 或 git rebase upstream/main
```

> 共享文件（`web/app/globals.css`、`api/routers/sessions.py`、`services/session/turn_runtime.py`、RAG pipeline 等）
> 与官方新版本合并时可能产生冲突，需手动取舍。
