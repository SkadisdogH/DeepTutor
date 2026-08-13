---
name: math-lit-review
description: 做数学主题的文献综述：用 deep_research + paper_search + web_search + rag 检索、组织并写出带引用的调研报告。当用户要求调研某个数学主题、综述某领域进展或追踪某篇论文的来龙去脉时使用。
tags:
  - math
  - research
---

# 数学文献综述

把一个数学主题做成带引用的调研报告。

## 流程

1. **先明确主题边界**：问清楚用户要的是「学习笔记 / 综述报告 / 对比 / 学习路径」，以及深度（quick / standard / deep / manual）。
2. **选对检索源**：
   - 已建的知识库用 `rag`（`--kb <name>` 挂载）。
   - 论文优先 `paper_search`（arXiv）。
   - 补充背景用 `web_search`。
3. **用 `deep_research` 跑研究**：它负责 澄清 → 拆解 → 逐块检索 → 带引用报告，不要自己手动拼引用。
4. **数学内容先验证**：报告里出现的关键公式/定理，能用 `math_symbolic` 验证的尽量验证，并标注验证结果。

## 引用要求

- 每条外部事实都要有来源（arXiv id / DOI / URL / 知识库文档）。
- 区分「文献结论」和「你自己的解读」。
- 不确定的内容明确标注。

## 输出

- 报告用 Markdown + LaTeX（`$...$` / `$$...$$`）。
- 结尾给 2–3 条「下一步该读什么」。
