"""Math Research capability — 数学研究模式，复用标准 chat agent loop。

没有独立的 pipeline：进入该模式后仍跑统一的 agentic chat loop，由
:class:`deeptutor.capabilities.math_research.loop.MathResearchLoopCapability`
注入数学研究的 playbook。模型在循环里自主选择
``math_symbolic`` / ``web_search`` / ``paper_search`` / ``rag`` /
``code_execution`` 等工具做验证与检索。
"""

from __future__ import annotations

from deeptutor.agents.chat.agentic_pipeline import AgenticChatPipeline
from deeptutor.core.capability_protocol import BaseCapability, CapabilityManifest
from deeptutor.core.context import UnifiedContext
from deeptutor.core.stream_bus import StreamBus
from deeptutor.runtime.request_contracts import get_capability_request_schema


class MathResearchCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="math_research",
        description="数学研究模式：猜想分析、文献检索、符号/数值验证、研究问题拆解。",
        stages=["responding"],
        tools_used=[
            "math_symbolic",
            "web_search",
            "paper_search",
            "rag",
            "code_execution",
            "reason",
            "read_memory",
            "write_memory",
        ],
        cli_aliases=["math-research"],
        request_schema=get_capability_request_schema("math_research"),
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        context.metadata["math_research_mode"] = True
        pipeline = AgenticChatPipeline(language=context.language)
        await pipeline.run(context, stream)


__all__ = ["MathResearchCapability"]
