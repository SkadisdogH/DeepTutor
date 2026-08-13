"""Math Research loop-capability hooks.

``math_research`` 复用了完整 chat 工具面，只是注入研究 playbook，因此不需要
专属 owned tools；``math_symbolic`` 已通过 chat 的代码沙箱门控自动挂载。
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from deeptutor.capabilities.protocol import PromptBlock
from deeptutor.core.context import UnifiedContext


class MathResearchLoopCapability:
    name = "math_research"
    owned_tools: tuple[str, ...] = ()

    def is_active(self, context: UnifiedContext) -> bool:
        return bool(context.metadata.get("math_research_mode"))

    def system_block(
        self,
        context: UnifiedContext,
        *,
        language: str,
        prompts: dict[str, Any],
    ) -> PromptBlock | None:
        if not self.is_active(context):
            return None
        lang = "zh" if language.lower().startswith("zh") else "en"
        prompt = resources.files(__package__).joinpath("prompts", lang, "system.md")
        return PromptBlock("math_research", prompt.read_text(encoding="utf-8").strip())

    def augment_kwargs(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        context: UnifiedContext,
    ) -> dict[str, Any]:
        return kwargs

    def pre_loop_seed(self, context: UnifiedContext) -> str:
        _ = context
        return ""


__all__ = ["MathResearchLoopCapability"]
