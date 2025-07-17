# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Causal analysis configuration."""

from typing import Any

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class CausalAnalysisConfig(BaseModel):
    """Causal analysis configuration."""

    enabled: bool = Field(
        description="Whether causal analysis is enabled.",
        default=defs.graphrag_config_defaults.causal_analysis.enabled,
    )
    """Whether causal analysis is enabled."""

    prompt: str | None = Field(
        description="The prompt to use for causal analysis.",
        default=defs.graphrag_config_defaults.causal_analysis.prompt,
    )
    """The prompt to use for causal analysis."""

    max_analysis_length: int = Field(
        description="The maximum length of the causal analysis report.",
        default=defs.graphrag_config_defaults.causal_analysis.max_analysis_length,
    )
    """The maximum length of the causal analysis report."""

    model_id: str = Field(
        description="The model ID to use for causal analysis.",
        default=defs.graphrag_config_defaults.causal_analysis.model_id,
    )
    """The model ID to use for causal analysis."""

    def resolved_strategy(self, root_dir: str, model_config: Any) -> dict[str, Any]:
        """Get the resolved strategy for causal analysis."""
        strategy = {}
        
        if self.prompt:
            strategy["prompt"] = self.prompt
        else:
            # Use default prompt from prompts directory
            from pathlib import Path
            prompt_path = Path(root_dir) / "prompts" / "causal_analysis.txt"
            if prompt_path.exists():
                strategy["prompt"] = str(prompt_path)
        
        strategy["llm"] = model_config.model_dump()
        strategy["max_analysis_length"] = self.max_analysis_length
        
        return strategy 