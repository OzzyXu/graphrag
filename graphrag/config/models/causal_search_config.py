# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the causal search configuration."""

from typing import Dict, List

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class CausalFinalContextConfig(BaseModel):
    """Configuration for final response context assembly."""

    include_causal_summary: bool = Field(
        description="Whether to include the causal summary in the final prompt.",
        default=graphrag_config_defaults.causal_search.final_context.include_causal_summary,
    )
    ensure_full_summary: bool = Field(
        description="Ensure the summary is clipped last so it remains intact when possible.",
        default=graphrag_config_defaults.causal_search.final_context.ensure_full_summary,
    )
    include_network_sections: List[str] = Field(
        description="Ordered list of network sections to append (e.g. entities, relationships).",
        default_factory=lambda: list(
            graphrag_config_defaults.causal_search.final_context.include_network_sections
        ),
    )
    section_token_ratios: Dict[str, float] = Field(
        description="Token ratio per component (including summary).",
        default_factory=lambda: dict(
            graphrag_config_defaults.causal_search.final_context.section_token_ratios
        ),
    )
    section_row_limits: Dict[str, int] = Field(
        description="Row limits per section when composing the final context.",
        default_factory=lambda: dict(
            graphrag_config_defaults.causal_search.final_context.section_row_limits
        ),
    )
    max_final_context_tokens: int | None = Field(
        description="Optional override for total Stage 3 context tokens.",
        default=graphrag_config_defaults.causal_search.final_context.max_final_context_tokens,
    )


class CausalSearchConfig(BaseModel):
    """Configuration section for causal search."""

    prompt: str | None = Field(
        description="The causal search prompt to use.",
        default=graphrag_config_defaults.causal_search.prompt,
    )
    s_parameter: int = Field(
        description="Additional nodes beyond local search for causal analysis.",
        default=graphrag_config_defaults.causal_search.s_parameter,
    )
    top_k_mapped_entities: int = Field(
        description="The top k mapped entities.",
        default=graphrag_config_defaults.causal_search.top_k_mapped_entities,
    )
    top_k_relationships: int = Field(
        description="The top k relationships.",
        default=graphrag_config_defaults.causal_search.top_k_relationships,
    )
    text_unit_prop: float = Field(
        description="The text unit proportion.",
        default=graphrag_config_defaults.causal_search.text_unit_prop,
    )
    community_prop: float = Field(
        description="The community proportion.",
        default=graphrag_config_defaults.causal_search.community_prop,
    )
    max_context_tokens: int = Field(
        description="The maximum context size in tokens.",
        default=graphrag_config_defaults.causal_search.max_context_tokens,
    )
    chat_model_id: str = Field(
        description="The model ID to use for causal search.",
        default=graphrag_config_defaults.causal_search.chat_model_id,
    )
    embedding_model_id: str = Field(
        description="The embedding model ID to use for causal search.",
        default=graphrag_config_defaults.causal_search.embedding_model_id,
    )
    save_network_data: bool = Field(
        description="Whether to save extracted network data to files.",
        default=graphrag_config_defaults.causal_search.save_network_data,
    )
    save_causal_report: bool = Field(
        description="Whether to save causal analysis report to files.",
        default=graphrag_config_defaults.causal_search.save_causal_report,
    )
    output_folder: str = Field(
        description="Subfolder under data/outputs/ for causal search outputs.",
        default=graphrag_config_defaults.causal_search.output_folder,
    )
    final_context: CausalFinalContextConfig = Field(
        description="Final response context assembly options.",
        default_factory=CausalFinalContextConfig,
    )
