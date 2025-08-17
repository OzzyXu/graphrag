# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the causal search configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


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
