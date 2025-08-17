# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All prompts for the query engine."""

from .basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
from .causal_discovery_prompt import CAUSAL_DISCOVERY_PROMPT
from .causal_summary_prompt import CAUSAL_SUMMARY_PROMPT
from .drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT, DRIFT_PRIMER_PROMPT
from .global_search_knowledge_system_prompt import GENERAL_KNOWLEDGE_INSTRUCTION
from .global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from .global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
from .local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from .question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

__all__ = [
    "BASIC_SEARCH_SYSTEM_PROMPT",
    "CAUSAL_DISCOVERY_PROMPT",
    "CAUSAL_SUMMARY_PROMPT",
    "DRIFT_LOCAL_SYSTEM_PROMPT",
    "DRIFT_REDUCE_PROMPT", 
    "DRIFT_PRIMER_PROMPT",
    "GENERAL_KNOWLEDGE_INSTRUCTION",
    "MAP_SYSTEM_PROMPT",
    "REDUCE_SYSTEM_PROMPT",
    "LOCAL_SEARCH_SYSTEM_PROMPT",
    "QUESTION_SYSTEM_PROMPT",
]
