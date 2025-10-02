# CausalRAG Implementation Review

## Architecture Overview
- CLI entrypoint wires `graphrag query --causal` to load index outputs and invoke the API-based search route, supporting both single and multi-index runs (`graphrag/cli/query.py:523`).
- The query API converts parquet artifacts into domain objects, resolves vector stores, and instantiates the causal search engine through `get_causal_search_engine` (`graphrag/api/query.py:1215`, `graphrag/query/factory.py:320`).
- `get_causal_search_engine` composes a `CausalSearch` orchestrator with `LocalSearchMixedContext`, using configured LLM and embedding backends plus token encoder details from the settings file (`graphrag/query/factory.py:320`).
- `CausalSearch` builds context, runs two sequential LLM calls (discovery then summary), and optionally persists intermediate artifacts (`graphrag/query/structured_search/causal_search/search.py:41`).

## Execution Flow Snapshot
1. **Context expansion** – `_extract_extended_nodes` attempts to widen the entity pool by bumping `top_k_mapped_entities`, but the returned IDs are not reused downstream (`graphrag/query/structured_search/causal_search/search.py:245`).
2. **Context rebuild** – `_extract_graph_information` rebuilds the mixed context but calls `build_context` with an empty query, bypassing the original user question (`graphrag/query/structured_search/causal_search/search.py:319`).
3. **Prompt prep** – `_format_network_data_for_causal_prompt` trims entities/relationships/text units before injecting JSON into the discovery prompt (`graphrag/query/structured_search/causal_search/search.py:340`).
4. **LLM stages** – `_generate_causal_report` and `_generate_final_response` perform sequential `achat` calls, using the discovery output as the summary stage context (`graphrag/query/structured_search/causal_search/search.py:452`, `graphrag/query/structured_search/causal_search/search.py:520`).
5. **Persistence** – `_save_outputs` writes optional JSON/Markdown artifacts to a configurable output folder (`graphrag/query/structured_search/causal_search/search.py:556`).

## Token Budget & Context Handling
- Local context construction already honours `max_context_tokens`, but downstream formatting hardcodes a separate `max_total_tokens = 8000`, ignoring the configured 12k window and risking double truncation or unnecessary clipping when larger budgets are set (`graphrag/query/structured_search/causal_search/search.py:345`).
- Stage 2 injects the entire causal report into the summary prompt without enforcing a cap, so long reports can exceed the chat model window and trigger 4xx errors in practice.
- Token accounting in the search result only records prompt/output tokens for discovery/summary, omitting the context-build cost and keeping `llm_calls` at zero (`graphrag/query/structured_search/causal_search/search.py:158`).

## Issues & Risks
- **High – Queryless context rebuild**: `_extract_graph_information` replays the local search builder with `query=""`, meaning the embedding lookup runs against an empty string and ignores the original user intent. As a result the second context build regresses to whatever the model considers "closest" to nothing (often high-degree hubs), and the widened entity list from Stage 1 is discarded because the `selected_nodes` argument is never applied (`graphrag/query/structured_search/causal_search/search.py:370`).
- **High – Text unit omission**: `_format_network_data_for_causal_prompt` looks for `context_records['text_units']`, but the mixed context stores text units under the `sources` key (`graphrag/query/structured_search/causal_search/search.py:378`, `graphrag/query/structured_search/local_search/mixed_context.py:312`), so sources rarely reach the discovery prompt.
- **High – Token budget mismatch**: Fixed 8k network budget ignores `config.causal_search.max_context_tokens`, and discovery output is replayed into Stage 2 with no truncation safeguards, leading to failures when the two prompts exceed model limits (`graphrag/query/structured_search/causal_search/search.py:345`, `graphrag/query/structured_search/causal_search/search.py:520`).
- **Medium – Instrumentation gaps**: `llm_calls` remains empty and prompt token tracking omits context-build usage, yielding misleading telemetry in API responses and logs (`graphrag/query/structured_search/causal_search/search.py:158`).
- **Medium – Streaming shim**: `stream_search` replays the full response word-by-word after blocking on `search`, so it neither reduces latency nor invokes `callbacks.on_llm_new_token` (`graphrag/query/structured_search/causal_search/search.py:231`).
- **Medium – Multi-index plumbing leftovers**: The aggregation helper builds `links` maps and mutates `communities` columns but never returns these structures, increasing complexity without observable benefit and leaving potential type issues (lists in `max()` calls) in `multi_index_causal_search` (`graphrag/api/query.py:1436`).

## Additional Notes
- Prompt fallbacks and regex extraction logic are defensive, but consider standardising on the response schema already used in Local Search to reduce brittle string parsing (`graphrag/query/structured_search/causal_search/search.py:472`).
- `_save_prompts` is unused; removing or wiring it behind a configuration flag would avoid confusion about expected side effects (`graphrag/query/structured_search/causal_search/search.py:600`).
