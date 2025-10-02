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
- **Resolved – Queryless context rebuild**: `_extract_graph_information` now receives the real query plus entity hints (see `search.py:268`), so retrieval stays anchored to the user request. Keep an eye on edge-cases where oversized history may consume the context budget.
- **Resolved – Text unit omission**: Discovery payloads fall back to `sources` when `text_units` is empty (`search.py:596`), ensuring snippet coverage.
- **Resolved – Token budget mismatch**: Dynamic budgets and prompt-level clipping enforce `max_context_tokens`, and saved artifacts note any trimming (`search.py:555`, `search.py:773`).
- **Medium – Multi-index plumbing**: The helper no longer builds unused link maps, but the broader flow remains untested; consider regression coverage and potential schema normalisation (`api/query.py:1408`).

## Additional Notes
- Prompt fallbacks and regex extraction logic are defensive, but consider standardising on the response schema already used in Local Search to reduce brittle string parsing (`graphrag/query/structured_search/causal_search/search.py:472`).
- `_save_prompts` is unused; removing or wiring it behind a configuration flag would avoid confusion about expected side effects (`graphrag/query/structured_search/causal_search/search.py:600`).
