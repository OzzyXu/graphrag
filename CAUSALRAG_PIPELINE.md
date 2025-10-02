# CausalRAG Response Generation Flow

## Stage 0 — Entry Points
- `search(query, ...)` orchestrates batch execution (`graphrag/query/structured_search/causal_search/search.py:148`).
- `stream_search(query, ...)` shares the discovery stage with `search`, then streams the summary tokens (`graphrag/query/structured_search/causal_search/search.py:233`).

## Stage 1 — Discovery Context
1. `_extract_extended_nodes` widens retrieval by oversampling entity IDs and titles, using the original query and mixed-context builder (`graphrag/query/structured_search/causal_search/search.py:394`).
2. `_extract_graph_information` rebuilds context with the real query plus entity hints, capturing entity/relationship/source tables and telemetry (`graphrag/query/structured_search/causal_search/search.py:509`).
3. `_format_network_data_for_causal_prompt` trims entities/relationships/sources into a JSON payload sized against `max_context_tokens` (`graphrag/query/structured_search/causal_search/search.py:555`).

## Stage 2 — Causal Report
- `_generate_causal_report` clips the network payload, renders the causal discovery system prompt, and performs the first `achat` LL.M call (`graphrag/query/structured_search/causal_search/search.py:689`).
- The prompt template originates from `CAUSAL_DISCOVERY_PROMPT` (`graphrag/prompts/query/causal_discovery_prompt.py:1`).
- Returned text, prompt, and trimmed payload feed downstream telemetry and optional persistence.

## Stage 3 — Summary Response
1. `_generate_final_response` trims the discovery report to a summary window, renders the summary prompt, and calls the chat model again (`graphrag/query/structured_search/causal_search/search.py:773`).
2. The summary uses `CAUSAL_SUMMARY_PROMPT` (`graphrag/prompts/query/causal_summary_prompt.py:1`).
3. Non-streaming code records token usage and returns a `SearchResult` with both context text and tables (`graphrag/query/structured_search/causal_search/search.py:214`).

## Streaming Path
- `_run_discovery_stage` executes Stages 1–2 once, then `_stream_final_response` forwards `achat_stream` tokens from the model driven by the summary prompt (`graphrag/query/structured_search/causal_search/search.py:268`, `graphrag/query/structured_search/causal_search/search.py:364`).

## Artifact Persistence
- `_save_outputs` optionally writes trimmed network data and summary report to disk, keyed by a deterministic query hash (`graphrag/query/structured_search/causal_search/search.py:859`).

## Telemetry
- Per-stage prompt/output token counts and LL.M call tallies are gathered during discovery and summary to populate `SearchResult.llm_calls_categories` (`graphrag/query/structured_search/causal_search/search.py:194`).
