# CausalRAG Test Coverage Notes

## Unit Tests Added
- `tests/unit/query/structured_search/test_causal_search.py::test_causal_search_passes_entity_hints_and_includes_sources`
  - Exercises the discovery stage end-to-end with stubbed chat model and context builder.
  - Asserts that the second context build reuses the original query and injects entity name hints, and that discovery prompts include source snippets.
  - Confirms token budgeting logic trims oversized causal reports before the summary stage and records telemetry for all stages.
- `tests/unit/query/structured_search/test_causal_search.py::test_stream_search_streams_llm_tokens`
  - Verifies the streaming API now relays tokens emitted by `achat_stream` instead of replaying the final answer.
- `tests/unit/query/structured_search/test_causal_search.py::test_causal_search_includes_conversation_history`
  - Ensures conversation history objects propagate through both context-build passes, enabling multi-turn recall.
- `tests/unit/query/structured_search/test_causal_search.py::test_causal_search_output_files_include_trim_markers`
  - Confirms saved artifacts bundle trimming metadata: JSON output embeds `_trimmed_meta`, and Markdown reports append a human-readable trimming note.
- `tests/unit/query/structured_search/test_causal_search.py::test_causal_search_raises_when_model_returns_no_content`
  - Guards against regressions where providers omit `output.content` by asserting a `CausalSearchError` with no brittle regex fallbacks.

## Design Notes
- Tests fully stub the LLM interface (`FakeChatModel`) to avoid external calls and associated costs.
- `RecordingContextBuilder` fakes the mixed-context builder while preserving key behaviours (entity hints, source tables) so we can assert pipeline wiring without graph fixtures.
- Token clipping relies on the real `tiktoken` encoder to match production sizing, but the payload sizes are tiny to keep execution fast.
- Async tests use `pytest.mark.asyncio`; no network or filesystem writes beyond temporary objects are performed.

## Next Steps
- Extend coverage with regression tests for token overflow edge cases once integration fixtures are available.
- Consider adding snapshot tests for saved network data artifacts when the IO layer is stabilised.
