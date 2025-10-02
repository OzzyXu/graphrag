# CausalRAG Final-Context Options – Development Notes

## Refined Goal
Add configuration that lets callers control whether selected network sections (entities, relationships, text units, community reports) and the causal summary share the Stage‑3 prompt. All components obey token budgets expressed as ratios; the default behaviour—summary only—remains unchanged.

## Proposed Config (`causal_search.final_context`)
- `include_causal_summary` (bool, default `true`).
- `ensure_full_summary` (bool, default `false`): when `true`, the summary is clipped last so it remains intact whenever budgets permit.
- `include_network_sections` (list of section identifiers: `entities`, `relationships`, `text_units`, `community_reports`).
- `section_token_ratios` (map: component → float ratio, summing ≤ 1.0) covering *all* components, including the summary. Ratios determine how much of `max_final_context_tokens` each block may consume. Example defaults: `{"summary": 0.4, "entities": 0.25, "relationships": 0.2, "text_units": 0.1, "community_reports": 0.05}`.
- `section_row_limits` (map: component → int) optional hard caps on the number of rows retained per section to keep tables focused. Example defaults: `{"entities": 12, "relationships": 30, "text_units": 8, "community_reports": 5}`.
- `max_final_context_tokens` (optional override; defaults to a value derived from `max_context_tokens`).

## Implementation Outline
1. **Config Modeling**
   - Extend `CausalSearchConfig` with a `final_context` dataclass carrying these fields, ratio defaults, and optional row caps.
   - Document new options in `docs/causal_search_usage.md` alongside existing causal search settings.

2. **Data Retention**
   - Reuse the formatted tables gathered during discovery. Track token counts per section so ratios can be enforced later.

3. **Context Assembly**
   - Start with the causal summary (subject to its ratio). Append requested sections verbatim, clipping each according to its ratio-derived token budget and row cap.
   - Respect `ensure_full_summary` by allocating the summary last if requested.
   - Enforce the aggregate `max_final_context_tokens` after concatenating all blocks.

4. **Streaming Parity**
   - Apply the same assembly logic inside `_stream_final_response` so streaming matches batch responses.

5. **Telemetry & Persistence**
   - Update trimming metadata to capture which sections were included and how much budget each consumed.

6. **Testing**
   - Unit tests for default behaviour, per-section inclusion, ratio edge cases, and `ensure_full_summary`.
   - Regression tests confirming the prompt stays unchanged when no sections are selected.

## Open Points
- Validate the initial defaults (ratios + row caps) against real datasets once implemented and adjust if necessary.
- Confirm interaction with trimming metadata and saved artifacts before release.
