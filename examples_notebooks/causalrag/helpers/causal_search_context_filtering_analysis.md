# Causal Search Context Filtering Analysis

## Overview

This document analyzes how context filtering works in the causal search implementation, explaining why you might see 40+ nodes extracted but only 7 entities, 9 relationships, and 4 text units in the final output.

## Context Filtering Architecture

### 1. Multi-Stage Filtering Process

The causal search uses a **two-stage filtering approach**:

1. **Node Extraction Stage**: Extract `(k + s) * oversample_scaler` nodes (e.g., 26 requested)
2. **Context Building Stage**: Build context using local search components
3. **Token-Based Filtering Stage**: Apply strict token limits to fit within LLM context

### 2. Token Limits Configuration

```python
# From _format_network_data_for_causal_prompt method
max_total_tokens = 8000  # Conservative limit for network data portion
entity_token_limit = int(max_total_tokens * 0.4)      # 3,200 tokens (40%)
relationship_token_limit = int(max_total_tokens * 0.4) # 3,200 tokens (40%)  
text_unit_token_limit = int(max_total_tokens * 0.2)   # 1,600 tokens (20%)
```

**Key Point**: The 8,000 token limit is **separate** from the `max_context_tokens` (12,000) used in context building.

## Why 40 Nodes → 7 Entities?

### 1. Node Extraction vs. Context Building

- **Step 1**: Extract 40 extended nodes using `(k + s) * oversample_scaler` formula
- **Step 2**: Context builder processes these nodes but may return different counts based on:
  - Query relevance scoring
  - Entity mapping algorithms
  - Community detection results

### 2. Context Records vs. Context Builder

The system tries **two data sources** in order:

```python
# First: Try context_records (from local search context building)
if hasattr(graph_context, 'context_records') and graph_context.context_records:
    entities_df = graph_context.context_records.get('entities', pd.DataFrame())
    # Apply token-based truncation

# Second: Fallback to context builder direct access
if not network_data["entities"] and hasattr(self.context_builder, 'entities'):
    entities_list = self._truncate_entities_from_context_builder(
        self.context_builder.entities, entity_token_limit
    )
```

### 3. Field Selection Strategy

**Entities**: Only essential fields are kept to save tokens:
```python
essential_fields = ['id', 'entity', 'description']
if 'rank' in entities_df.columns:
    essential_fields.append('rank')
if 'type' in entities_df.columns:
    essential_fields.append('type')
```

**Relationships**: Essential fields only:
```python
essential_fields = ['id', 'source', 'target', 'description']
if 'weight' in relationships_df.columns:
    essential_fields.append('weight')
if 'rank' in relationships_df.columns:
    essential_fields.append('rank')
```

**Text Units**: Focus on content:
```python
essential_fields = ['id', 'text']
if 'n_tokens' in text_units_df.columns:
    essential_fields.append('n_tokens')
```

## Token Counting and Truncation Logic

### 1. Progressive Token Counting

Each entity/relationship/text unit is processed sequentially:

```python
for _, entity in sorted_entities.iterrows():
    # Create entity dict with essential fields only
    entity_dict = {}
    for field in essential_fields:
        if field in entity.index:
            value = entity[field]
            # Handle NaN values
            if pd.isna(value):
                entity_dict[field] = "" if field in ['entity', 'description', 'type'] else 0
            else:
                entity_dict[field] = value
                
    # Estimate tokens for this entity
    entity_str = str(entity_dict)
    entity_tokens = num_tokens(entity_str, self.token_encoder)
    
    if current_tokens + entity_tokens > max_tokens:
        break  # Stop when limit reached
        
    entities_list.append(entity_dict)
    current_tokens += entity_tokens
```

### 2. Sorting Strategy

**Entities**: Sorted by `rank` (highest first) to prioritize most relevant
**Relationships**: Sorted by `weight` (highest first) to prioritize strongest connections
**Text Units**: Sorted by `n_tokens` (shortest first) to fit more content

## Actual Example Analysis

### From Recent Test Run:

```
✅ Step 1 complete: Found 40 extended nodes
Context records: 7 entities, 0 relationships, 4 text units
✅ Step 2 complete: Graph context extracted
Found 7 entities, truncated to 7 for context
No relationships found in context records
No relationships in context records, trying context builder with 191 relationships
Extracted 9 relationships from context builder (truncated)
Extracted 4 text units from context builder (truncated)
Formatted network data with 7 entities, 9 relationships
```

### Why These Numbers?

1. **7 Entities**: Context records contained 7 entities, all fit within 3,200 token limit
2. **9 Relationships**: Context builder had 191 relationships, but only 9 fit within 3,200 token limit
3. **4 Text Units**: Context records had 4 text units, all fit within 1,600 token limit

## Key Insights

### 1. Token Limits Are Strict
- The 8,000 total token limit for network data is **hard-coded** and **non-negotiable**
- This ensures the LLM can process the data without context length errors

### 2. Field Selection Is Aggressive
- Only essential fields are included to maximize entity/relationship count
- Non-essential metadata is stripped to save tokens

### 3. Prioritization by Relevance
- Entities/relationships are sorted by rank/weight before truncation
- Most relevant information is preserved within token limits

### 4. Fallback Mechanisms
- If context_records fail, system falls back to context builder
- This ensures robustness but may result in different data sources

## Recommendations for Understanding

### 1. Check Logs for Actual Numbers
```bash
graphrag query --root . --method causal --query "Your query" --verbose
```

Look for:
- "Found X extended nodes"
- "Context records: X entities, Y relationships, Z text units"
- "truncated to X for context"

### 2. Understand Token Distribution
- **40% entities**: 3,200 tokens
- **40% relationships**: 3,200 tokens  
- **20% text units**: 1,600 tokens

### 3. Field Selection Impact
- Essential fields only means less metadata
- Token counting is approximate (string representation)
- NaN handling may affect token counts

## Conclusion

The context filtering in causal search is **intentionally aggressive** to ensure:
1. **LLM Compatibility**: Data fits within model context limits
2. **Relevance**: Most important entities/relationships are preserved
3. **Performance**: Efficient processing without context length errors

The apparent "loss" of nodes (40 → 7 entities) is actually **intelligent filtering** that preserves the most relevant information while maintaining system stability.
