# Causal Search Implementation Analysis

## Overview

The causal search implementation in GraphRAG is a two-stage LLM-based approach that generates responses to user queries through structured causal analysis. This document provides a detailed analysis of how it works based on the actual code implementation.

## Response Generation Flow

### 1. Overall Process (6 Steps)

The causal search follows a structured 6-step process:

1. **Extract Extended Nodes**: Get (k + s) * oversample_scaler entities
2. **Extract Graph Information**: Build context using local search components  
3. **Format Network Data**: Truncate and structure data for LLM consumption
4. **Generate Causal Report**: First LLM call using causal discovery prompt
5. **Generate Final Response**: Second LLM call using causal summary prompt
6. **Save Outputs**: Store network data and report files

### 2. Node Extraction Strategy (Corrected Implementation)

**Formula**: Total nodes = (k + s) * oversample_scaler
- `k` = top_k_mapped_entities (default: 10)
- `s` = s_parameter (default: 3) 
- `oversample_scaler` = 2 (hardcoded)
- **Result**: (10 + 3) * 2 = 26 total nodes requested

**Single-phase extraction** (corrected logic):
1. **Calculate total nodes**: (k + s) * oversample_scaler = 26 nodes
2. **Request all nodes at once**: Use context builder with overridden top_k_mapped_entities
3. **Use ALL extracted nodes**: No artificial truncation to k (previous bug fixed)
4. **Apply token filtering**: Final filtering during context formatting phase

#### **Actual Results**: 
- **Nodes requested**: 26 (calculated)
- **Nodes extracted**: 42+ (context builder may return more based on query relevance)
- **Final context**: Determined by token limits (8K budget) during formatting
- **Improvement**: Much better entity coverage vs. previous flawed 13-node limit

## Context and Length Control

### 1. Token Allocation Strategy

**Network Data Token Limits** (8,000 total tokens):
- **Entities**: 40% (3,200 tokens)
- **Relationships**: 40% (3,200 tokens)
- **Text Units**: 20% (1,600 tokens)

### 2. Field Selection (Essential Fields Only)

**Entities**:
```python
essential_fields = ['id', 'entity', 'description', 'rank', 'type']
```

**Relationships**:
```python
essential_fields = ['id', 'source', 'target', 'description', 'weight', 'rank']
```

**Text Units**:
```python
essential_fields = ['id', 'text', 'n_tokens']
# Text truncated at 1000 characters with "..." if longer
```

### 3. Prioritization Rules

- **Entities**: Sorted by rank (highest first)
- **Relationships**: Sorted by weight/rank (highest first)
- **Text Units**: Sorted by n_tokens (shortest first to fit more)

### 4. Progressive Truncation

The system uses real-time token counting (`num_tokens`) and stops adding records when approaching token limits, ensuring the most important data is preserved within constraints.

## Prompts Used

### 1. Stage 1: Causal Discovery Prompt

**Location**: `graphrag/prompts/query/causal_discovery_prompt.py`

**Template Variables**:
- `{graph_data}` - The formatted network data JSON

**Structure**:
- Role: Smart assistant for causal discovery and impact assessment
- Goal: Analyze network data and generate professional causality analysis report
- Format: 5-section structured report:
  1. Introduction
  2. Key Entities and Their Roles  
  3. Major Causal Pathways
  4. Confidence and Evidence Strength
  5. Implications and Recommendations

### 2. Stage 2: Causal Summary Prompt

**Location**: `graphrag/prompts/query/causal_summary_prompt.py`

**Template Variables**:
- `{causal_summary}` - Output from Stage 1
- `{query}` - Original user query
- `{response_type}` - Response format (default: "Multiple Paragraphs")

**Structure**:
- Role: Assistant specializing in causal reasoning and impact assessment
- Goal: Generate structured response based on causal summary
- Requirements: Preserve original meaning, remove irrelevant info, merge into comprehensive answer

## LLM Interaction Details

### 1. Model Parameters

**Configuration**:
- `max_context_tokens`: 12,000 (default)
- `response_type`: "Multiple Paragraphs" (default)
- `model_params`: Passed through from configuration

### 2. LLM Call Structure

**Stage 1 (Causal Discovery)**:
```python
response = await self.model.achat(
    prompt="Generate a causal analysis report",
    history=[{"role": "system", "content": formatted_prompt}],
    model_parameters=self.model_params
)
```

**Stage 2 (Final Response)**:
```python
response = await self.model.achat(
    prompt=user_query,  # Original user query
    history=[{"role": "system", "content": formatted_prompt}],
    model_parameters=self.model_params
)
```

### 3. Content Extraction

The implementation handles both direct content access and fallback extraction from BaseModelOutput string representations:

```python
if hasattr(response, 'content') and response.content:
    response_content = response.content
else:
    # Fallback: Extract from string representation using regex
    match = re.search(r"content='(.*?)', full_response=", response_str, re.DOTALL)
```

## Context Building Process

### 1. Graph Context Extraction

Uses `LocalSearchMixedContext.build_context()` with parameters:
- Query-based entity mapping
- Token-based truncation ("Reached token limit - reverting to previous context state")
- Mixed community reports, entities, and source data

### 2. Context Records Structure

**Available keys**: `['reports', 'entities', 'sources']`
- **reports**: Community reports data
- **entities**: Entity information 
- **sources**: Text unit sources
- **Note**: Relationships and text_units often empty in context records

### 3. Fallback Data Sources

When context records are incomplete, the system falls back to context builder direct access:
- `self.context_builder.entities` (191 relationships available)
- `self.context_builder.relationships` (42 text units available)  
- `self.context_builder.text_units`

## Token Usage and Performance

### 1. Token Calculation

**Input Tokens**:
- `causal_discovery`: Network data tokens
- `response_generation`: Causal report tokens

**Output Tokens**:
- `causal_discovery`: Causal report tokens
- `response_generation`: Final response tokens

### 2. Typical Results

Based on actual execution:
- **Network data**: ~85KB (down from 465KB without truncation)
- **Entities**: 7 (all fit within limit)
- **Relationships**: 9 (from 191 total, truncated)
- **Text units**: 4 (from 42 total, truncated)
- **Execution time**: ~42 seconds for complete pipeline

## Output Generation

### 1. File Structure

**Network Data**: `causal_search_network_data_{query_id}.json`
```json
{
  "entities": [...],
  "relationships": [...], 
  "text_units": [...],
  "community_reports": [...],
  "context_summary": "..."
}
```

**Report**: `causal_search_report_{query_id}.md`
- Clean markdown format (BaseModelOutput extraction implemented)
- Structured causal analysis report
- Query metadata and generation timestamp

### 2. Query ID Format

Format: `{hash}_{timestamp}`
- **Hash**: First 8 characters of query content hash
- **Timestamp**: Unix timestamp when query was processed
- **Example**: `6c867dce_1755478569`

## Configuration Parameters
◊
### 1. Key Parameters

- `s_parameter`: 3 (additional nodes for causal analysis)
- `max_context_tokens`: 12,000 (context builder limit)
- `response_type`: "Multiple Paragraphs" (LLM output format)
- `save_network_data`: true (file output control)
- `save_causal_report`: true (file output control)

### 2. Model Configuration (settings.yaml)

**Critical**: The `max_tokens` parameter must be set appropriately in the model configuration:

```yaml
models:
  default_chat_model:
    max_tokens: 4000  # Must match model's completion token limits
    temperature: 0.0   # Consistent with other search methods
```

**Model-specific limits**:
- `gpt-4-turbo-preview`: 4,096 completion tokens (set to 4,000)
- `gpt-4`: 8,192 completion tokens (set to 8,000)
- Other models: Check provider documentation

**Error symptoms without proper configuration**:
- `max_tokens is too large: 16000. This model supports at most 4096 completion tokens`
- Caused by missing `max_tokens` in settings.yaml, defaulting to large values

### 3. Token Limits

- **Network data total**: 8,000 tokens (conservative for LLM context)
- **Entity allocation**: 3,200 tokens (40%)
- **Relationship allocation**: 3,200 tokens (40%)  
- **Text unit allocation**: 1,600 tokens (20%)

## Error Handling

### 1. Context Length Management

- Progressive truncation prevents "context_length_exceeded" errors
- Token counting with real-time monitoring
- Fallback to most important data when limits reached

### 2. Content Extraction Robustness

- Multiple extraction methods for LLM responses
- Regex-based fallback for BaseModelOutput format
- Comprehensive logging for debugging

## Summary

The causal search implementation provides a sophisticated two-stage approach to causal analysis that:

1. **Intelligently selects** the most relevant entities and relationships using query-based mapping and ranking
2. **Truncates context** using token-aware prioritization to fit within LLM limits  
3. **Generates structured causal analysis** through specialized prompts and two-stage LLM processing
4. **Produces comprehensive reports** with both detailed analysis and user-friendly responses
5. **Handles edge cases** with robust content extraction and error handling

The system successfully balances comprehensive analysis with practical constraints, making it suitable for production use with large knowledge graphs.