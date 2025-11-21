---
title: GraphRAG Analysis Helpers
description: Helper functions for GraphRAG analysis notebooks.
author: Chuan Xu
date: 2025-11-30
last_updated: 2025-11-20
version: 0.0.1
---

# GraphRAG Analysis Helpers

This directory contains helper functions extracted from the GraphRAG analysis notebooks to streamline the notebook code and make it more maintainable.

## Files

- **`analysis_helpers.py`**: Core helper functions for GraphRAG analysis including:
  - Embedding calculations and cosine similarity
  - Entity analysis with similarity scores
  - Relationship classification (in-network vs out-network)
  - Map-reduce analysis visualization
  - Token counting utilities

## Usage in Notebooks

### Setup

Add this cell near the top of your notebook to import the helpers:

```python
import sys
from pathlib import Path

# Add the helpers directory to path
helpers_dir = Path.cwd()
if str(helpers_dir) not in sys.path:
    sys.path.insert(0, str(helpers_dir))

from analysis_helpers import (
    get_embedding,
    cosine_similarity,
    calculate_entity_similarities,
    print_entities_analysis,
    classify_relationships,
    print_relationships_analysis,
    analyze_map_reduce_results,
    count_tokens,
    analyze_context_tokens,
    analyze_text_chunk_sizes
)
```

### Example: Entity Analysis

```python
# Calculate similarities
entities_df = calculate_entity_similarities(
    entities_df=local_context['Entities'],
    query=test_question,
    embedding_client=embedding_client,
    deployment_name=EMBEDDING_DEPLOYMENT
)

# Print analysis
print_entities_analysis(
    entities_df=entities_df,
    relationships_df=local_context.get('Relationships'),
    top_n=20
)
```

### Example: Relationship Analysis

```python
# Get retrieved entity names
retrieved_entities = set(local_context['Entities']['entity'].tolist())

# Print relationships with classification
print_relationships_analysis(
    relationships_df=local_context['Relationships'],
    retrieved_entities=retrieved_entities,
    top_n=15
)
```

### Example: Map-Reduce Analysis

```python
# Analyze global search map-reduce results
analyze_map_reduce_results(
    result=global_search_result,  # GlobalSearchResult object
    query=test_question
)
```

### Example: Token Counting

```python
# Count tokens in context
token_counts = analyze_context_tokens(
    context_data=local_context,
    encoding_name="cl100k_base"
)

print("Token usage by component:")
for component, count in token_counts.items():
    print(f"  {component}: {count:,} tokens")
```

### Example: Text Chunk Analysis

```python
# Analyze text chunk sizes and token distributions
analyze_text_chunk_sizes(
    text_units_df=text_units,
    text_column='text',
    id_column='id',
    encoding_name='cl100k_base'
)
```

## Benefits

1. **Cleaner Notebooks**: Removes hundreds of lines of helper code from notebooks
2. **Reusability**: Same functions can be used across multiple notebooks
3. **Maintainability**: Single place to update and fix helper functions
4. **Better Documentation**: Functions have proper docstrings
5. **Type Hints**: All functions have type annotations for better IDE support

## Migration Guide

To migrate an existing notebook:

1. Replace the inline helper function definitions with the import statement above
2. Replace calls to inline functions with calls to the imported functions
3. Remove the old helper function code blocks
4. Update any function calls if parameter names changed (check docstrings)

## Functions Reference

### Embedding & Similarity

- `get_embedding(text, client, deployment_name)` - Get embedding vector for text
- `cosine_similarity(vec1, vec2)` - Calculate cosine similarity between vectors

### Entity Analysis

- `calculate_entity_similarities(entities_df, query, embedding_client, deployment_name)` - Calculate similarity scores
- `print_entities_analysis(entities_df, relationships_df, top_n)` - Print detailed entity analysis

### Relationship Analysis

- `classify_relationships(relationships_df, retrieved_entities)` - Classify as in/out-network
- `print_relationships_analysis(relationships_df, retrieved_entities, top_n)` - Print relationship details

### Map-Reduce Analysis

- `analyze_map_reduce_results(result, query)` - Comprehensive map-reduce visualization

### Token Counting

- `count_tokens(text, encoding_name)` - Count tokens in text
- `analyze_context_tokens(context_data, encoding_name)` - Analyze tokens per component

### Text Chunk Analysis

- `analyze_text_chunk_sizes(text_units_df, text_column, id_column, encoding_name)` - Analyze text chunk sizes and token statistics
