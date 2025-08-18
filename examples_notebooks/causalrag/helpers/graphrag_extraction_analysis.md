# GraphRAG Extraction Sequence and Top-K Dependencies Analysis

## **🔍 GraphRAG Extraction Sequence and Top-K Dependencies**

### **📋 Extraction Sequence (Standard Workflow)**

The GraphRAG indexing pipeline follows this **sequential order**:

```python
_standard_workflows = [
    "create_base_text_units",      # 1. Create text chunks
    "create_final_documents",      # 2. Process documents
    "extract_graph",               # 3. Extract entities & relationships
    "finalize_graph",              # 4. Clean up graph
    "extract_covariates",          # 5. Extract additional metadata
    "create_communities",          # 6. Cluster entities into communities
    "create_final_text_units",     # 7. Finalize text processing
    "create_community_reports",    # 8. Generate community summaries
    "generate_text_embeddings",    # 9. Create embeddings
]
```

### **🏗️ How Entities and Relationships Are Extracted**

#### **1. Entity Extraction (`extract_graph` workflow)**
- **Input**: Text units (chunks of documents)
- **Process**: Uses LLM to extract entities and relationships from each text unit
- **Output**: Raw entities and relationships with descriptions
- **Post-processing**: Summarizes entity and relationship descriptions

#### **2. Community Creation (`create_communities` workflow)**
- **Input**: Entities and relationships from previous step
- **Process**: 
  - Creates a graph from relationships
  - Applies clustering algorithms (e.g., Louvain)
  - Groups entities into hierarchical communities
  - **Key**: Communities are formed based on **graph structure**, not top-k values

#### **3. Community Reports (`create_community_reports` workflow)**
- **Input**: Communities, entities, relationships, and claims
- **Process**: 
  - Builds local context for each community
  - Uses LLM to generate summaries for each community
  - **Key**: Reports are generated for **all communities**, not limited by top-k

### **🔍 Top-K Dependencies in Search Context Building**

The **top-k values are used ONLY during search/query time**, not during the indexing pipeline:

#### **Entity Retrieval (`top_k_mapped_entities`)**
```python
# In LocalSearchMixedContext.build_context()
selected_entities = map_query_to_entities(
    query=query,
    text_embedding_vectorstore=self.entity_text_embeddings,
    text_embedder=self.text_embedder,
    all_entities_dict=self.entities,
    k=top_k_mapped_entities,  # ← Controls how many entities are retrieved
    oversample_scaler=2,
)
```

#### **Relationship Retrieval (`top_k_relationships`)**
```python
# In _filter_relationships()
relationship_budget = top_k_relationships * len(selected_entities)
return in_network_relationships + out_network_relationships[:relationship_budget]
```

#### **Community Report Retrieval**
```python
# Communities are selected based on entity matches, not top-k
community_matches = {}
for entity in selected_entities:
    if entity.community_ids:
        for community_id in entity.community_ids:
            community_matches[community_id] = (
                community_matches.get(community_id, 0) + 1
            )

# Sort by number of matched entities and rank
selected_communities = [
    self.community_reports[community_id]
    for community_id in community_matches
    if community_id in self.community_reports
]
```

### **🔑 Key Insights**

#### **1. Indexing vs. Search Separation**
- **Indexing**: Extracts **ALL** entities, relationships, and communities
- **Search**: Uses top-k to limit **retrieval** during query processing

#### **2. Community Reports Are NOT Limited by Top-K**
- Community reports are generated for **every community** during indexing
- During search, communities are selected based on **entity matches**, not top-k limits
- The `community_prop` parameter controls **context allocation**, not community selection

#### **3. Top-K Controls Retrieval Breadth**
- `top_k_mapped_entities`: Limits initial entity retrieval from vector store
- `top_k_relationships`: Limits relationship retrieval per entity
- These values affect **search performance** and **context quality**, not **indexing depth**

#### **4. Context Building Strategy**
```python
# Token allocation based on proportions
community_tokens = max(int(max_context_tokens * community_prop), 0)
local_tokens = max(int(max_context_tokens * local_prop), 0)
text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)
```

### **📊 Summary**

| Component | Extraction | Top-K Dependency |
|-----------|------------|------------------|
| **Entities** | All extracted during indexing | `top_k_mapped_entities` limits retrieval |
| **Relationships** | All extracted during indexing | `top_k_relationships` limits per-entity retrieval |
| **Communities** | All created during indexing | No top-k limit, based on graph structure |
| **Community Reports** | All generated during indexing | No top-k limit, based on entity matches |

**Bottom Line**: Top-k values control **search-time retrieval breadth**, not **indexing-time extraction depth**. The indexing pipeline extracts everything, while search uses top-k to efficiently retrieve the most relevant subset for each query.

## **🔍 Code Analysis Details**

### **Files Analyzed**
- `graphrag/index/workflows/factory.py` - Workflow sequence definition
- `graphrag/index/workflows/extract_graph.py` - Entity/relationship extraction
- `graphrag/index/workflows/create_communities.py` - Community clustering
- `graphrag/index/workflows/create_community_reports.py` - Community report generation
- `graphrag/query/structured_search/local_search/mixed_context.py` - Search context building
- `graphrag/query/context_builder/entity_extraction.py` - Entity retrieval logic
- `graphrag/query/context_builder/local_context.py` - Relationship filtering

### **Key Functions**
- `map_query_to_entities()` - Entity retrieval with top-k limit
- `_filter_relationships()` - Relationship filtering with top-k limit
- `_build_community_context()` - Community selection based on entity matches
- `build_context()` - Main context building orchestration

### **Configuration Parameters**
- `top_k_mapped_entities`: Controls entity retrieval breadth
- `top_k_relationships`: Controls relationship retrieval per entity
- `community_prop`: Controls context token allocation for communities
- `text_unit_prop`: Controls context token allocation for text units
- `max_context_tokens`: Overall context size limit
