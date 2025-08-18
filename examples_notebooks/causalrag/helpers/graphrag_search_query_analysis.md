# GraphRAG Search/Query Time Analysis: Sequence, Relationships, and Token Budget

## **🔍 Overview**

This document provides a detailed analysis of what happens during **search/query time** in GraphRAG, focusing on the sequence of operations, the concept of in_network_relationships, and how token budget allocation works.

## **📋 Search/Query Time Sequence**

### **1. Query Processing and Entity Retrieval**

#### **Initial Query Handling**
```python
# In LocalSearchMixedContext.build_context()
if conversation_history:
    pre_user_questions = "\n".join(
        conversation_history.get_user_turns(conversation_history_max_turns)
    )
    query = f"{query}\n{pre_user_questions}"

# Map query to entities using vector similarity
selected_entities = map_query_to_entities(
    query=query,
    text_embedding_vectorstore=self.entity_text_embeddings,
    text_embedder=self.text_embedder,
    all_entities_dict=self.entities,
    k=top_k_mapped_entities,  # ← Controls initial entity retrieval breadth
    oversample_scaler=2,      # ← Retrieves 2x more entities initially for filtering
)
```

**What Happens:**
1. **Query Enhancement**: If conversation history exists, previous user questions are appended
2. **Entity Mapping**: Query is embedded and compared against entity descriptions
3. **Top-K Selection**: `top_k_mapped_entities` determines how many entities are initially retrieved
4. **Oversampling**: 2x more entities are retrieved initially to account for potential exclusions

#### **Entity Selection Process**
```python
# In entity_extraction.py
def map_query_to_entities(query, text_embedding_vectorstore, text_embedder, 
                         all_entities_dict, k=10, oversample_scaler=2):
    if query != "":
        # Get entities with highest semantic similarity to query
        search_results = text_embedding_vectorstore.similarity_search_by_text(
            text=query,
            text_embedder=lambda t: text_embedder.embed(t),
            k=k * oversample_scaler,  # ← Retrieves k * 2 entities
        )
        
        # Map results back to entity objects
        for result in search_results:
            matched = get_entity_by_id(all_entities_dict, result.document.id)
            if matched:
                matched_entities.append(matched)
    
    # Filter out excluded entities and add included ones
    if exclude_entity_names:
        matched_entities = [entity for entity in matched_entities 
                          if entity.title not in exclude_entity_names]
    
    # Add explicitly included entities
    for entity_name in include_entity_names:
        included_entities.extend(get_entity_by_name(all_entities, entity_name))
    
    return included_entities + matched_entities
```

### **2. Context Building Sequence**

#### **Token Budget Allocation**
```python
# Calculate token budgets for each context type
community_tokens = max(int(max_context_tokens * community_prop), 0)
local_tokens = max(int(max_context_tokens * local_prop), 0)
text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)

# local_prop is calculated as: 1 - community_prop - text_unit_prop
```

**Token Allocation Strategy:**
- **Community Context**: `community_prop` × `max_context_tokens`
- **Local Context**: `(1 - community_prop - text_unit_prop)` × `max_context_tokens`
- **Text Unit Context**: `text_unit_prop` × `max_context_tokens`
- **Total**: Must equal `max_context_tokens`

#### **Context Building Order**
```python
# 1. Conversation History (if exists)
if conversation_history:
    conversation_history_context = conversation_history.build_context(...)
    final_context.append(conversation_history_context)
    max_context_tokens -= num_tokens(conversation_history_context)

# 2. Community Context
community_context, community_context_data = self._build_community_context(
    selected_entities=selected_entities,
    max_context_tokens=community_tokens,  # ← Pre-allocated budget
    use_community_summary=use_community_summary,
    include_community_rank=include_community_rank,
)

# 3. Local Context (Entities + Relationships + Covariates)
local_context, local_context_data = self._build_local_context(
    selected_entities=selected_entities,
    max_context_tokens=local_tokens,      # ← Pre-allocated budget
    top_k_relationships=top_k_relationships,
    include_entity_rank=include_entity_rank,
)

# 4. Text Unit Context
text_unit_context, text_unit_context_data = self._build_text_unit_context(
    selected_entities=selected_entities,
    max_context_tokens=text_unit_tokens,  # ← Pre-allocated budget
)
```

## **🔗 In-Network vs Out-Network Relationships**

### **Relationship Classification System**

#### **1. In-Network Relationships**
```python
def get_in_network_relationships(selected_entities, relationships, ranking_attribute="rank"):
    """Get all directed relationships BETWEEN selected entities."""
    selected_entity_names = [entity.title for entity in selected_entities]
    selected_relationships = [
        relationship
        for relationship in relationships
        if relationship.source in selected_entity_names
        and relationship.target in selected_entity_names  # ← BOTH entities are selected
    ]
    return sort_relationships_by_rank(selected_relationships, ranking_attribute)
```

**Characteristics:**
- **Source AND target** are both in the selected entity set
- **Highest priority** in relationship selection
- **Direct connections** between retrieved entities
- **Represents core network structure** of the query

**Example:**
```
Selected Entities: [Entity A, Entity B, Entity C]
In-Network Relationships:
- A → B (relationship between two selected entities)
- B → C (relationship between two selected entities)
- C → A (relationship between two selected entities)
```

#### **2. Out-Network Relationships**
```python
def get_out_network_relationships(selected_entities, relationships, ranking_attribute="rank"):
    """Get relationships FROM selected entities TO other entities."""
    selected_entity_names = [entity.title for entity in selected_entities]
    
    # Source relationships: selected entity → external entity
    source_relationships = [
        relationship
        for relationship in relationships
        if relationship.source in selected_entity_names
        and relationship.target not in selected_entity_names
    ]
    
    # Target relationships: external entity → selected entity
    target_relationships = [
        relationship
        for relationship in relationships
        if relationship.target in selected_entities
        and relationship.source not in selected_entity_names
    ]
    
    return sort_relationships_by_rank(source_relationships + target_relationships, ranking_attribute)
```

**Characteristics:**
- **One entity** is in the selected set, **one is external**
- **Second priority** in relationship selection
- **Expands context** beyond directly selected entities
- **Provides broader network perspective**

**Example:**
```
Selected Entities: [Entity A, Entity B]
Out-Network Relationships:
- A → X (A connects to external entity X)
- Y → B (External entity Y connects to B)
- A → Z (A connects to external entity Z)
```

### **3. Relationship Prioritization and Selection**

#### **Priority Order**
```python
def _filter_relationships(selected_entities, relationships, top_k_relationships, ranking_attribute):
    # 1. FIRST PRIORITY: All in-network relationships
    in_network_relationships = get_in_network_relationships(...)
    
    # 2. SECOND PRIORITY: Out-network relationships (limited by budget)
    out_network_relationships = get_out_network_relationships(...)
    
    # 3. Calculate relationship budget
    relationship_budget = top_k_relationships * len(selected_entities)
    
    # 4. Return in-network + limited out-network
    return in_network_relationships + out_network_relationships[:relationship_budget]
```

#### **Out-Network Relationship Scoring**
```python
# Score out-network relationships by connectivity
out_network_entity_links = defaultdict(int)
for entity_name in out_network_entity_names:
    targets = [rel.target for rel in out_network_relationships if rel.source == entity_name]
    sources = [rel.source for rel in out_network_relationships if rel.target == entity_name]
    out_network_entity_links[entity_name] = len(set(targets + sources))

# Sort by connectivity first, then by ranking attribute
for rel in out_network_relationships:
    rel.attributes["links"] = out_network_entity_links[rel.source or rel.target]

# Sort by (connectivity, ranking_attribute) in descending order
out_network_relationships.sort(
    key=lambda x: (x.attributes["links"], x.rank), 
    reverse=True
)
```

**Scoring Logic:**
1. **Connectivity Score**: How many relationships an external entity has with selected entities
2. **Ranking Attribute**: Secondary sorting by rank, weight, or custom attribute
3. **Higher connectivity** = **Higher priority** for inclusion

## **💰 Token Budget Management**

### **1. Hierarchical Token Allocation**

#### **Initial Budget Distribution**
```python
# Total available tokens
total_tokens = max_context_tokens

# Proportional allocation
community_tokens = max(int(total_tokens * community_prop), 0)
local_tokens = max(int(total_tokens * local_prop), 0)
text_unit_tokens = max(int(total_tokens * text_unit_prop), 0)

# Validation
if community_prop + text_unit_prop > 1:
    raise ValueError("The sum of community_prop and text_unit_prop should not exceed 1.")
```

#### **Dynamic Budget Adjustment**
```python
# Conversation history consumes tokens first
if conversation_history_context.strip() != "":
    final_context.append(conversation_history_context)
    max_context_tokens = max_context_tokens - num_tokens(conversation_history_context)

# Remaining budget is distributed proportionally
remaining_tokens = max_context_tokens
community_tokens = max(int(remaining_tokens * community_prop), 0)
local_tokens = max(int(remaining_tokens * local_prop), 0)
text_unit_tokens = max(int(remaining_tokens * text_unit_prop), 0)
```

### **2. Per-Context Token Management**

#### **Entity Context Token Counting**
```python
def build_entity_context(selected_entities, max_context_tokens, token_encoder):
    current_context_text = f"-----Entities-----\n"
    header = ["id", "entity", "description", "rank"]
    current_context_text += "|".join(header) + "\n"
    
    current_tokens = num_tokens(current_context_text, token_encoder)
    
    for entity in selected_entities:
        new_context_text = f"{entity.id}|{entity.title}|{entity.description}|{entity.rank}\n"
        new_tokens = num_tokens(new_context_text, token_encoder)
        
        # Check if adding this entity exceeds budget
        if current_tokens + new_tokens > max_context_tokens:
            break  # Stop adding entities
            
        current_context_text += new_context_text
        current_tokens += new_tokens
    
    return current_context_text
```

#### **Relationship Context Token Counting**
```python
def build_relationship_context(selected_relationships, max_context_tokens, token_encoder):
    current_context_text = f"-----Relationships-----\n"
    header = ["id", "source", "target", "description", "weight"]
    current_context_text += "|".join(header) + "\n"
    
    current_tokens = num_tokens(current_context_text, token_encoder)
    
    for rel in selected_relationships:
        new_context_text = f"{rel.id}|{rel.source}|{rel.target}|{rel.description}|{rel.weight}\n"
        new_tokens = num_tokens(new_context_text, token_encoder)
        
        # Check token budget
        if current_tokens + new_tokens > max_context_tokens:
            break  # Stop adding relationships
            
        current_context_text += new_context_text
        current_tokens += new_tokens
    
    return current_context_text
```

### **3. Token Budget Optimization Strategies**

#### **Proportional Allocation Example**
```python
# Example configuration
max_context_tokens = 12000
community_prop = 0.25      # 25% for community reports
text_unit_prop = 0.5       # 50% for text units
local_prop = 0.25          # 25% for entities/relationships (calculated)

# Token allocation
community_tokens = 12000 * 0.25 = 3000 tokens
text_unit_tokens = 12000 * 0.5 = 6000 tokens  
local_tokens = 12000 * 0.25 = 3000 tokens

# Total: 3000 + 6000 + 3000 = 12000 tokens ✓
```

#### **Adaptive Budget Adjustment**
```python
# If community context uses fewer tokens than allocated
actual_community_tokens = num_tokens(community_context, token_encoder)
unused_community_tokens = community_tokens - actual_community_tokens

# Redistribute unused tokens to other contexts
if unused_community_tokens > 0:
    local_tokens += unused_community_tokens
    text_unit_tokens += unused_community_tokens
```

## **🔄 Complete Search/Query Flow**

### **Step-by-Step Process**

```python
def build_context(query, max_context_tokens, **kwargs):
    # STEP 1: Entity Retrieval
    selected_entities = map_query_to_entities(
        query=query,
        k=top_k_mapped_entities,  # e.g., 10 entities
        oversample_scaler=2       # Retrieve 20, filter to 10
    )
    
    # STEP 2: Token Budget Allocation
    community_tokens = max_context_tokens * community_prop      # e.g., 3000
    local_tokens = max_context_tokens * local_prop             # e.g., 3000  
    text_unit_tokens = max_context_tokens * text_unit_prop     # e.g., 6000
    
    # STEP 3: Community Context Building
    community_context = _build_community_context(
        selected_entities=selected_entities,
        max_context_tokens=community_tokens
    )
    
    # STEP 4: Local Context Building
    local_context = _build_local_context(
        selected_entities=selected_entities,
        max_context_tokens=local_tokens,
        top_k_relationships=top_k_relationships  # e.g., 10 per entity
    )
    
    # STEP 5: Text Unit Context Building
    text_unit_context = _build_text_unit_context(
        selected_entities=selected_entities,
        max_context_tokens=text_unit_tokens
    )
    
    # STEP 6: Context Assembly
    final_context = [
        community_context,
        local_context, 
        text_unit_context
    ]
    
    return ContextBuilderResult(
        context_chunks="\n\n".join(final_context),
        context_records=final_context_data
    )
```

### **Relationship Selection Process**

```python
def _build_local_context(selected_entities, max_context_tokens, top_k_relationships):
    # 1. Get in-network relationships (highest priority)
    in_network = get_in_network_relationships(selected_entities, relationships)
    
    # 2. Get out-network relationships (second priority)
    out_network = get_out_network_relationships(selected_entities, relationships)
    
    # 3. Score and sort out-network relationships
    for rel in out_network:
        rel.attributes["links"] = count_connections_to_selected_entities(rel)
    
    out_network.sort(key=lambda x: (x.attributes["links"], x.rank), reverse=True)
    
    # 4. Apply relationship budget
    relationship_budget = top_k_relationships * len(selected_entities)
    selected_relationships = in_network + out_network[:relationship_budget]
    
    # 5. Build context within token budget
    return build_relationship_context(selected_relationships, max_context_tokens)
```

## **🎯 Key Insights**

### **1. Priority Hierarchy**
- **In-Network Relationships**: Always included (highest priority)
- **Out-Network Relationships**: Included based on connectivity + ranking (second priority)
- **Token Budget**: Enforces limits on context size

### **2. Token Budget Strategy**
- **Proportional Allocation**: Pre-allocates tokens to each context type
- **Dynamic Adjustment**: Adapts based on actual content size
- **Efficient Usage**: Stops adding content when budget is reached

### **3. Relationship Selection Logic**
- **Connectivity Scoring**: External entities with more connections get higher priority
- **Ranking Attributes**: Secondary sorting by rank, weight, or custom attributes
- **Budget Enforcement**: `top_k_relationships * entity_count` limits total relationships

### **4. Context Building Efficiency**
- **Sequential Processing**: Each context type is built independently
- **Token Counting**: Real-time token usage tracking
- **Early Termination**: Stops adding content when budget is exceeded

## **📊 Configuration Parameters Impact**

| Parameter | Effect on Search/Query Time |
|-----------|------------------------------|
| `top_k_mapped_entities` | Controls initial entity retrieval breadth |
| `top_k_relationships` | Limits relationships per entity × entity count |
| `community_prop` | Allocates token budget for community reports |
| `text_unit_prop` | Allocates token budget for source documents |
| `max_context_tokens` | Overall context size limit |
| `relationship_ranking_attribute` | Determines relationship sorting priority |

## **🔍 Code Locations**

### **Key Files Analyzed**
- `graphrag/query/structured_search/local_search/mixed_context.py` - Main context building orchestration
- `graphrag/query/context_builder/local_context.py` - Local context building and relationship filtering
- `graphrag/query/input/retrieval/relationships.py` - Relationship retrieval and classification
- `graphrag/query/context_builder/entity_extraction.py` - Entity retrieval and mapping

### **Key Functions**
- `LocalSearchMixedContext.build_context()` - Main orchestration
- `_filter_relationships()` - Relationship prioritization and selection
- `get_in_network_relationships()` - In-network relationship extraction
- `get_out_network_relationships()` - Out-network relationship extraction
- `build_entity_context()` - Entity context building with token counting
- `build_relationship_context()` - Relationship context building with token counting

This analysis provides a comprehensive understanding of how GraphRAG processes search queries, manages relationships, and allocates token budgets during context building.
