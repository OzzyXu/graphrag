"""
Helper functions for GraphRAG analysis notebooks.

This module contains utility functions for:
- Embedding calculations and similarity
- Context analysis and visualization
- Token counting and statistics
- Map-reduce analysis
"""

import pandas as pd
import numpy as np
import tiktoken
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI


def get_embedding(text: str, client: AzureOpenAI, deployment_name: str) -> List[float]:
    """
    Get embedding for a text using Azure OpenAI's API.
    
    Args:
        text: The text to embed
        client: Azure OpenAI client instance
        deployment_name: Name of the embedding deployment
        
    Returns:
        List of floats representing the embedding vector
    """
    response = client.embeddings.create(
        model=deployment_name,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (float between -1 and 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return float(similarity)


def calculate_entity_similarities(
    entities_df: pd.DataFrame,
    query: str,
    embedding_client: AzureOpenAI,
    deployment_name: str
) -> pd.DataFrame:
    """
    Calculate similarity scores for entities against a query.
    
    Args:
        entities_df: DataFrame containing entities
        query: Query string to compare against
        embedding_client: Azure OpenAI client for embeddings
        deployment_name: Name of the embedding deployment
        
    Returns:
        DataFrame with added 'calculated_similarity' column
    """
    print(f"🔄 Calculating entity similarity scores to query...")
    query_embedding = get_embedding(query, embedding_client, deployment_name)
    
    entity_similarities = []
    for idx, row in entities_df.iterrows():
        # Use entity name + description for similarity calculation
        entity_name = row.get('entity', row.get('title', row.get('name', '')))
        entity_desc = row.get('description', row.get('entity_description', ''))
        
        # Combine name and description for better matching
        entity_text = f"{entity_name}: {entity_desc}" if entity_desc and str(entity_desc) != 'nan' else entity_name
        
        if entity_text:
            entity_embedding = get_embedding(str(entity_text), embedding_client, deployment_name)
            similarity = cosine_similarity(query_embedding, entity_embedding)
            entity_similarities.append(similarity)
        else:
            entity_similarities.append(0.0)
    
    entities_df['calculated_similarity'] = entity_similarities
    print(f"✅ Similarity scores calculated!")
    
    return entities_df


def print_entities_analysis(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame = None,
    top_n: int = 20
):
    """
    Print detailed analysis of entities with similarity scores and connections.
    
    Args:
        entities_df: DataFrame with entities (must have 'calculated_similarity' column)
        relationships_df: Optional DataFrame with relationships for degree calculation
        top_n: Number of top entities to display
    """
    # Calculate degree from relationships if available
    if relationships_df is not None and len(relationships_df) > 0:
        entity_names = entities_df['entity'].tolist() if 'entity' in entities_df.columns else []
        
        # Calculate degree for each entity
        degree_dict = {}
        for entity_name in entity_names:
            degree = 0
            if 'source' in relationships_df.columns and 'target' in relationships_df.columns:
                degree = len(relationships_df[
                    (relationships_df['source'] == entity_name) | 
                    (relationships_df['target'] == entity_name)
                ])
            degree_dict[entity_name] = degree
        
        entities_df['calculated_degree'] = entities_df['entity'].map(degree_dict) if 'entity' in entities_df.columns else 0
    
    # Sort by calculated similarity
    entities_sorted = entities_df.sort_values('calculated_similarity', ascending=False)
    
    print(f"\n{'─' * 100}")
    print("TOP ENTITIES (sorted by similarity to query)")
    print(f"{'─' * 100}\n")
    
    for idx, row in entities_sorted.head(top_n).iterrows():
        entity_name = row.get('entity', row.get('title', row.get('name', 'Unknown')))
        
        # Get all available scores
        calc_similarity = row.get('calculated_similarity', 0.0)
        rank = row.get('rank', 'N/A')
        score = row.get('score', row.get('similarity', 'N/A'))
        degree = row.get('calculated_degree', row.get('degree', row.get('entity_degree', 'N/A')))
        
        print(f"📌 {entity_name}")
        print(f"   ├─ Calculated Similarity: {calc_similarity:.6f}")
        
        # Visual bar for similarity
        bar_length = int(calc_similarity * 50)
        print(f"   │  {'█' * bar_length}{'░' * (50 - bar_length)} {calc_similarity:.4f}")
        
        if rank != 'N/A':
            print(f"   ├─ Rank: {rank}")
        if score != 'N/A':
            print(f"   ├─ Score: {score}")
        print(f"   ├─ Degree (connections): {degree}")
        
        # Get description if available
        desc = row.get('description', row.get('entity_description', ''))
        if desc and str(desc) != 'nan':
            desc_preview = str(desc)[:150] + "..." if len(str(desc)) > 150 else str(desc)
            print(f"   └─ Description: {desc_preview}")
        else:
            print(f"   └─ Description: N/A")
        print()
    
    # Statistics
    print(f"{'─' * 100}")
    print("ENTITY STATISTICS")
    print(f"{'─' * 100}")
    
    print(f"   • Similarity range: {entities_df['calculated_similarity'].min():.6f} to {entities_df['calculated_similarity'].max():.6f}")
    print(f"   • Mean similarity: {entities_df['calculated_similarity'].mean():.6f}")
    print(f"   • Median similarity: {entities_df['calculated_similarity'].median():.6f}")
    
    if 'calculated_degree' in entities_df.columns:
        degree_numeric = pd.to_numeric(entities_df['calculated_degree'], errors='coerce')
        if degree_numeric.notna().any():
            print(f"   • Degree range: {int(degree_numeric.min())} to {int(degree_numeric.max())}")
            print(f"   • Mean degree: {degree_numeric.mean():.2f}")
    
    if 'rank' in entities_df.columns:
        rank_numeric = pd.to_numeric(entities_df['rank'], errors='coerce')
        if rank_numeric.notna().any():
            print(f"   • Rank range: {rank_numeric.min():.4f} to {rank_numeric.max():.4f}")
            print(f"   • Mean rank: {rank_numeric.mean():.4f}")
            print(f"   • Median rank: {rank_numeric.median():.4f}")
    
    print()


def classify_relationships(
    relationships_df: pd.DataFrame,
    retrieved_entities: set
) -> pd.DataFrame:
    """
    Classify relationships as in-network or out-network.
    
    Args:
        relationships_df: DataFrame with relationships
        retrieved_entities: Set of entity names that were retrieved
        
    Returns:
        DataFrame with added 'network_type' column
    """
    def classify_relationship(row):
        source = row.get('source', row.get('source_entity', ''))
        target = row.get('target', row.get('target_entity', ''))
        
        source_in = source in retrieved_entities
        target_in = target in retrieved_entities
        
        if source_in and target_in:
            return 'in-network'
        else:
            return 'out-network'
    
    relationships_df['network_type'] = relationships_df.apply(classify_relationship, axis=1)
    return relationships_df


def print_relationships_analysis(
    relationships_df: pd.DataFrame,
    retrieved_entities: set = None,
    top_n: int = 15
):
    """
    Print detailed analysis of relationships.
    
    Args:
        relationships_df: DataFrame with relationships
        retrieved_entities: Optional set of retrieved entity names for classification
        top_n: Number of top relationships to display
    """
    print("=" * 100)
    print("🔗 RELATIONSHIPS IN CONTEXT")
    print("=" * 100)
    print(f"\nTotal relationships retrieved: {len(relationships_df)}")
    print(f"Available columns: {list(relationships_df.columns)}")
    
    # Classify relationships if we have retrieved entities
    if retrieved_entities is not None:
        relationships_df = classify_relationships(relationships_df, retrieved_entities)
        
        in_network_rels = relationships_df[relationships_df['network_type'] == 'in-network']
        out_network_rels = relationships_df[relationships_df['network_type'] == 'out-network']
        
        print(f"\n📊 Relationship Classification:")
        print(f"   • In-Network (both entities retrieved): {len(in_network_rels)} ({len(in_network_rels)/len(relationships_df)*100:.1f}%)")
        print(f"   • Out-Network (at least one external entity): {len(out_network_rels)} ({len(out_network_rels)/len(relationships_df)*100:.1f}%)")
    
    # Sort by rank/weight
    if 'rank' in relationships_df.columns:
        rels_sorted = relationships_df.sort_values('rank', ascending=False)
    elif 'weight' in relationships_df.columns:
        rels_sorted = relationships_df.sort_values('weight', ascending=False)
    else:
        rels_sorted = relationships_df
    
    print(f"\n{'─' * 100}")
    print("TOP RELATIONSHIPS (ALL)")
    print(f"{'─' * 100}\n")
    
    for idx, row in rels_sorted.head(top_n).iterrows():
        source = row.get('source', row.get('source_entity', 'Unknown'))
        target = row.get('target', row.get('target_entity', 'Unknown'))
        weight = row.get('weight', 'N/A')
        rank = row.get('rank', 'N/A')
        network_type = row.get('network_type', 'unknown')
        
        # Use different emoji based on network type
        if network_type == 'in-network':
            emoji = "🔗"
            type_label = "[IN-NETWORK]"
        elif network_type == 'out-network':
            emoji = "🔀"
            type_label = "[OUT-NETWORK]"
        else:
            emoji = "🔗"
            type_label = ""
        
        print(f"{emoji} {source} ──→ {target} {type_label}")
        print(f"   ├─ Weight: {weight}")
        print(f"   ├─ Rank: {rank}")
        
        desc = row.get('description', row.get('relationship_description', ''))
        if desc and str(desc) != 'nan':
            desc_preview = str(desc)[:150] + "..." if len(str(desc)) > 150 else str(desc)
            print(f"   └─ Description: {desc_preview}")
        else:
            print(f"   └─ Description: N/A")
        print()


def analyze_map_reduce_results(result, query: str):
    """
    Analyze and print detailed map-reduce results from global search.
    
    Args:
        result: GlobalSearchResult object with map_responses
        query: The query string that was used
    """
    print("\n" + "=" * 100)
    print("MAP-REDUCE DETAILED ANALYSIS")
    print("=" * 100)
    
    map_responses = result.map_responses
    
    print(f"\n📊 Total number of batches (map operations): {len(map_responses)}")
    print(f"🎯 Query: {query}\n")
    
    # Analyze each batch's response
    all_key_points = []
    
    for batch_idx, response in enumerate(map_responses, 1):
        print(f"\n{'=' * 100}")
        print(f"BATCH #{batch_idx}")
        print(f"{'=' * 100}")
        
        # Get the response data
        batch_response = response.response
        
        if isinstance(batch_response, list):
            print(f"\n📝 Key points extracted: {len(batch_response)}")
            
            # Display each key point with its score
            for point_idx, point in enumerate(batch_response, 1):
                if isinstance(point, dict):
                    answer = point.get('answer', 'N/A')
                    score = point.get('score', 0)
                    
                    # Store for overall analysis
                    all_key_points.append({
                        'batch': batch_idx,
                        'point_num': point_idx,
                        'answer': answer,
                        'score': score
                    })
                    
                    print(f"\n   Point #{point_idx}:")
                    print(f"   ├─ Importance Score: {score}")
                    
                    # Visual score bar (0-100 scale)
                    if isinstance(score, (int, float)):
                        bar_length = int(score / 2)  # 0-100 -> 0-50 chars
                        print(f"   │  {'█' * bar_length}{'░' * (50 - bar_length)} {score}/100")
                    
                    # Display answer preview
                    answer_preview = str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
                    print(f"   └─ Answer: {answer_preview}")
        else:
            print(f"\n📝 Response (non-list format): {str(batch_response)[:300]}...")
        
        # Show batch metadata
        print(f"\n⏱️  Batch processing time: {response.completion_time:.2f}s")
        print(f"📊 Tokens - Prompt: {response.prompt_tokens:,}, Output: {response.output_tokens:,}")
    
    # Overall statistics
    if all_key_points:
        _print_map_reduce_statistics(all_key_points, result)


def _print_map_reduce_statistics(all_key_points: List[Dict], result):
    """Print statistics for map-reduce results."""
    print(f"\n\n{'=' * 100}")
    print("OVERALL MAP PHASE STATISTICS")
    print(f"{'=' * 100}\n")
    
    total_points = len(all_key_points)
    scores = [p['score'] for p in all_key_points if isinstance(p['score'], (int, float))]
    
    print(f"📊 Total key points extracted across all batches: {total_points}")
    
    if scores:
        print(f"\n📈 Importance Score Statistics (0-100 scale):")
        print(f"   • Minimum: {min(scores):.2f}")
        print(f"   • Maximum: {max(scores):.2f}")
        print(f"   • Mean: {np.mean(scores):.2f}")
        print(f"   • Median: {np.median(scores):.2f}")
        
        # Count by score threshold
        high_score = [s for s in scores if s >= 70]
        medium_score = [s for s in scores if 40 <= s < 70]
        low_score = [s for s in scores if s > 0 and s < 40]
        zero_score = [s for s in scores if s == 0]
        
        print(f"\n📊 Score Distribution:")
        print(f"   • High importance (≥70): {len(high_score)} points ({len(high_score)/len(scores)*100:.1f}%)")
        print(f"   • Medium importance (40-69): {len(medium_score)} points ({len(medium_score)/len(scores)*100:.1f}%)")
        print(f"   • Low importance (1-39): {len(low_score)} points ({len(low_score)/len(scores)*100:.1f}%)")
        print(f"   • No relevance (0): {len(zero_score)} points ({len(zero_score)/len(scores)*100:.1f}%)")
    
    # Show top-ranked points across all batches
    print(f"\n{'─' * 100}")
    print("TOP 10 HIGHEST-SCORED KEY POINTS (ACROSS ALL BATCHES)")
    print(f"{'─' * 100}\n")
    
    sorted_points = sorted(all_key_points, key=lambda x: x['score'] if isinstance(x['score'], (int, float)) else 0, reverse=True)
    
    for rank, point in enumerate(sorted_points[:10], 1):
        print(f"{rank}. [Batch {point['batch']}, Point {point['point_num']}] Score: {point['score']}")
        answer_preview = str(point['answer'])[:150] + "..." if len(str(point['answer'])) > 150 else str(point['answer'])
        print(f"   → {answer_preview}\n")
    
    # Show which points likely made it to the final response
    print(f"\n{'─' * 100}")
    print("REDUCE PHASE FILTERING")
    print(f"{'─' * 100}\n")
    
    # Points with score > 0 (what gets through to reduce phase)
    filtered_points = [p for p in sorted_points if isinstance(p['score'], (int, float)) and p['score'] > 0]
    print(f"✅ Points with score > 0: {len(filtered_points)} out of {total_points}")
    print(f"❌ Points filtered out (score = 0): {total_points - len(filtered_points)}")
    
    # Show reduce context
    if result.reduce_context_text:
        print(f"\n📝 Reduce phase context preview (first 800 chars):")
        print(f"{'─' * 100}")
        reduce_text = str(result.reduce_context_text)[:800] + "..." if len(str(result.reduce_context_text)) > 800 else str(result.reduce_context_text)
        print(reduce_text)
        print(f"{'─' * 100}")
        print(f"Total reduce context length: {len(str(result.reduce_context_text))} characters")


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in a text string.
    
    Args:
        text: Text to count tokens for
        encoding_name: Name of the tiktoken encoding to use
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(str(text)))


def analyze_context_tokens(context_data: Dict[str, pd.DataFrame], encoding_name: str = "cl100k_base") -> Dict[str, int]:
    """
    Analyze token counts for each component of context data.
    
    Args:
        context_data: Dictionary of DataFrames with context components
        encoding_name: Name of the tiktoken encoding to use
        
    Returns:
        Dictionary mapping component names to token counts (includes 'total_tokens' key)
    """
    token_counts = {}
    
    for key, df in context_data.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            # Concatenate all text columns
            text_content = df.to_string()
            token_counts[key] = count_tokens(text_content, encoding_name)
    
    # Add total tokens
    token_counts['total_tokens'] = sum(token_counts.values())
    
    return token_counts


def analyze_text_chunk_sizes(text_units_df: pd.DataFrame, text_column: str = 'text', 
                              id_column: str = 'id', encoding_name: str = "cl100k_base") -> None:
    """
    Analyze and display statistics about text chunk sizes and token counts.
    
    Args:
        text_units_df: DataFrame containing text chunks
        text_column: Name of the column containing text content
        id_column: Name of the column containing chunk IDs
        encoding_name: Name of the tiktoken encoding to use
    """
    import numpy as np
    
    # Analyze text unit sizes
    text_lengths = []
    token_counts_list = []
    
    for _, row in text_units_df.iterrows():
        text = row[text_column]
        text_lengths.append(len(text))
        token_counts_list.append(count_tokens(text, encoding_name))
    
    # Calculate statistics
    print("=" * 80)
    print("TEXT CHUNK SIZE ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Character Length Statistics:")
    print(f"   • Total chunks: {len(text_lengths)}")
    print(f"   • Min length: {min(text_lengths):,} chars")
    print(f"   • Max length: {max(text_lengths):,} chars")
    print(f"   • Mean length: {np.mean(text_lengths):,.0f} chars")
    print(f"   • Median length: {np.median(text_lengths):,.0f} chars")
    
    print(f"\n📊 Token Count Statistics:")
    print(f"   • Min tokens: {min(token_counts_list):,} tokens")
    print(f"   • Max tokens: {max(token_counts_list):,} tokens")
    print(f"   • Mean tokens: {np.mean(token_counts_list):,.0f} tokens")
    print(f"   • Median tokens: {np.median(token_counts_list):,.0f} tokens")
    print(f"   • 75th percentile: {np.percentile(token_counts_list, 75):,.0f} tokens")
    print(f"   • 90th percentile: {np.percentile(token_counts_list, 90):,.0f} tokens")
    print(f"   • 95th percentile: {np.percentile(token_counts_list, 95):,.0f} tokens")
    
    print(f"\n🔍 Top 10 Largest Chunks (by tokens):")
    largest_indices = np.argsort(token_counts_list)[-10:][::-1]
    for i, idx in enumerate(largest_indices, 1):
        tokens = token_counts_list[idx]
        chars = text_lengths[idx]
        chunk_id = text_units_df.iloc[idx][id_column]
        print(f"   {i}. Chunk {chunk_id}: {tokens:,} tokens ({chars:,} chars)")
    
    # Show distribution
    print(f"\n📈 Token Count Distribution:")
    max_tokens_val = max(token_counts_list)
    # Create bins that are guaranteed to be monotonically increasing
    bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000]
    # Only add the max+1 bin if it's larger than the last bin
    if max_tokens_val >= 20000:
        bins.append(max_tokens_val + 1)
    
    hist, _ = np.histogram(token_counts_list, bins=bins)
    for i in range(len(bins)-1):
        count = hist[i]
        pct = (count / len(token_counts_list)) * 100
        print(f"   • {bins[i]:>5,}-{bins[i+1]:>6,} tokens: {count:4d} chunks ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)

