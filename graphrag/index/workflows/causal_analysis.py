# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition for causal analysis."""

import logging
from typing import Any

import networkx as nx
import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.causal_analysis.causal_analyzer import (
    CausalAnalyzer,
    CausalAnalysisResult,
)
from graphrag.index.operations.causal_analysis.causal_graph_builder import CausalGraphBuilder
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.language_model.manager import ModelManager
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Run causal analysis on the extracted knowledge graph."""
    logger.info("Workflow started: causal_analysis")
    
    # Load required data
    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)
    
    # Load graph if available
    try:
        import networkx as nx
        from graphrag.index.operations.create_graph import create_graph
        
        # Create graph from relationships
        graph = create_graph(
            edges=relationships,
            edge_attr=["weight", "description"],
            nodes=entities,
            node_id="title"
        )
    except Exception as e:
        logger.warning(f"Could not create graph for causal analysis: {e}")
        graph = None
    
    # Get LLM configuration
    causal_analysis_llm_settings = config.get_language_model_config(
        config.causal_analysis.model_id
    )
    
    # Create causal analyzer
    model = ModelManager().get_or_create_chat_model(
        name="causal_analysis",
        model_type=causal_analysis_llm_settings.type,
        config=causal_analysis_llm_settings,
    )
    
    analyzer = CausalAnalyzer(
        model_invoker=model,
        max_analysis_length=config.causal_analysis.max_analysis_length,
    )
    
    # Perform causal analysis
    if graph is not None:
        result = await analyzer(
            graph=graph,
            entities=entities,
            relationships=relationships,
        )
    else:
        # Fallback: create minimal graph from relationships
        result = await _analyze_without_graph(
            analyzer, entities, relationships, context
        )
    
    # Save results
    await _save_causal_analysis_results(result, context.output_storage)
    
    # Create and save causal graph snapshot
    await _create_causal_graph_snapshot(result, entities, relationships, context.output_storage, config)
    
    logger.info("Workflow completed: causal_analysis")
    return WorkflowFunctionOutput(
        result={
            "causal_report": result.report,
            "causal_relationships": result.causal_relationships,
            "confidence_scores": result.confidence_scores,
            "key_entities": result.key_entities,
        }
    )


async def _analyze_without_graph(
    analyzer: CausalAnalyzer,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    context: PipelineRunContext,
) -> CausalAnalysisResult:
    """Perform causal analysis without a full graph."""
    # Create a minimal graph representation
    import networkx as nx
    
    graph = nx.Graph()
    
    # Add entities as nodes
    for _, entity in entities.iterrows():
        graph.add_node(
            entity['title'],
            type=entity.get('type', ''),
            description=entity.get('description', ''),
        )
    
    # Add relationships as edges
    for _, rel in relationships.iterrows():
        graph.add_edge(
            rel['source'],
            rel['target'],
            weight=rel.get('weight', 1.0),
            description=rel.get('description', ''),
        )
    
    return await analyzer(
        graph=graph,
        entities=entities,
        relationships=relationships,
    )


async def _save_causal_analysis_results(
    result: CausalAnalysisResult,
    output_storage: Any,
) -> None:
    """Save causal analysis results to storage."""
    # Save the report as text
    report_df = pd.DataFrame({
        'report': [result.report],
        'timestamp': [pd.Timestamp.now()],
    })
    await write_table_to_storage(report_df, "causal_analysis_report", output_storage)
    
    # Save causal relationships
    if result.causal_relationships:
        relationships_df = pd.DataFrame(result.causal_relationships)
        await write_table_to_storage(relationships_df, "causal_relationships", output_storage)
    
    # Save confidence scores
    if result.confidence_scores:
        confidence_df = pd.DataFrame([result.confidence_scores])
        await write_table_to_storage(confidence_df, "causal_confidence_scores", output_storage)
    
    # Save key entities
    if result.key_entities:
        entities_df = pd.DataFrame({
            'entity': result.key_entities,
            'rank': range(1, len(result.key_entities) + 1),
        })
        await write_table_to_storage(entities_df, "causal_key_entities", output_storage)


async def _create_causal_graph_snapshot(
    result: CausalAnalysisResult,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    output_storage: Any,
    config: GraphRagConfig,
) -> None:
    """Create and save a causal graph snapshot in GraphML format."""
    try:
        # Create causal graph builder
        builder = CausalGraphBuilder()
        
        # Build the causal graph
        logger.info(f"Building causal graph with {len(entities)} entities and {len(relationships)} relationships")
        causal_graph = builder.build_causal_graph(result, entities, relationships)
        logger.info(f"Created causal graph with {len(list(causal_graph.nodes()))} nodes and {len(list(causal_graph.edges()))} edges")
        
        # Create a filtered subgraph for visualization
        subgraph = builder.create_causal_subgraph(
            causal_graph,
            min_confidence=0.5,
            max_nodes=50
        )
        logger.info(f"Created subgraph with {len(list(subgraph.nodes()))} nodes and {len(list(subgraph.edges()))} edges")
        
        # Export to GraphML using write_graphml method
        import io
        import tempfile
        import os
        
        # Create temporary files for GraphML export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as temp_sub:
            nx.write_graphml(subgraph, temp_sub.name)
            temp_sub.flush()
            
            # Read the content and save to storage
            with open(temp_sub.name, 'r', encoding='utf-8') as f:
                subgraph_content = f.read()
            await output_storage.set("causal_graph.graphml", subgraph_content)
            os.unlink(temp_sub.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as temp_full:
            nx.write_graphml(causal_graph, temp_full.name)
            temp_full.flush()
            
            # Read the content and save to storage
            with open(temp_full.name, 'r', encoding='utf-8') as f:
                full_content = f.read()
            await output_storage.set("causal_graph_full.graphml", full_content)
            os.unlink(temp_full.name)
        
        logger.info("Causal graph snapshots created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create causal graph snapshot: {e}")
        # Don't fail the entire workflow if snapshot creation fails 