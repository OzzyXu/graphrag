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
    """Run causal analysis on the extracted fallback source."""
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
    
    # Log the analysis length setting
    max_length = config.causal_analysis.max_analysis_length
    if max_length == "full":
        logger.info("Causal analysis configured with 'full' option - no length limits will be applied")
    else:
        logger.info(f"Causal analysis configured with max length: {max_length} characters")
    
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
    
    # Output prompt template and network data
    await _output_prompt_template(context.output_storage)
    await _output_network_data(entities, relationships, context.output_storage)
    
    # Save results
    await _save_causal_analysis_results(result, context.output_storage)
    
    # Create and save causal graph snapshot
    await _create_causal_graph_snapshot(result, entities, relationships, context.output_storage, config)
    
    # Create and save fallback-only graph based on fallback source
    await _create_fallback_only_graph(entities, relationships, context.output_storage)
    
    logger.info("Workflow completed: causal_analysis")
    return WorkflowFunctionOutput(
        result={
            "causal_report": result.report,
            "causal_relationships": result.causal_relationships,
            "confidence_scores": result.confidence_scores,
            "key_entities": result.key_entities,
            "formatted_prompt": result.formatted_prompt,
            "causal_report_file": "output/causal_analysis_report.md",
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
    # Save the report as a markdown file
    if result.report:
        try:
            import os
            from pathlib import Path
            
            # Get the base directory from the storage
            base_dir = getattr(output_storage, '_root_dir', 'output')
            
            # Save the report directly under output directory
            report_file_path = Path(base_dir) / 'causal_analysis_report.md'
            
            with open(report_file_path, 'w', encoding='utf-8') as f:
                f.write("# Causal Analysis Report\n")
                f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
                f.write("---\n\n")
                f.write(result.report)
            
            logger.info(f"Causal analysis report saved to: {report_file_path}")
            
            # Also save to storage for consistency
            await output_storage.set("causal_analysis_report.md", result.report)
            
        except Exception as e:
            logger.error(f"Failed to save causal analysis report as markdown: {e}")
            # Fallback to saving as DataFrame if markdown fails
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


async def _output_prompt_template(output_storage: Any) -> None:
    """Output the actual causal analysis prompt used in the workflow to output/prompts directory."""
    try:
        import os
        from pathlib import Path
        from graphrag.prompts.index.causal_analysis import CAUSAL_ANALYSIS_PROMPT
        
        # Get the base directory from storage and create prompts subdirectory
        base_dir = getattr(output_storage, '_root_dir', 'output')
        # Go up one level from output to create prompts directory at the same level
        root_dir = Path(base_dir).parent
        prompts_dir = root_dir / 'prompts'
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the actual prompt used in the workflow
        prompt_file_path = prompts_dir / 'causal_analysis_prompt.txt'
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write("# Causal Analysis Prompt Used in Workflow\n")
            f.write(f"# Source: graphrag/prompts/index/causal_analysis.py\n")
            f.write("# " + "="*50 + "\n\n")
            f.write(CAUSAL_ANALYSIS_PROMPT)
        
        logger.info(f"Causal analysis prompt saved to: {prompt_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to output prompt template: {e}")
        # Don't fail the entire workflow if prompt template output fails


async def _output_network_data(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    output_storage: Any,
) -> None:
    """Extract and output the network data to output directory."""
    try:
        import os
        from pathlib import Path
        
        # Get the base directory from storage
        base_dir = getattr(output_storage, '_root_dir', 'output')
        
        # Create detailed network data output
        network_data_content = "---Network Data--\n"
        network_data_content += "=== ENTITIES ===\n"
        
        # Add entities
        for _, entity in entities.iterrows():
            entity_id = entity.get('id', 'Unknown')
            entity_type = entity.get('type', '')
            description = entity.get('description', '')
            degree = entity.get('degree', 0)
            centrality = entity.get('centrality', 0.0)
            
            network_data_content += f"Entity: {entity_id}\n"
            network_data_content += f"  Type: {entity_type}\n"
            network_data_content += f"  Description: {description}\n"
            network_data_content += f"  Degree: {degree}\n"
            network_data_content += f"  Centrality: {centrality:.3f}\n\n"
        
        network_data_content += "=== RELATIONSHIPS ===\n"
        
        # Add relationships
        for _, rel in relationships.iterrows():
            source = rel.get('source', 'Unknown')
            target = rel.get('target', 'Unknown')
            weight = rel.get('weight', 1.0)
            description = rel.get('description', '')
            
            network_data_content += f"From: {source} -> To: {target}\n"
            network_data_content += f"  Weight: {weight}\n"
            network_data_content += f"  Description: {description}\n\n"
        
        # Save detailed network data
        network_data_file_path = Path(base_dir) / 'network_data_detailed.txt'
        with open(network_data_file_path, 'w', encoding='utf-8') as f:
            f.write(network_data_content)
        
        logger.info(f"Network data detailed output saved to: {network_data_file_path}")
        
        # Also save to storage for consistency
        await output_storage.set("network_data_detailed.txt", network_data_content)
        
    except Exception as e:
        logger.error(f"Failed to output network data: {e}")
        # Don't fail the entire workflow if network data output fails


async def _create_fallback_only_graph(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    output_storage: Any,
) -> None:
    """Create and save a fallback-only graph based on original fallback relationships."""
    try:
        # Create causal graph builder
        builder = CausalGraphBuilder()
        
        # Build the fallback-only graph
        logger.info(f"Building fallback-only graph with {len(entities)} entities and {len(relationships)} relationships")
        fallback_graph = builder.build_fallback_only_graph(entities, relationships)
        logger.info(f"Created fallback-only graph with {len(list(fallback_graph.nodes()))} nodes and {len(list(fallback_graph.edges()))} edges")
        
        # Create a filtered subgraph for visualization
        subgraph = builder.create_causal_subgraph(
            fallback_graph,
            min_confidence=0.1,  # Lower threshold for fallback relationships
            max_nodes=50
        )
        logger.info(f"Created fallback-only subgraph with {len(list(subgraph.nodes()))} nodes and {len(list(subgraph.edges()))} edges")
        
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
            await output_storage.set("causal_graph_fallback_only.graphml", subgraph_content)
            os.unlink(temp_sub.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as temp_full:
            nx.write_graphml(fallback_graph, temp_full.name)
            temp_full.flush()
            
            # Read the content and save to storage
            with open(temp_full.name, 'r', encoding='utf-8') as f:
                full_content = f.read()
            await output_storage.set("causal_graph_fallback_only_full.graphml", full_content)
            os.unlink(temp_full.name)
        
        logger.info("Fallback-only graph snapshots created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create fallback-only graph snapshot: {e}")
        # Don't fail the entire workflow if fallback graph creation fails