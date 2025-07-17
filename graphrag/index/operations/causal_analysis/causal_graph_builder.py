# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module for building causal graphs from analysis results."""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

from graphrag.index.operations.causal_analysis.causal_analyzer import CausalAnalysisResult

logger = logging.getLogger(__name__)


class CausalGraphBuilder:
    """Builds a causal graph from causal analysis results."""

    def __init__(self):
        """Initialize the causal graph builder."""
        pass

    def build_causal_graph(
        self, 
        causal_result: CausalAnalysisResult,
        entities: pd.DataFrame,
        relationships: pd.DataFrame
    ) -> nx.DiGraph:
        """Build a directed causal graph from analysis results.
        
        Parameters
        ----------
        causal_result : CausalAnalysisResult
            The result from causal analysis
        entities : pd.DataFrame
            Original entities from the knowledge graph
        relationships : pd.DataFrame
            Original relationships from the knowledge graph
            
        Returns
        -------
        nx.DiGraph
            A directed graph representing causal relationships
        """
        # Create a directed graph for causal relationships
        causal_graph = nx.DiGraph()
        
        # Add nodes from original entities (limit to top entities for visualization)
        top_entities = entities.nlargest(20, 'degree')  # Get top 20 entities by degree
        for _, entity in top_entities.iterrows():
            entity_id = entity['id']
            causal_graph.add_node(
                entity_id,
                type='entity',
                description=entity.get('description', ''),
                entity_type=entity.get('type', ''),
                source='original',
                degree=entity.get('degree', 0)
            )
        
        # Add causal relationships as directed edges
        if causal_result.causal_relationships:
            for rel in causal_result.causal_relationships:
                cause = rel.get('cause', '')
                effect = rel.get('effect', '')
                description = rel.get('description', '')
                confidence = rel.get('confidence', 0.5)
                
                # Add nodes if they don't exist
                if cause and cause not in causal_graph:
                    causal_graph.add_node(cause, type='causal_entity', source='causal_analysis')
                if effect and effect not in causal_graph:
                    causal_graph.add_node(effect, type='causal_entity', source='causal_analysis')
                
                # Add directed edge from cause to effect
                if cause and effect:
                    causal_graph.add_edge(
                        cause, 
                        effect,
                        relationship_type='causal',
                        description=description,
                        confidence=confidence,
                        source='causal_analysis'
                    )
        
        # Add key entities as highlighted nodes
        for i, entity in enumerate(causal_result.key_entities):
            if entity in causal_graph:
                causal_graph.nodes[entity]['is_key_entity'] = True
                causal_graph.nodes[entity]['key_entity_rank'] = i + 1
            else:
                causal_graph.add_node(
                    entity,
                    type='key_entity',
                    is_key_entity=True,
                    key_entity_rank=i + 1,
                    source='causal_analysis'
                )
        
        # If no causal relationships were found, create some from the original relationships
        if causal_result.causal_relationships is None or len(causal_result.causal_relationships) == 0:
            logger.info("No causal relationships found, creating inferred causal relationships from original graph")
            # Add some high-weight relationships as causal
            top_relationships = relationships.nlargest(10, 'weight')
            for _, rel in top_relationships.iterrows():
                source = rel['source']
                target = rel['target']
                description = rel.get('description', '')
                weight = rel.get('weight', 1.0)
                
                # Add nodes if they don't exist
                if source not in causal_graph:
                    causal_graph.add_node(source, type='inferred_entity', source='inferred')
                if target not in causal_graph:
                    causal_graph.add_node(target, type='inferred_entity', source='inferred')
                
                # Add edge with lower confidence since it's inferred
                weight_float = float(weight) if weight is not None else 1.0
                causal_graph.add_edge(
                    source, target,
                    relationship_type='inferred_causal',
                    description=description,
                    confidence=min(weight_float / 10.0, 0.8),  # Scale weight to confidence
                    source='inferred'
                )
        
        # Note: Graph-level attributes removed to avoid Gephi warnings
        # Confidence scores and metadata are preserved in node/edge attributes
        # Note: Isolated nodes are preserved in the full graph but removed in subgraph
        
        return causal_graph

    def create_causal_subgraph(
        self,
        causal_graph: nx.DiGraph,
        min_confidence: float = 0.5,
        max_nodes: int = 50
    ) -> nx.DiGraph:
        """Create a filtered subgraph for visualization.
        
        Parameters
        ----------
        causal_graph : nx.DiGraph
            The full causal graph
        min_confidence : float
            Minimum confidence threshold for edges
        max_nodes : int
            Maximum number of nodes to include
            
        Returns
        -------
        nx.DiGraph
            A filtered subgraph suitable for visualization
        """
        # Filter edges by confidence
        edges_to_keep = []
        for source, target, data in causal_graph.edges(data=True):
            confidence = data.get('confidence', 0.0)
            if confidence >= min_confidence:
                edges_to_keep.append((source, target))
        
        # Create subgraph with filtered edges
        subgraph = causal_graph.edge_subgraph(edges_to_keep).copy()
        
        # If still too many nodes, select most important ones
        if subgraph.number_of_nodes() > max_nodes:
            # Score nodes by importance
            node_scores = {}
            for node in subgraph.nodes():
                score = 0
                # Key entities get high priority
                if subgraph.nodes[node].get('is_key_entity', False):
                    score += 1000
                # High degree nodes get priority
                try:
                    score += int(subgraph.degree(node))
                except (TypeError, ValueError):
                    score += 1
                # In-degree (being an effect) gets priority for DiGraph
                if isinstance(subgraph, nx.DiGraph):
                    try:
                        score += int(subgraph.in_degree(node)) * 2
                    except (TypeError, ValueError):
                        score += 1
                node_scores[node] = score
            
            # Keep top nodes
            top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = [node for node, _ in top_nodes]
            
            subgraph = subgraph.subgraph(top_node_names).copy()
        
        # Remove isolated nodes from subgraph (for clean visualization)
        isolated_nodes = [node for node in subgraph.nodes() if subgraph.degree(node) == 0]
        if isolated_nodes:
            subgraph.remove_nodes_from(isolated_nodes)
            logger.info(f"Removed {len(isolated_nodes)} isolated nodes from subgraph: {isolated_nodes}")
        
        return subgraph

    def export_to_graphml(
        self,
        causal_graph: nx.DiGraph,
        output_path: str
    ) -> None:
        """Export the causal graph to GraphML format.
        
        Parameters
        ----------
        causal_graph : nx.DiGraph
            The causal graph to export
        output_path : str
            Path to save the GraphML file
        """
        try:
            nx.write_graphml(causal_graph, output_path)
            logger.info(f"Causal graph exported to GraphML: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export causal graph to GraphML: {e}")
            raise 