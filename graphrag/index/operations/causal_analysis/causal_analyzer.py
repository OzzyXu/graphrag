# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing causal analysis functionality."""

import logging
from dataclasses import dataclass
from typing import Any, Union

import networkx as nx
import pandas as pd

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.causal_analysis import CAUSAL_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class CausalAnalysisResult:
    """Causal analysis result class definition."""

    report: str
    """The causal analysis report."""
    
    causal_relationships: list[dict[str, Any]]
    """Extracted causal relationships."""
    
    confidence_scores: dict[str, float]
    """Confidence scores for causal claims."""
    
    key_entities: list[str]
    """Key entities identified in causal analysis."""
    
    formatted_prompt: str
    """The formatted prompt that was sent to the LLM."""


class CausalAnalyzer:
    """Causal analysis extractor class definition."""

    _model: ChatModel
    _analysis_prompt: str
    _max_analysis_length: Union[int, str]

    def __init__(
        self,
        model_invoker: ChatModel,
        prompt: str | None = None,
        max_analysis_length: Union[int, str] = 2000,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._analysis_prompt = prompt or CAUSAL_ANALYSIS_PROMPT
        self._max_analysis_length = max_analysis_length
        
        # Validate max_analysis_length
        if isinstance(max_analysis_length, str) and max_analysis_length.lower() != "full":
            raise ValueError("max_analysis_length must be an integer or 'full'")
    
    @property
    def is_full_length_enabled(self) -> bool:
        """Check if full length analysis is enabled."""
        return self._max_analysis_length == "full"
    
    @property
    def max_length(self) -> Union[int, str]:
        """Get the maximum analysis length setting."""
        return self._max_analysis_length

    async def __call__(
        self, 
        graph: nx.Graph,
        entities: pd.DataFrame,
        relationships: pd.DataFrame,
        prompt_variables: dict[str, Any] | None = None
    ) -> CausalAnalysisResult:
        """Perform causal analysis on the knowledge graph."""
        if prompt_variables is None:
            prompt_variables = {}
            
        # Prepare graph data for analysis
        graph_data = self._prepare_graph_data(graph, entities, relationships)
        
        # Perform causal analysis
        formatted_prompt, result = await self._analyze_causality(graph_data, prompt_variables)
        
        # Extract structured information
        causal_relationships = self._extract_causal_relationships(result)
        confidence_scores = self._extract_confidence_scores(result)
        key_entities = self._extract_key_entities(result)
        
        return CausalAnalysisResult(
            report=result,
            causal_relationships=causal_relationships,
            confidence_scores=confidence_scores,
            key_entities=key_entities,
            formatted_prompt=formatted_prompt,
        )

    def _prepare_graph_data(
        self, 
        graph: nx.Graph, 
        entities: pd.DataFrame, 
        relationships: pd.DataFrame
    ) -> str:
        """Prepare graph data for causal analysis."""
        # Extract key graph information
        nodes_info = []
        for node, data in graph.nodes(data=True):
            node_info = {
                'id': node,
                'type': data.get('type', ''),
                'description': data.get('description', ''),
                'degree': graph.degree(node),
                'centrality': nx.degree_centrality(graph).get(node, 0),
            }
            nodes_info.append(node_info)
        
        edges_info = []
        for source, target, data in graph.edges(data=True):
            edge_info = {
                'source': source,
                'target': target,
                'weight': data.get('weight', 1.0),
                'description': data.get('description', ''),
            }
            edges_info.append(edge_info)
        
        # Format as structured text
        graph_data = "=== ENTITIES ===\n"
        for node in nodes_info:
            graph_data += f"Entity: {node['id']}\n"
            graph_data += f"  Type: {node['type']}\n"
            graph_data += f"  Description: {node['description']}\n"
            graph_data += f"  Degree: {node['degree']}\n"
            graph_data += f"  Centrality: {node['centrality']:.3f}\n\n"
        
        graph_data += "=== RELATIONSHIPS ===\n"
        for edge in edges_info:
            graph_data += f"From: {edge['source']} -> To: {edge['target']}\n"
            graph_data += f"  Weight: {edge['weight']}\n"
            graph_data += f"  Description: {edge['description']}\n\n"
        
        return graph_data

    async def _analyze_causality(
        self, 
        graph_data: str, 
        prompt_variables: dict[str, Any]
    ) -> tuple[str, str]:
        """Perform causal analysis using LLM."""
        # Format the prompt with graph data
        formatted_prompt = self._analysis_prompt.format(
            graph_data=graph_data,
            **prompt_variables
        )
        
        # Get LLM response
        response = await self._model.achat(formatted_prompt)
        result = response.output.content or ""
        
        # Truncate if too long (only if not using "full" option)
        if self._max_analysis_length != "full" and isinstance(self._max_analysis_length, int):
            if len(result) > self._max_analysis_length:
                logger.info(f"Truncating causal analysis report from {len(result)} to {self._max_analysis_length} characters")
                result = result[:self._max_analysis_length] + "..."
            else:
                logger.info(f"Causal analysis report generated: {len(result)} characters (within {self._max_analysis_length} limit)")
        else:
            logger.info(f"Full-length causal analysis report generated: {len(result)} characters (no truncation)")
        
        return formatted_prompt, result

    def _extract_causal_relationships(self, report: str) -> list[dict[str, Any]]:
        """Extract structured causal relationships from the report."""
        relationships = []
        
        # Simple extraction based on common patterns
        lines = report.split('\n')
        current_relationship = {}
        
        for line in lines:
            line = line.strip()
            if 'cause' in line.lower() or 'effect' in line.lower():
                # Extract cause-effect pairs
                if '->' in line or '→' in line:
                    parts = line.replace('->', '→').split('→')
                    if len(parts) == 2:
                        relationships.append({
                            'cause': parts[0].strip(),
                            'effect': parts[1].strip(),
                            'description': line,
                            'confidence': 0.7  # Default confidence
                        })
        
        return relationships

    def _extract_confidence_scores(self, report: str) -> dict[str, float]:
        """Extract confidence scores from the report."""
        confidence_scores = {}
        
        # Look for confidence indicators in the report
        lines = report.split('\n')
        for line in lines:
            line = line.lower()
            if 'confidence' in line or 'reliability' in line:
                # Extract confidence scores (simple pattern matching)
                if 'high' in line:
                    confidence_scores['overall'] = 0.8
                elif 'medium' in line:
                    confidence_scores['overall'] = 0.6
                elif 'low' in line:
                    confidence_scores['overall'] = 0.4
        
        if not confidence_scores:
            confidence_scores['overall'] = 0.6  # Default confidence
        
        return confidence_scores

    def _extract_key_entities(self, report: str) -> list[str]:
        """Extract key entities from the 'Key Entities and Their Roles' section."""
        entities = []
        
        # Find the "Key Entities and Their Roles" section
        lines = report.split('\n')
        in_entities_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're entering the key entities section
            if '2. Key Entities and Their Roles' in line or '## 2. Key Entities and Their Roles' in line:
                in_entities_section = True
                continue
            
            # Check if we're leaving the section (next numbered section)
            if in_entities_section and (
                line.startswith('3.') or line.startswith('## 3.') or 
                line.startswith('**3.') or 'Major Causal Pathways' in line
            ):
                break
            
            # Extract entities from this section
            if in_entities_section and line:
                # Look for pattern: - **Entity Name**: description
                import re
                entity_match = re.match(r'^-\s*\*\*([^*]+)\*\*:', line)
                if entity_match:
                    entity_name = entity_match.group(1).strip()
                    if entity_name and entity_name not in entities:
                        entities.append(entity_name)
                
                # Also look for entities mentioned in regular text within this section
                # Extract capitalized words that look like proper nouns
                words = line.split()
                for word in words:
                    if len(word) > 2 and word[0].isupper():
                        clean_word = word.strip('.,;:!?*-')
                        # Skip common words and formatting
                        if (clean_word not in ['The', 'This', 'These', 'They', 'Key', 'Entities', 'Their', 'Roles'] 
                            and clean_word not in entities 
                            and not clean_word.startswith('**')):
                            entities.append(clean_word)
        
        # If no entities found in the specific section, fallback to the old method
        if not entities:
            entities = self._extract_key_entities_fallback(report)
        
        return entities[:10]  # Return top 10 entities

    def _extract_key_entities_fallback(self, report: str) -> list[str]:
        """Fallback method for entity extraction if section-specific extraction fails."""
        entities = []
        lines = report.split('\n')
        for line in lines:
            words = line.split()
            for word in words:
                if len(word) > 2 and word[0].isupper():
                    clean_word = word.strip('.,;:!?')
                    if clean_word not in entities:
                        entities.append(clean_word)
        return entities 