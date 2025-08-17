# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CausalSearch implementation."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken

from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.protocol.base import ChatModel
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
)

logger = logging.getLogger(__name__)


class CausalSearchError(Exception):
    """Custom exception for causal search failures."""
    pass


class CausalSearch(BaseSearch[LocalSearchMixedContext]):
    """Search orchestration for causal search mode."""

    def __init__(
        self,
        model: ChatModel,
        context_builder: LocalSearchMixedContext,
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str | None = None,
        response_type: str = "Multiple Paragraphs",
        callbacks: list[QueryCallbacks] | None = None,
        model_params: dict[str, Any] | None = None,
        context_builder_params: dict | None = None,
        s_parameter: int = 3,
        max_context_tokens: int = 12000,
    ):
        super().__init__(
            model=model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            model_params=model_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks or []
        self.response_type = response_type
        self.s_parameter = s_parameter
        self.max_context_tokens = max_context_tokens
        
        # Load prompts
        self.causal_discovery_prompt = self._load_causal_discovery_prompt()
        self.causal_summary_prompt = self._load_causal_summary_prompt()

    def _load_causal_discovery_prompt(self) -> str:
        """Load the causal discovery prompt template."""
        try:
            from graphrag.prompts.query import CAUSAL_DISCOVERY_PROMPT
            logger.info("Loaded causal discovery prompt from GraphRAG prompts")
            return CAUSAL_DISCOVERY_PROMPT
        except ImportError:
            logger.warning("Failed to import causal discovery prompt, using default")
            return self._get_default_causal_discovery_prompt()
        except Exception as e:
            logger.warning(f"Failed to load causal discovery prompt: {e}, using default")
            return self._get_default_causal_discovery_prompt()

    def _load_causal_summary_prompt(self) -> str:
        """Load the causal summary prompt template."""
        try:
            from graphrag.prompts.query import CAUSAL_SUMMARY_PROMPT
            logger.info("Loaded causal summary prompt from GraphRAG prompts")
            return CAUSAL_SUMMARY_PROMPT
        except ImportError:
            logger.warning("Failed to import causal summary prompt, using default")
            return self._get_default_causal_summary_prompt()
        except Exception as e:
            logger.warning(f"Failed to load causal summary prompt: {e}, using default")
            return self._get_default_causal_summary_prompt()

    def _get_default_causal_discovery_prompt(self) -> str:
        """Get default causal discovery prompt if file not found."""
        return """---Role---
You are a smart assistant that helps a human analyst to perform **causal discovery** and **impact assessment**. Your task is to
analyze a **Network Data** and generate a professional report summarizing the causal effect and key insights.
--- Goal ---
Write a **structured, professional causality analysis report** that:
- **Identifies** key entities and their roles in the causality
- **Explains** the observed causal relationships and their potential impact
- **Assesses** the strength and credibility of causal claims based on available data
---Network Data---
{graph_data}
--- Report Format ---
**1. Introduction**
Briefly introduce the context and purpose of this causal analysis.
**2. Key Entities and Their Roles**
Provide an overview of the most important entities in the causal network and their relevance.
**3. Major Causal Pathways**
Describe the primary causal chains observed, emphasizing key cause-and-effect relationships.
**4. Confidence and Evidence Strength**
Assess the reliability of the causal claims, mentioning supporting data where available.
**5. Implications and Recommendations**
Discuss the potential impact of these causal relationships and suggest possible actions.
Write a **structured, analytical, and professional** report."""

    def _get_default_causal_summary_prompt(self) -> str:
        """Get default causal summary prompt if file not found."""
        return """---Role---
You are a helpful assistant responding to questions about data in the tables provided. You are also specializing in **causal
reasoning and impact assessment**. Your task is to generate a structured response based on an extracted causal summary.
---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all the Causal Summary
from multiple analysts who focused on different parts of the dataset.
If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do
not make anything up.
The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a
comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and
format.
The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
The response should also preserve all the data references previously included in the analysts' reports, but do not mention the
roles of multiple analysts in the analysis process.
---Causal Summary---
{causal_summary}
---Target Response Length and Format---
{response_type}
---User Query---
{query}
Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown."""

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build causal search context and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        
        try:
            # Step 1: Extract extended nodes (k + s)
            local_search_top_k = kwargs.get('top_k_mapped_entities', 10)
            extended_nodes = await self._extract_extended_nodes(
                query, local_search_top_k, **kwargs
            )
            
            # Step 2: Extract graph information using local search components
            graph_context = await self._extract_graph_information(
                extended_nodes, **kwargs
            )
            
            # Step 3: Format network data for causal discovery prompt
            network_data = self._format_network_data_for_causal_prompt(graph_context)
            
            # Step 4: Generate causal report (Stage 1)
            causal_report = await self._generate_causal_report(network_data)
            
            # Step 5: Generate final response (Stage 2)
            final_response = await self._generate_final_response(causal_report, query)
            
            # Step 6: Save outputs to data folder
            await self._save_outputs(network_data, causal_report, query)
            
            # Step 7: Save prompts to data folder
            await self._save_prompts()
            
            llm_calls["causal_discovery"] = 1
            llm_calls["response_generation"] = 1
            prompt_tokens["causal_discovery"] = num_tokens(
                self.causal_discovery_prompt.format(graph_data=network_data), 
                self.token_encoder
            )
            prompt_tokens["response_generation"] = num_tokens(
                self.causal_summary_prompt.format(
                    causal_summary=causal_report,
                    query=query,
                    response_type=self.response_type
                ), 
                self.token_encoder
            )
            output_tokens["causal_discovery"] = num_tokens(causal_report, self.token_encoder)
            output_tokens["response_generation"] = num_tokens(final_response, self.token_encoder)

            for callback in self.callbacks:
                callback.on_context(graph_context.context_records)

            return SearchResult(
                response=final_response,
                context_data=graph_context.context_records,
                context_text=graph_context.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=sum(output_tokens.values()),
                llm_calls_categories=llm_calls,
                prompt_tokens_categories=prompt_tokens,
                output_tokens_categories=output_tokens,
            )

        except Exception as e:
            logger.error(f"Causal search failed: {e}")
            raise CausalSearchError(f"Causal search method failed: {e}")

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the causal search response."""
        # For streaming, we'll generate the full response first, then stream it
        result = await self.search(query, conversation_history)
        # Split response into chunks and stream
        response_text = result.response if isinstance(result.response, str) else str(result.response)
        for chunk in response_text.split():
            yield chunk + " "
            await asyncio.sleep(0.01)  # Small delay for streaming effect

    async def _extract_extended_nodes(
        self, 
        query: str, 
        local_search_top_k: int,
        **kwargs
    ) -> list:
        """Extract k+s nodes where k = local_search_top_k, s = self.s_parameter."""
        try:
            # Get the normal local search nodes (k nodes)
            local_nodes = await self._get_local_search_nodes(query, local_search_top_k, **kwargs)
            
            # Get additional s nodes for causal analysis
            additional_nodes = await self._get_additional_causal_nodes(query, **kwargs)
            
            # Combine and deduplicate
            all_nodes = list(set(local_nodes + additional_nodes))
            
            logger.info(f"Extracted {len(all_nodes)} nodes: {len(local_nodes)} local + {len(additional_nodes)} additional")
            return all_nodes
            
        except Exception as e:
            logger.error(f"Failed to extract extended nodes: {e}")
            raise CausalSearchError(f"Node extraction failed: {e}")

    async def _get_local_search_nodes(
        self, 
        query: str, 
        top_k: int,
        **kwargs
    ) -> list:
        """Get the normal local search nodes using oversample_scaler."""
        try:
            # Use the context builder to get local search nodes
            # This mimics the local search entity extraction process
            context_result = self.context_builder.build_context(
                query=query,
                top_k_mapped_entities=top_k,
                **self.context_builder_params
            )
            
            # Extract entity IDs from the context
            if hasattr(context_result, 'context_records') and context_result.context_records:
                # Look for entities in the context records
                entities = context_result.context_records.get('entities', pd.DataFrame())
                if not entities.empty and len(entities) > 0:
                    # Extract entity IDs from the DataFrame
                    if 'id' in entities.columns:
                        entity_ids = entities['id'].tolist()
                    elif 'entity_id' in entities.columns:
                        entity_ids = entities['entity_id'].tolist()
                    else:
                        # If no ID column, try to get from index or other columns
                        entity_ids = entities.index.tolist() if not entities.empty else []
                    
                    logger.info(f"Extracted {len(entity_ids)} local search nodes")
                    return entity_ids[:top_k]  # Ensure we only return top_k nodes
            
            # If no entities found in context, try to extract from the context builder directly
            if hasattr(self.context_builder, 'entities') and self.context_builder.entities:
                # Get entities directly from the context builder
                all_entity_ids = list(self.context_builder.entities.keys())
                logger.info(f"Found {len(all_entity_ids)} entities in context builder")
                
                # For now, return top entities by rank (this is a fallback)
                if all_entity_ids:
                    # Sort by rank if available
                    sorted_entities = sorted(
                        self.context_builder.entities.values(),
                        key=lambda x: x.rank if hasattr(x, 'rank') and x.rank else 0,
                        reverse=True
                    )
                    top_entity_ids = [entity.id for entity in sorted_entities[:top_k]]
                    logger.info(f"Returning top {len(top_entity_ids)} entities by rank")
                    return top_entity_ids
            
            logger.warning("No entities found in local search context")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get local search nodes: {e}")
            return []

    async def _get_additional_causal_nodes(
        self, 
        query: str,
        **kwargs
    ) -> list:
        """Get additional s nodes for causal analysis using causal heuristics."""
        try:
            # Get the s_parameter value
            s_parameter = getattr(self, 's_parameter', 3)
            
            if s_parameter <= 0:
                logger.info("s_parameter is 0 or negative, no additional nodes needed")
                return []
            
            # Use causal analysis heuristics to find additional nodes
            # Strategy 1: Find entities with high relationship counts (potential causal hubs)
            # Strategy 2: Find entities mentioned in community reports (potential causal factors)
            # Strategy 3: Find entities with high rank but not in top-k
            
            additional_nodes = []
            
            # Strategy 1: High relationship count entities
            if hasattr(self.context_builder, 'relationships') and self.context_builder.relationships:
                # Sort entities by relationship count
                entity_relationship_counts = {}
                for rel in self.context_builder.relationships.values():
                    source = rel.source
                    target = rel.target
                    entity_relationship_counts[source] = entity_relationship_counts.get(source, 0) + 1
                    entity_relationship_counts[target] = entity_relationship_counts.get(target, 0) + 1
                
                # Get top entities by relationship count
                high_relationship_entities = sorted(
                    entity_relationship_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Add top entities that aren't already in local search
                local_nodes = await self._get_local_search_nodes(query, 10, **kwargs)  # Get more for comparison
                for entity_name, _ in high_relationship_entities[:s_parameter * 2]:  # Get 2x more for filtering
                    if entity_name not in local_nodes and len(additional_nodes) < s_parameter:
                        additional_nodes.append(entity_name)
            
            # Strategy 2: Entities from community reports
            if hasattr(self.context_builder, 'community_reports') and self.context_builder.community_reports:
                for community_id, community in self.context_builder.community_reports.items():
                    if len(additional_nodes) >= s_parameter:
                        break
                    
                    # Look for entity mentions in community descriptions
                    if hasattr(community, 'description') and community.description:
                        # Simple entity extraction from description
                        # This could be enhanced with more sophisticated NLP
                        for entity_name in self.context_builder.entities.keys():
                            if entity_name in community.description and entity_name not in additional_nodes:
                                additional_nodes.append(entity_name)
                                if len(additional_nodes) >= s_parameter:
                                    break
            
            # Strategy 3: High rank entities not in top-k
            if hasattr(self.context_builder, 'entities') and self.context_builder.entities:
                # Sort entities by rank
                sorted_entities = sorted(
                    self.context_builder.entities.values(),
                    key=lambda x: x.rank if hasattr(x, 'rank') and x.rank else 0,
                    reverse=True
                )
                
                local_nodes = await self._get_local_search_nodes(query, 10, **kwargs)
                for entity in sorted_entities:
                    if len(additional_nodes) >= s_parameter:
                        break
                    if entity.id not in local_nodes and entity.id not in additional_nodes:
                        additional_nodes.append(entity.id)
            
            logger.info(f"Extracted {len(additional_nodes)} additional causal nodes using s_parameter={s_parameter}")
            return additional_nodes[:s_parameter]  # Ensure we only return s_parameter nodes
            
        except Exception as e:
            logger.error(f"Failed to get additional causal nodes: {e}")
            return []

    async def _extract_graph_information(
        self, 
        selected_nodes: list,
        **kwargs
    ) -> Any:
        """Reuse local search components to extract graph information."""
        try:
            # Use the context builder to extract graph information
            # This reuses the local search context building logic
            # Pass the selected nodes to focus the context on them
            context_result = self.context_builder.build_context(
                query="",  # Empty query since we already have selected nodes
                **self.context_builder_params
            )
            
            logger.info(f"Extracted graph information with context length: {len(context_result.context_chunks) if context_result.context_chunks else 0}")
            return context_result
            
        except Exception as e:
            logger.error(f"Failed to extract graph information: {e}")
            raise CausalSearchError(f"Graph information extraction failed: {e}")

    def _format_network_data_for_causal_prompt(self, graph_context: Any) -> str:
        """Format extracted graph information for causal discovery prompt."""
        try:
            # Convert context data to structured format
            network_data = {
                "entities": [],
                "relationships": [],
                "text_units": [],
                "community_reports": []
            }
            
            # Extract entities from context
            if hasattr(graph_context, 'context_records') and graph_context.context_records:
                entities_df = graph_context.context_records.get('entities', pd.DataFrame())
                if not entities_df.empty:
                    network_data["entities"] = entities_df.to_dict('records')
                
                relationships_df = graph_context.context_records.get('relationships', pd.DataFrame())
                if not relationships_df.empty:
                    network_data["relationships"] = relationships_df.to_dict('records')
                
                text_units_df = graph_context.context_records.get('text_units', pd.DataFrame())
                if not text_units_df.empty:
                    network_data["text_units"] = text_units_df.to_dict('records')
                
                community_reports_df = graph_context.context_records.get('reports', pd.DataFrame())
                if not community_reports_df.empty:
                    network_data["community_reports"] = community_reports_df.to_dict('records')
            
            # Add context chunks as text summary
            if hasattr(graph_context, 'context_chunks') and graph_context.context_chunks:
                network_data["context_summary"] = graph_context.context_chunks
            
            # If no entities found in context records, try to get from context builder directly
            if not network_data["entities"] and hasattr(self.context_builder, 'entities'):
                # Extract entities directly from context builder
                entities_list = []
                for entity_id, entity in self.context_builder.entities.items():
                    entity_dict = {
                        'id': entity.id,
                        'title': getattr(entity, 'title', ''),
                        'rank': getattr(entity, 'rank', 0),
                        'description': getattr(entity, 'description', '')
                    }
                    entities_list.append(entity_dict)
                network_data["entities"] = entities_list
                
                # Extract relationships directly from context builder
                if hasattr(self.context_builder, 'relationships'):
                    relationships_list = []
                    for rel_id, rel in self.context_builder.relationships.items():
                        rel_dict = {
                            'id': rel.id,
                            'source': rel.source,
                            'target': rel.target,
                            'type': getattr(rel, 'type', ''),
                            'weight': getattr(rel, 'weight', 1.0)
                        }
                        relationships_list.append(rel_dict)
                    network_data["relationships"] = relationships_list
                
                # Extract community reports directly from context builder
                if hasattr(self.context_builder, 'community_reports'):
                    reports_list = []
                    for comm_id, comm in self.context_builder.community_reports.items():
                        comm_dict = {
                            'id': comm.community_id,
                            'description': getattr(comm, 'description', ''),
                            'summary': getattr(comm, 'summary', '')
                        }
                        reports_list.append(comm_dict)
                    network_data["community_reports"] = reports_list
            
            # Format as JSON string for prompt insertion
            formatted_data = json.dumps(network_data, indent=2, ensure_ascii=False)
            
            logger.info(f"Formatted network data with {len(network_data['entities'])} entities, {len(network_data['relationships'])} relationships")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Failed to format network data: {e}")
            raise CausalSearchError(f"Network data formatting failed: {e}")

    async def _generate_causal_report(self, network_data: str) -> str:
        """Generate causal analysis report using causal_discovery_prompt.txt."""
        try:
            # Load and format the causal discovery prompt
            prompt_template = self.causal_discovery_prompt
            formatted_prompt = prompt_template.format(graph_data=network_data)
            
            # Call LLM to generate causal report
            response = await self.model.achat(
                prompt="Generate a causal analysis report",
                history=[{"role": "system", "content": formatted_prompt}],
                model_parameters=self.model_params
            )
            
            # Extract content from response
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Generated causal report of length {len(response_content)}")
            return response_content
            
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            raise CausalSearchError(f"Causal discovery stage failed: {e}")

    async def _generate_final_response(self, causal_report: str, user_query: str) -> str:
        """Generate final response using causal_summary_report.txt."""
        try:
            # Load and format the causal summary prompt
            prompt_template = self.causal_summary_prompt
            formatted_prompt = prompt_template.format(
                causal_summary=causal_report,
                query=user_query,
                response_type=self.response_type
            )
            
            # Call LLM to generate final response
            response = await self.model.achat(
                prompt=user_query,
                history=[{"role": "system", "content": formatted_prompt}],
                model_parameters=self.model_params
            )
            
            # Extract content from response
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Generated final response of length {len(response_content)}")
            return response_content
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise CausalSearchError(f"Response generation stage failed: {e}")

    async def _save_outputs(self, network_data: str, causal_report: str, query: str):
        """Save outputs to data folder if enabled in configuration."""
        try:
            # Check if output saving is enabled
            if not hasattr(self, 'context_builder_params') or not self.context_builder_params:
                logger.warning("No context builder params available, skipping output saving")
                return
                
            save_network_data = self.context_builder_params.get('save_network_data', False)
            save_causal_report = self.context_builder_params.get('save_causal_report', False)
            output_folder = self.context_builder_params.get('output_folder', 'causal_search')
            
            if not save_network_data and not save_causal_report:
                logger.info("Output saving disabled in configuration")
                return
            
            # Create outputs directory
            outputs_dir = Path("data/outputs") / output_folder
            outputs_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate query ID for file naming
            query_id = self._generate_query_id(query)
            
            # Save network data if enabled
            if save_network_data:
                network_data_file = outputs_dir / f"causal_search_network_data_{query_id}.json"
                with open(network_data_file, 'w', encoding='utf-8') as f:
                    f.write(network_data)
                logger.info(f"Saved network data to {network_data_file}")
            
            # Save causal report if enabled
            if save_causal_report:
                causal_report_file = outputs_dir / f"causal_search_report_{query_id}.md"
                with open(causal_report_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Causal Analysis Report\n\n")
                    f.write(f"**Query:** {query}\n\n")
                    f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(causal_report)
                logger.info(f"Saved causal report to {causal_report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save outputs: {e}")
    
    def _generate_query_id(self, query: str) -> str:
        """Generate a unique query ID for file naming."""
        import hashlib
        # Create a hash of the query for consistent file naming
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
        # Also include timestamp for uniqueness
        timestamp = int(time.time())
        return f"{query_hash}_{timestamp}"

    async def _save_prompts(self):
        """Save prompts to data folder."""
        try:
            # Create prompts directory
            prompts_dir = Path("data/prompts")
            prompts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save causal discovery prompt
            with open(prompts_dir / "causal_discovery_prompt.txt", "w", encoding="utf-8") as f:
                f.write(self.causal_discovery_prompt)
            
            # Save causal summary prompt
            with open(prompts_dir / "causal_summary_prompt.txt", "w", encoding="utf-8") as f:
                f.write(self.causal_summary_prompt)
            
            logger.info("Saved prompts to data/prompts/")
            
        except Exception as e:
            logger.warning(f"Failed to save prompts: {e}")
