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
        
        logger.info(f"🚀 Starting causal search for query: '{query}'")
        logger.info(f"📊 Parameters: s_parameter={self.s_parameter}, max_context_tokens={self.max_context_tokens}")
        
        try:
            # Step 1: Extract extended nodes (k + s) * oversample_scaler
            local_search_top_k = kwargs.get('top_k_mapped_entities', 10)
            logger.info(f"🔍 Step 1: Extracting extended nodes with k={local_search_top_k}, s={self.s_parameter}")
            
            extended_nodes = await self._extract_extended_nodes(
                query, local_search_top_k, **kwargs
            )
            logger.info(f"✅ Step 1 complete: Found {len(extended_nodes)} extended nodes")
            
            # Step 2: Extract graph information using local search components
            logger.info(f"🔍 Step 2: Extracting graph information for {len(extended_nodes)} nodes")
            graph_context = await self._extract_graph_information(
                extended_nodes, **kwargs
            )
            logger.info(f"✅ Step 2 complete: Graph context extracted")
            
            # Step 3: Format network data for causal discovery prompt
            logger.info(f"🔍 Step 3: Formatting network data for causal discovery prompt")
            network_data = self._format_network_data_for_causal_prompt(graph_context)
            logger.info(f"✅ Step 3 complete: Network data formatted ({len(network_data)} characters)")
            
            # Step 4: Generate causal report (Stage 1)
            logger.info(f"🔍 Step 4: Generating causal report using LLM")
            causal_report = await self._generate_causal_report(network_data)
            logger.info(f"✅ Step 4 complete: Causal report generated ({len(causal_report)} characters)")
            
            # Step 5: Generate final response (Stage 2)
            logger.info(f"🔍 Step 5: Generating final response using LLM")
            final_response = await self._generate_final_response(causal_report, query)
            logger.info(f"✅ Step 5 complete: Final response generated ({len(final_response)} characters)")
            
            # Save outputs if configured
            logger.info(f"🔍 Step 6: Saving outputs if configured")
            await self._save_outputs(network_data, causal_report, query)
            logger.info(f"✅ Step 6 complete: Outputs saved")
            
            # Calculate token usage
            prompt_tokens["causal_discovery"] = num_tokens(network_data, self.token_encoder)
            prompt_tokens["response_generation"] = num_tokens(causal_report, self.token_encoder)
            output_tokens["causal_discovery"] = num_tokens(causal_report, self.token_encoder)
            output_tokens["response_generation"] = num_tokens(final_response, self.token_encoder)

            for callback in self.callbacks:
                callback.on_context(graph_context.context_records)

            logger.info(f"🎉 Causal search completed successfully in {time.time() - start_time:.2f}s")
            
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
            logger.error(f"❌ Causal search failed: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
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
        """Extract (k+s) * oversample_scaler nodes where k = local_search_top_k, s = self.s_parameter."""
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
            # Calculate total nodes needed: (k + s) * oversample_scaler
            s_parameter = getattr(self, 's_parameter', 3)
            total_nodes_needed = (top_k + s_parameter) * 2  # oversample_scaler = 2
            
            # Use the context builder to get local search nodes
            # This mimics the local search entity extraction process
            # Create a copy of params and override top_k_mapped_entities
            params = dict(self.context_builder_params)
            params['top_k_mapped_entities'] = total_nodes_needed  # Request (k + s) * oversample_scaler nodes
            context_result = self.context_builder.build_context(
                query=query,
                **params
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
                    return entity_ids[:top_k]  # Return only the top k nodes for local search
            
            # If no entities found in context, try to extract from the context builder directly
            # Cast to LocalSearchMixedContext to access the attributes
            from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
            if isinstance(self.context_builder, LocalSearchMixedContext):
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
            
            # Cast to LocalSearchMixedContext to access the attributes
            from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
            if not isinstance(self.context_builder, LocalSearchMixedContext):
                logger.warning("Context builder is not LocalSearchMixedContext, cannot extract additional nodes")
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
                    # Use getattr to safely access attributes that may not exist
                    community_desc = getattr(community, 'description', None)
                    if community_desc:
                        # Simple entity extraction from description
                        # This could be enhanced with more sophisticated NLP
                        for entity_name in self.context_builder.entities.keys():
                            if entity_name in community_desc and entity_name not in additional_nodes:
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
                logger.debug(f"Context records keys: {list(graph_context.context_records.keys())}")
                
                entities_df = graph_context.context_records.get('entities', pd.DataFrame())
                if not entities_df.empty:
                    network_data["entities"] = entities_df.to_dict('records')
                    logger.debug(f"Found {len(entities_df)} entities in context records")
                
                relationships_df = graph_context.context_records.get('relationships', pd.DataFrame())
                if not relationships_df.empty:
                    network_data["relationships"] = relationships_df.to_dict('records')
                    logger.debug(f"Found {len(relationships_df)} relationships in context records")
                else:
                    logger.debug("No relationships found in context records")
                
                text_units_df = graph_context.context_records.get('text_units', pd.DataFrame())
                if not text_units_df.empty:
                    network_data["text_units"] = text_units_df.to_dict('records')
                    logger.debug(f"Found {len(text_units_df)} text units in context records")
                else:
                    logger.debug("No text units found in context records")
                
                community_reports_df = graph_context.context_records.get('reports', pd.DataFrame())
                if not community_reports_df.empty:
                    network_data["community_reports"] = community_reports_df.to_dict('records')
            
            # Add context chunks as text summary
            if hasattr(graph_context, 'context_chunks') and graph_context.context_chunks:
                network_data["context_summary"] = graph_context.context_chunks
            
            # If no entities found in context records, try to get from context builder directly
            if not network_data["entities"] and hasattr(self.context_builder, 'entities'):
                logger.debug(f"No entities in context records, trying context builder with {len(self.context_builder.entities)} entities")
                # Extract entities directly from context builder
                entities_list = []
                for entity_id, entity in self.context_builder.entities.items():
                    entity_dict = {
                        'id': entity.id,
                        'short_id': getattr(entity, 'short_id', None),
                        'title': getattr(entity, 'title', ''),
                        'type': getattr(entity, 'type', None),
                        'description': getattr(entity, 'description', ''),
                        'description_embedding': getattr(entity, 'description_embedding', None),
                        'name_embedding': getattr(entity, 'name_embedding', None),
                        'community_ids': getattr(entity, 'community_ids', None),
                        'text_unit_ids': getattr(entity, 'text_unit_ids', None),
                        'rank': getattr(entity, 'rank', 1),
                        'attributes': getattr(entity, 'attributes', None),
                    }
                    entities_list.append(entity_dict)
                network_data["entities"] = entities_list
                logger.debug(f"Extracted {len(entities_list)} entities from context builder")
                
            # Extract relationships directly from context builder (even if entities were found in context records)
            if not network_data["relationships"] and hasattr(self.context_builder, 'relationships'):
                logger.debug(f"No relationships in context records, trying context builder with {len(self.context_builder.relationships)} relationships")
                relationships_list = []
                for rel_id, rel in self.context_builder.relationships.items():
                    rel_dict = {
                        'id': rel.id,
                        'short_id': getattr(rel, 'short_id', None),
                        'source': rel.source,
                        'target': rel.target,
                        'weight': getattr(rel, 'weight', 1.0),
                        'description': getattr(rel, 'description', None),
                        'description_embedding': getattr(rel, 'description_embedding', None),
                        'text_unit_ids': getattr(rel, 'text_unit_ids', None),
                        'rank': getattr(rel, 'rank', 1),
                        'attributes': getattr(rel, 'attributes', None),
                    }
                    relationships_list.append(rel_dict)
                network_data["relationships"] = relationships_list
                logger.debug(f"Extracted {len(relationships_list)} relationships from context builder")
                
                # Extract text units directly from context builder (even if other data was found in context records)
                if not network_data["text_units"] and hasattr(self.context_builder, 'text_units'):
                    logger.debug(f"No text units in context records, trying context builder with {len(self.context_builder.text_units)} text units")
                    text_units_list = []
                    for unit_id, unit in self.context_builder.text_units.items():
                        unit_dict = {
                            'id': unit.id,
                            'short_id': getattr(unit, 'short_id', None),
                            'text': getattr(unit, 'text', ''),
                            'entity_ids': getattr(unit, 'entity_ids', None),
                            'relationship_ids': getattr(unit, 'relationship_ids', None),
                            'covariate_ids': getattr(unit, 'covariate_ids', None),
                            'n_tokens': getattr(unit, 'n_tokens', None),
                            'document_ids': getattr(unit, 'document_ids', None),
                            'attributes': getattr(unit, 'attributes', None),
                        }
                        text_units_list.append(unit_dict)
                    network_data["text_units"] = text_units_list
                    logger.debug(f"Extracted {len(text_units_list)} text units from context builder")
                
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
            if hasattr(response, 'content') and response.content:
                response_content = response.content
                logger.debug(f"Extracted content successfully from response.content")
            else:
                logger.warning(f"Response object has no content attribute or content is empty. Response type: {type(response)}")
                # Try to extract from string representation if it's BaseModelOutput format
                response_str = str(response)
                if 'BaseModelOutput(content=' in response_str:
                    # Extract content from the string representation
                    import re
                    # Handle multi-line content properly
                    match = re.search(r"content='(.*?)', full_response=", response_str, re.DOTALL)
                    if match:
                        response_content = match.group(1).replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                        logger.debug(f"Extracted content from string representation")
                    else:
                        response_content = response_str
                        logger.warning(f"Could not extract content from BaseModelOutput string")
                else:
                    response_content = response_str
            
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
            if hasattr(response, 'content') and response.content:
                response_content = response.content
                logger.debug(f"Extracted final response content successfully from response.content")
            else:
                logger.warning(f"Final response object has no content attribute or content is empty. Response type: {type(response)}")
                # Try to extract from string representation if it's BaseModelOutput format
                response_str = str(response)
                if 'BaseModelOutput(content=' in response_str:
                    # Extract content from the string representation
                    import re
                    # Handle multi-line content properly
                    match = re.search(r"content='(.*?)', full_response=", response_str, re.DOTALL)
                    if match:
                        response_content = match.group(1).replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                        logger.debug(f"Extracted final response content from string representation")
                    else:
                        response_content = response_str
                        logger.warning(f"Could not extract final response content from BaseModelOutput string")
                else:
                    response_content = response_str
            
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
            
            # Create outputs directory using the configured output base directory
            output_base_dir = self.context_builder_params.get('output_base_dir', 'output')
            outputs_dir = Path(output_base_dir) / output_folder
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
