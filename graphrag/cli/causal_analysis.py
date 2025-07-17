# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI for causal analysis."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import typer

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.workflows.causal_analysis import run_workflow
from graphrag.config.load_config import load_config
from graphrag.storage.factory import StorageFactory

logger = logging.getLogger(__name__)


def causal_analysis_cli(
    root_dir: Path,
    verbose: bool,
    config_filepath: Path | None,
    output_dir: Path | None,
) -> None:
    """Run causal analysis on the knowledge graph."""
    # Set up logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load configuration
    config = load_config(root_dir, config_filepath)
    
    # Override output directory if specified
    if output_dir:
        config.output.base_dir = str(output_dir)

    # Run causal analysis
    asyncio.run(_run_causal_analysis(config))


async def _run_causal_analysis(config: GraphRagConfig) -> None:
    """Run the causal analysis workflow."""
    logger.info("Starting causal analysis...")
    
    try:
        # Create storage and context using utilities
        from graphrag.utils.api import create_storage_from_config, create_cache_from_config
        from graphrag.index.run.utils import create_run_context
        
        input_storage = create_storage_from_config(config.input.storage)
        output_storage = create_storage_from_config(config.output)
        cache = create_cache_from_config(config.cache, config.root_dir)
        
        context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
            cache=cache,
        )
        
        # Run the causal analysis workflow
        result = await run_workflow(config, context)
        
        # Print results
        print("\n" + "="*80)
        print("CAUSAL ANALYSIS RESULTS")
        print("="*80)
        
        if result.result and "causal_report" in result.result:
            print("\n📊 CAUSAL ANALYSIS REPORT:")
            print("-" * 40)
            print(result.result["causal_report"])
        
        if result.result and "causal_relationships" in result.result and result.result["causal_relationships"]:
            print("\n🔗 CAUSAL RELATIONSHIPS:")
            print("-" * 40)
            for i, rel in enumerate(result.result["causal_relationships"], 1):
                print(f"{i}. {rel.get('cause', 'Unknown')} → {rel.get('effect', 'Unknown')}")
                if 'description' in rel:
                    print(f"   Description: {rel['description']}")
                print()
        
        if result.result and "confidence_scores" in result.result:
            print("\n📈 CONFIDENCE SCORES:")
            print("-" * 40)
            for key, score in result.result["confidence_scores"].items():
                print(f"{key}: {score:.2f}")
        
        if result.result and "key_entities" in result.result:
            print("\n🎯 KEY ENTITIES:")
            print("-" * 40)
            for i, entity in enumerate(result.result["key_entities"], 1):
                print(f"{i}. {entity}")
        
        print("\n" + "="*80)
        print("Causal analysis completed successfully!")
        print("Results have been saved to the output directory.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        typer.echo(f"❌ Causal analysis failed: {e}", err=True)
        raise typer.Exit(1) 