# Causal Search Usage Guide

## Overview

Causal Search is a specialized search method in GraphRAG that performs causal analysis on knowledge graphs. It works in two stages:

1. **Stage 1**: Extracts extended graph information (k + s nodes) and generates a causal analysis report
2. **Stage 2**: Uses the causal report to generate a final response to the user's query

## Key Features

- **Extended Node Extraction**: Retrieves k + s nodes where k is the local search limit and s is configurable
- **Two-Stage Processing**: Causal discovery followed by response generation
- **Automatic Output Saving**: Saves network data and causal reports to data folders
- **Configurable Parameters**: Control retrieval breadth, context proportions, and token limits
- **CLI Integration**: Available as `--method causal` in the GraphRAG CLI

## Configuration

### Settings in `settings.yaml`

```yaml
causal_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  # Retrieval breadth control parameters
  s_parameter: 3  # Additional nodes beyond local search for causal analysis
  top_k_mapped_entities: 10  # Number of entities to retrieve from embedding store
  top_k_relationships: 10    # Number of relationships to retrieve per entity
  text_unit_prop: 0.5        # Proportion of context for text units
  community_prop: 0.25       # Proportion of context for community reports
  max_context_tokens: 12000  # Maximum context window size
  # Output control parameters
  save_network_data: true   # Whether to save extracted network data to files
  save_causal_report: true  # Whether to save causal analysis report to files
  output_folder: "causal_search"  # Subfolder under data/outputs/ for causal search outputs
```

### Parameter Descriptions

- **`s_parameter`**: Number of additional nodes to extract beyond the local search limit
- **`top_k_mapped_entities`**: Maximum number of entities to retrieve from the embedding store
- **`top_k_relationships`**: Maximum number of relationships to retrieve per entity
- **`text_unit_prop`**: Proportion of context window dedicated to text units (0.0 to 1.0)
- **`community_prop`**: Proportion of context window dedicated to community reports (0.0 to 1.0)
- **`max_context_tokens`**: Maximum token limit for the context window
- **`save_network_data`**: Whether to save extracted network data as JSON files (true/false)
- **`save_causal_report`**: Whether to save causal analysis reports as Markdown files (true/false)
- **`output_folder`**: Subfolder name under `data/outputs/` for storing causal search outputs

## Usage Methods

### 1. CLI Usage

```bash
# Basic causal search
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "Who is Scrooge and what are his main relationships?"

# With streaming
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "Analyze the causal factors" \
  --streaming

# With custom response type
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "What causes X?" \
  --response-type "Detailed Report"
```

### 2. Python API Usage

```python
import asyncio
from graphrag.query.factory import get_causal_search_engine
from graphrag.config.load_config import load_config

async def run_causal_search():
    # Load configuration
    config = load_config("ragtest", "settings.yaml")
    
    # Get your data
    entities = [...]  # Your entity list
    relationships = [...]  # Your relationship list
    text_units = [...]  # Your text units
    community_reports = [...]  # Your community reports
    covariates = {...}  # Your covariates
    
    # Create causal search engine
    causal_search = get_causal_search_engine(
        config=config,
        reports=community_reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        response_type="Multiple Paragraphs",
        description_embedding_store=your_embedding_store
    )
    
    # Run search
    result = await causal_search.search(
        query="What are the causal relationships?",
        top_k_mapped_entities=15,  # Override config if needed
        top_k_relationships=15
    )
    
    return result

# Run it
result = asyncio.run(run_causal_search())
```

### 3. Two-Stage Processing

The causal search method automatically performs two stages:

#### Stage 1: Causal Discovery
- Extracts k + s nodes from the knowledge graph
- Builds comprehensive graph context
- Generates causal analysis report using `causal_discovery_prompt.txt`
- Saves network data to `data/outputs/causal_search_network_data.json`
- Saves causal report to `data/outputs/causal_search_report.md`

#### Stage 2: Response Generation
- Uses the causal report as context
- Generates final response using `causal_summary_report.txt`
- Combines causal insights with user query
- Returns comprehensive answer

## Output Files

### Automatic Outputs

The method automatically saves outputs to the configured output directory with unique filenames:

```
output/
└── causal_search/
    ├── causal_search_network_data_<query_id>.json    # Extracted graph information
    └── causal_search_report_<query_id>.md           # Causal analysis report
```

Where `<query_id>` is a unique identifier (e.g., `a70cc37d_1755454480`) that ensures each query's outputs are distinct. The query ID format is:
- **First 8 characters**: Hash of the query content (e.g., `a70cc37d`)
- **Last 10 characters**: Unix timestamp when the query was processed (e.g., `1755454480`)

This naming scheme allows you to:
- **Track specific queries** by identifying them through the hash prefix
- **Sort files chronologically** using the timestamp suffix
- **Avoid filename conflicts** when running multiple queries

### Prompt Storage

Prompts are stored as Python constants in the GraphRAG codebase:

```
graphrag/
└── prompts/
    └── query/
        ├── causal_discovery_prompt.py    # Causal discovery prompt
        └── causal_summary_prompt.py      # Response generation prompt
```

### Network Data Format

The `causal_search_network_data.json` contains:

```json
{
  "entities": [
    {
      "id": "entity_id",
      "title": "Entity Name",
      "description": "Entity description",
      "type": "entity_type"
    }
  ],
  "relationships": [
    {
      "source": "source_entity",
      "target": "target_entity",
      "description": "Relationship description",
      "weight": 1.0
    }
  ],
  "text_units": [
    {
      "id": "text_unit_id",
      "text": "Text content",
      "document_ids": ["doc1", "doc2"]
    }
  ],
  "community_reports": [
    {
      "community_id": "community_id",
      "summary": "Community summary",
      "rank": 1
    }
  ]
}
```

### Causal Report Format

The `causal_search_report.md` contains:

```markdown
# Causal Analysis Report

**Query:** [User's query]

**Generated:** [Timestamp]

[Structured causal analysis with sections for:]
1. Introduction
2. Key Entities and Their Roles
3. Major Causal Pathways
4. Confidence and Evidence Strength
5. Implications and Recommendations
```

## Advanced Usage

### Customizing Search Parameters

```python
# Override configuration parameters
result = await causal_search.search(
    query="What causes X to happen?",
    top_k_mapped_entities=20,  # Override default 10
    top_k_relationships=15,     # Override default 10
    text_unit_prop=0.6,        # Override default 0.5
    community_prop=0.2,         # Override default 0.25
    max_context_tokens=15000    # Override default 12000
)
```

### Manual Causal Summary Insertion

For advanced users who want to provide their own causal summary:

```python
# Create causal search with custom context
causal_search = get_causal_search_engine(
    config=config,
    reports=community_reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=covariates,
    response_type="Multiple Paragraphs",
    description_embedding_store=your_embedding_store
)

# Extract graph information only (Stage 1)
graph_context = await causal_search._extract_graph_information(selected_nodes)
network_data = causal_search._format_network_data_for_causal_prompt(graph_context)

# Use external tool to generate causal summary
external_causal_summary = your_external_causal_analysis_tool(network_data)

# Generate final response with external summary (Stage 2)
final_response = await causal_search._generate_final_response(
    external_causal_summary, 
    "Your query here"
)
```

### Error Handling

```python
from graphrag.query.structured_search.causal_search.search import CausalSearchError

try:
    result = await causal_search.search("Your query here")
except CausalSearchError as e:
    print(f"Causal search failed: {e}")
    # Handle the error appropriately
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Troubleshooting

### Common Issues

1. **Prompt Loading Errors**: Ensure prompt files exist in `graphrag/prompts/query/causal_search/`
2. **Configuration Errors**: Verify all required parameters are set in `settings.yaml`
3. **Memory Issues**: Reduce `max_context_tokens` if you encounter memory problems
4. **Performance Issues**: Adjust `s_parameter` and `top_k` values based on your needs

### Debug Mode

Enable verbose logging to see detailed information:

```bash
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "Your query" \
  --verbose
```

## Integration with Other Workflows

Causal Search integrates seamlessly with the GraphRAG pipeline:

1. **Indexing**: Run your normal GraphRAG indexing workflow
2. **Causal Search**: Use the generated entities, relationships, and community reports
3. **Outputs**: Access all outputs through the standard GraphRAG storage system

## Best Practices

1. **Start with Defaults**: Begin with the default configuration and adjust as needed
2. **Monitor Token Usage**: Keep `max_context_tokens` within your model's limits
3. **Balance Proportions**: Ensure `text_unit_prop + community_prop <= 1.0`
4. **Test Incrementally**: Start with small `s_parameter` values and increase gradually
5. **Review Outputs**: Check the generated reports to understand the causal analysis quality

## Examples

### Example 1: Basic Causal Analysis

```bash
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "What are the main causal factors in this dataset?"
```

### Example 2: Detailed Causal Investigation

```bash
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "How do different factors influence the outcome?" \
  --response-type "Comprehensive Analysis"
```

### Example 3: Streaming Causal Search

```bash
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "Analyze causal patterns" \
  --streaming
```

## Support

For issues or questions about Causal Search:

1. Check the generated output files for error details
2. Review the configuration parameters
3. Enable verbose logging for debugging
4. Check the GraphRAG documentation for general troubleshooting

---

*This documentation covers the Causal Search method in GraphRAG. For more information about other search methods, see the main GraphRAG documentation.*
