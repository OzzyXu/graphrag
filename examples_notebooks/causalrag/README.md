# Causal Search Demo

This directory contains a comprehensive demonstration of the new Causal Search method in GraphRAG.

## Contents

- **`causal_search_demo.ipynb`**: Main demonstration notebook showing how to use causal search
- **`README.md`**: This file explaining the contents

## What is Causal Search?

Causal Search is a specialized search method in GraphRAG that performs causal analysis on knowledge graphs through a two-stage process:

1. **Stage 1**: Extract extended graph information (k + s nodes) and generate causal analysis report
2. **Stage 2**: Use the causal report to generate final response to user query

## Key Features Demonstrated

- ✅ **Configuration Setup**: Loading GraphRAG configuration and setting up environment
- ✅ **Data Loading**: Loading entities, relationships, text units, and community reports
- ✅ **Engine Creation**: Building the causal search engine with proper parameters
- ✅ **Query Execution**: Running causal search with example queries
- ✅ **Results Analysis**: Examining search results and generated outputs
- ✅ **Output Files**: Checking automatically generated network data and causal reports
- ✅ **Configuration Tuning**: Understanding parameter effects and recommendations

## Prerequisites

Before running the notebook, ensure you have:

1. **GraphRAG Pipeline**: Run the GraphRAG indexing pipeline to generate entities, relationships, and community reports
2. **Configuration**: Set up your configuration in `settings.yaml` with causal search parameters
3. **Language Models**: Configured your language models and API keys
4. **Dependencies**: All required Python packages installed

## Usage

1. **Open the notebook**: `causal_search_demo.ipynb`
2. **Adjust paths**: Modify `ROOT_DIR` to point to your project root
3. **Run cells**: Execute cells sequentially to see the demonstration
4. **Customize**: Modify parameters and queries based on your needs

## CLI Alternative

You can also use causal search from the command line:

```bash
graphrag query \
  --root ./ragtest \
  --method causal \
  --query "What are the causal relationships in this dataset?"
```

## Documentation

For comprehensive usage information, see:
- **`docs/causal_search_usage.md`**: Detailed usage guide with examples
- **Main GraphRAG docs**: General GraphRAG documentation

## Example Queries

The notebook includes several demo queries:
1. "What are the main causal relationships in this dataset?"
2. "How do different factors influence the outcomes?"
3. "What causes the observed patterns in the data?"
4. "Analyze the causal factors driving the main themes."

## Output Files

The causal search automatically generates:
- **Network Data**: `data/outputs/causal_search_network_data.json`
- **Causal Report**: `data/outputs/causal_search_report.md`
- **Used Prompts**: `data/prompts/causal_discovery_prompt.txt` and `causal_summary_prompt.txt`

## Configuration Parameters

Key parameters you can adjust:
- **`s_parameter`**: Additional nodes beyond local search (default: 3)
- **`top_k_mapped_entities`**: Number of entities to retrieve (default: 10)
- **`top_k_relationships`**: Number of relationships per entity (default: 10)
- **`text_unit_prop`**: Proportion for text units (default: 0.5)
- **`community_prop`**: Proportion for community reports (default: 0.25)
- **`max_context_tokens`**: Maximum context window (default: 12000)

## Troubleshooting

Common issues and solutions:
1. **Configuration errors**: Verify all parameters in `settings.yaml`
2. **Data loading failures**: Ensure GraphRAG pipeline has been run
3. **Memory issues**: Reduce `max_context_tokens` if needed
4. **Performance issues**: Adjust `s_parameter` and `top_k` values

## Support

For issues or questions:
1. Check the generated output files for error details
2. Review configuration parameters
3. Enable verbose logging for debugging
4. Check the main GraphRAG documentation

---

*This demo showcases the Causal Search method in GraphRAG. For more information, see the comprehensive documentation.*
