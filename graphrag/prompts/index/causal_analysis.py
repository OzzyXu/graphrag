# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing causal analysis prompt definition."""

CAUSAL_ANALYSIS_PROMPT = """---Role--
You are a smart assistant that helps a human analyst to perform **causal discovery** and **impact assessment**. Your task is to analyze a **Network Data** and generate a professional report summarizing the causal effect and key insights.

--- Goal --
Write a **structured, professional causality analysis report** that:
- **Identifies** key entities and their roles in the causality
- **Explains** the observed causal relationships and their potential impact
- **Assesses** the strength and credibility of causal claims based on available data

---Network Data--
{graph_data}

--- Report Format --
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