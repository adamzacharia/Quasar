---
title: 'QUASAR: A Radio Astronomy Intelligence System'
tags:
  - Python
  - astronomy
  - radio astronomy
  - ALMA
  - LLM
  - AI
authors:
  - name: Quasar Contributors
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 04 December 2025
bibliography: paper.bib
---

# Summary

QUASAR (Query System for Automated Science and Astronomy Research) is an AI-powered assistant designed to democratize access to radio astronomy data. It integrates Large Language Models (LLMs) with the ALMA (Atacama Large Millimeter/submillimeter Array) archive, enabling researchers and students to query complex astronomical databases using natural language.

# Statement of need

Modern astronomical archives are vast and complex. Accessing data from facilities like ALMA typically requires knowledge of specialized query languages (ADQL) or familiarity with complex web interfaces. This creates a barrier to entry for students and researchers from other fields.

QUASAR bridges this gap by providing a conversational interface that:
1.  Translates natural language questions into precise database queries.
2.  Retrieves and filters data from the NRAO/ALMA archives.
3.  Provides context-aware summaries and explanations using RAG (Retrieval Augmented Generation) on technical documentation.

By automating the technical intricacies of data retrieval, QUASAR allows astronomers to focus on the science rather than the syntax.

# Functionality

QUASAR features a modular architecture:
-   **Core Agent**: Orchestrates tool usage and manages conversation state.
-   **Integration Layer**: Connects to external services like `alminer` and `pyvo` for archive access.
-   **RAG Engine**: Indexes technical manuals (e.g., ALMA Proposer's Guide) to answer domain-specific questions.
-   **Streamlit UI**: Provides an interactive, chat-based user interface.

# References

None yet.
