# Quasar Architecture Diagrams

> Auto-generated architecture diagrams. View in any Mermaid-compatible renderer (GitHub, VS Code with Mermaid plugin, etc.)

## Diagrams

| # | Diagram | Description |
| --- | --- | --- |
| 1 | [High-Level Architecture](01_high_level_architecture.md) | Package-to-package dependency overview |
| 2 | [Request Data Flow](02_data_flow.md) | How a user query flows through the system |
| 3 | [Core Internals](03_core_internals.md) | All 27 core modules and connections |
| 4 | [Services Internals](04_services_internals.md) | All 31 service modules |
| 5 | [Agent Hierarchy](05_agent_hierarchy.md) | Sub-agent class tree |
| 6 | [Integrations](06_integrations.md) | External API client map |
| 7 | [Conductor Flow](07_conductor_sequence.md) | DAG orchestration sequence |
| 8 | [Full Import Map](08_full_import_map.md) | Every module, every import edge |

## Regeneration

```bash
conda run -n quasar python scripts/generate_mermaid_graphs.py
```
