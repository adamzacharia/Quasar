#!/usr/bin/env python3
"""
generate_mermaid_graphs.py — Build comprehensive Mermaid architecture diagrams
for the Quasar codebase by parsing real Python AST (imports, classes, functions).

Generates:
  1. Module-level dependency graph (which file imports which)
  2. Class hierarchy & composition graph
  3. Per-package internal wiring diagrams
  4. Full system graph (everything in one diagram)

Output: docs/architecture/  (one .md file per diagram with embedded Mermaid)
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "architecture"

# Packages to scan (relative to PROJECT_ROOT)
PACKAGES = [
    "core",
    "services",
    "agents",
    "integrations",
    "config",
    "utils",
]

# Top-level entry points
ENTRY_POINTS = [
    "quasar.py",
]

# Files to skip
SKIP_FILES = {"__pycache__", ".pyc", "__init__.py"}

# Package display colors (for subgraph styling)
PACKAGE_COLORS = {
    "core":         "#1a1a2e",
    "services":     "#16213e",
    "agents":       "#0f3460",
    "integrations": "#533483",
    "config":       "#2b2d42",
    "utils":        "#3d405b",
    "entry":        "#e07a5f",
}

# ── AST Parsing ──────────────────────────────────────────────────────────────

def parse_file(filepath: Path) -> dict:
    """Parse a Python file and extract imports, classes, functions."""
    result = {
        "imports": [],          # (module, names)
        "classes": [],          # (class_name, bases, methods)
        "functions": [],        # function_name
        "file": str(filepath.relative_to(PROJECT_ROOT)),
    }
    
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  [SKIP] {filepath}: {e}")
        return result
    
    for node in ast.walk(tree):
        # Imports: from X import Y
        if isinstance(node, ast.ImportFrom):
            if node.module:
                names = [alias.name for alias in node.names]
                result["imports"].append((node.module, names))
        
        # Imports: import X
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append((alias.name, []))
        
        # Classes
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{_get_attr_name(base)}")
            
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_")
            ]
            result["classes"].append((node.name, bases, methods))
        
        # Top-level functions
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Only top-level (not methods inside classes)
            if not any(
                isinstance(parent, ast.ClassDef)
                for parent in ast.walk(tree)
                if node in getattr(parent, 'body', [])
            ):
                if not node.name.startswith("_"):
                    result["functions"].append(node.name)
    
    return result


def _get_attr_name(node) -> str:
    """Recursively get dotted attribute name."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_attr_name(node.value)}.{node.attr}"
    return "?"


# ── Scanning ─────────────────────────────────────────────────────────────────

def scan_project() -> Dict[str, dict]:
    """Scan all Python files in the project and return parsed data."""
    all_files = {}
    
    # Scan packages
    for pkg in PACKAGES:
        pkg_path = PROJECT_ROOT / pkg
        if not pkg_path.exists():
            print(f"  [WARN] Package not found: {pkg}")
            continue
        
        for py_file in sorted(pkg_path.rglob("*.py")):
            if py_file.name in SKIP_FILES or "__pycache__" in str(py_file):
                continue
            rel = str(py_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            print(f"  Parsing: {rel}")
            all_files[rel] = parse_file(py_file)
    
    # Scan entry points
    for ep in ENTRY_POINTS:
        ep_path = PROJECT_ROOT / ep
        if ep_path.exists():
            rel = ep
            print(f"  Parsing: {rel}")
            all_files[rel] = parse_file(ep_path)
    
    # Also scan main.py in ui-pro/api if it exists
    main_api = PROJECT_ROOT / "ui-pro" / "api" / "main.py"
    if main_api.exists():
        rel = "ui-pro/api/main.py"
        print(f"  Parsing: {rel}")
        all_files[rel] = parse_file(main_api)
    
    return all_files


def resolve_import_to_file(module_path: str, all_files: Dict[str, dict]) -> Optional[str]:
    """Resolve a Python import path to a file in our project."""
    # core.conductor -> core/conductor.py
    candidates = [
        module_path.replace(".", "/") + ".py",
        module_path.replace(".", "/") + "/__init__.py",
    ]
    for c in candidates:
        if c in all_files:
            return c
    # Try partial match (e.g., "core.prompts" -> "core/prompts/__init__.py")
    for f in all_files:
        mod = f.replace("/", ".").replace(".py", "").replace(".__init__", "")
        if mod == module_path:
            return f
    return None


# ── Graph Building ───────────────────────────────────────────────────────────

def build_dependency_edges(all_files: Dict[str, dict]) -> List[Tuple[str, str, str]]:
    """Build (source_file, target_file, import_detail) edges from import analysis."""
    edges = []
    
    for src_file, data in all_files.items():
        for module, names in data["imports"]:
            # Only track internal imports (within our packages)
            target = resolve_import_to_file(module, all_files)
            if target and target != src_file:
                detail = ", ".join(names[:3]) if names else module.split(".")[-1]
                edges.append((src_file, target, detail))
    
    return edges


def build_class_edges(all_files: Dict[str, dict]) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[str]]]:
    """Build class inheritance edges and class->methods map."""
    inheritance_edges = []   # (child_class, parent_class, file)
    class_methods = {}       # class_name -> [methods]
    class_locations = {}     # class_name -> file
    
    # First pass: collect all class names and locations
    for filepath, data in all_files.items():
        for cls_name, bases, methods in data["classes"]:
            class_locations[cls_name] = filepath
            class_methods[cls_name] = methods
    
    # Second pass: build inheritance edges
    for filepath, data in all_files.items():
        for cls_name, bases, methods in data["classes"]:
            for base in bases:
                base_simple = base.split(".")[-1]
                if base_simple in class_locations:
                    inheritance_edges.append((cls_name, base_simple, filepath))
    
    return inheritance_edges, class_methods


# ── Mermaid Generation ───────────────────────────────────────────────────────

def sanitize_id(name: str) -> str:
    """Make a string safe for Mermaid node IDs."""
    return name.replace("/", "_").replace(".", "_").replace("-", "_").replace(" ", "_")


def file_label(filepath: str) -> str:
    """Short display label for a file."""
    return filepath.split("/")[-1].replace(".py", "")


def get_package(filepath: str) -> str:
    """Get the package name from a filepath."""
    parts = filepath.split("/")
    if len(parts) > 1 and parts[0] in PACKAGES:
        return parts[0]
    return "entry"


def generate_module_dependency_graph(all_files, edges) -> str:
    """Generate the full module-level dependency graph."""
    lines = [
        "```mermaid",
        "graph LR",
        "",
    ]
    
    # Group files by package for subgraphs
    packages = defaultdict(list)
    for f in all_files:
        packages[get_package(f)].append(f)
    
    # Emit subgraphs
    for pkg in sorted(packages.keys()):
        files = sorted(packages[pkg])
        pkg_label = pkg.upper() if pkg != "entry" else "ENTRY POINTS"
        lines.append(f"    subgraph {sanitize_id(pkg)}[\"{pkg_label}\"]")
        for f in files:
            fid = sanitize_id(f)
            label = file_label(f)
            lines.append(f"        {fid}[\"{label}\"]")
        lines.append("    end")
        lines.append("")
    
    # Emit edges (deduplicated)
    seen = set()
    for src, tgt, detail in edges:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            sid = sanitize_id(src)
            tid = sanitize_id(tgt)
            # Truncate long labels
            short_detail = detail[:25] + "..." if len(detail) > 25 else detail
            lines.append(f"    {sid} -->|{short_detail}| {tid}")
    
    lines.append("```")
    return "\n".join(lines)


def generate_class_hierarchy_graph(all_files, inheritance_edges, class_methods) -> str:
    """Generate class hierarchy and composition diagram."""
    lines = [
        "```mermaid",
        "classDiagram",
        "",
    ]
    
    # Collect all classes with their file location for namespacing
    class_pkg = {}
    for filepath, data in all_files.items():
        pkg = get_package(filepath)
        for cls_name, bases, methods in data["classes"]:
            class_pkg[cls_name] = pkg
    
    # Emit namespace groupings
    pkg_classes = defaultdict(list)
    for cls, pkg in class_pkg.items():
        pkg_classes[pkg].append(cls)
    
    for pkg in sorted(pkg_classes.keys()):
        classes = sorted(pkg_classes[pkg])
        lines.append(f"    namespace {pkg} {{")
        for cls in classes:
            lines.append(f"        class {cls}")
        lines.append("    }")
        lines.append("")
    
    # Emit class members (limit to 8 methods per class for readability)
    for cls_name, methods in sorted(class_methods.items()):
        for method in methods[:8]:
            lines.append(f"    {cls_name} : +{method}()")
        if len(methods) > 8:
            lines.append(f"    {cls_name} : +... {len(methods)-8} more")
    
    lines.append("")
    
    # Emit inheritance edges
    for child, parent, filepath in inheritance_edges:
        lines.append(f"    {parent} <|-- {child}")
    
    lines.append("```")
    return "\n".join(lines)


def generate_package_graph(pkg_name: str, all_files, edges) -> str:
    """Generate internal wiring diagram for a single package."""
    # Filter to only files and edges within this package
    pkg_files = {f: d for f, d in all_files.items() if get_package(f) == pkg_name}
    
    if not pkg_files:
        return ""
    
    lines = [
        "```mermaid",
        "graph TD",
        "",
    ]
    
    # Emit nodes with class/function details
    for f, data in sorted(pkg_files.items()):
        fid = sanitize_id(f)
        label = file_label(f)
        
        # Build a rich label with classes and key functions
        parts = [f"<b>{label}</b>"]
        for cls_name, bases, methods in data["classes"][:4]:
            base_str = f" : {bases[0]}" if bases else ""
            parts.append(f"📦 {cls_name}{base_str}")
        for fn in data["functions"][:4]:
            parts.append(f"⚡ {fn}()")
        
        rich_label = "<br/>".join(parts)
        lines.append(f"    {fid}[\"{rich_label}\"]")
    
    lines.append("")
    
    # Internal edges (within this package)
    seen = set()
    for src, tgt, detail in edges:
        if get_package(src) == pkg_name and get_package(tgt) == pkg_name:
            key = (src, tgt)
            if key not in seen:
                seen.add(key)
                lines.append(f"    {sanitize_id(src)} --> {sanitize_id(tgt)}")
    
    # External edges (going OUT of this package)
    lines.append("")
    lines.append("    %% External dependencies")
    ext_targets = set()
    for src, tgt, detail in edges:
        if get_package(src) == pkg_name and get_package(tgt) != pkg_name:
            key = (src, tgt)
            if key not in seen:
                seen.add(key)
                tid = sanitize_id(tgt)
                if tgt not in ext_targets:
                    ext_targets.add(tgt)
                    lines.append(f"    {tid}[\"{file_label(tgt)}<br/><i>{get_package(tgt)}</i>\"]:::external")
                lines.append(f"    {sanitize_id(src)} -.-> {tid}")
    
    lines.append("")
    lines.append("    classDef external fill:#444,stroke:#888,color:#ccc")
    lines.append("```")
    return "\n".join(lines)


def generate_high_level_architecture(all_files, edges) -> str:
    """Generate a high-level package-to-package dependency graph."""
    # Aggregate edges at the package level
    pkg_edges = defaultdict(int)
    for src, tgt, _ in edges:
        src_pkg = get_package(src)
        tgt_pkg = get_package(tgt)
        if src_pkg != tgt_pkg:
            pkg_edges[(src_pkg, tgt_pkg)] += 1
    
    # Count files per package
    pkg_counts = defaultdict(int)
    for f in all_files:
        pkg_counts[get_package(f)] += 1
    
    lines = [
        "```mermaid",
        "graph TD",
        "",
    ]
    
    # Nodes
    for pkg in sorted(pkg_counts.keys()):
        pid = sanitize_id(pkg)
        label = pkg.upper()
        count = pkg_counts[pkg]
        lines.append(f"    {pid}[\"{label}<br/>{count} modules\"]")
    
    lines.append("")
    
    # Edges with weight
    for (src_pkg, tgt_pkg), count in sorted(pkg_edges.items(), key=lambda x: -x[1]):
        sid = sanitize_id(src_pkg)
        tid = sanitize_id(tgt_pkg)
        thickness = "==>" if count >= 5 else "-->"
        lines.append(f"    {sid} {thickness}|{count} imports| {tid}")
    
    lines.append("")
    
    # Styling
    lines.append("    style core fill:#e63946,color:#fff,stroke:#fff")
    lines.append("    style services fill:#457b9d,color:#fff,stroke:#fff")
    lines.append("    style agents fill:#2a9d8f,color:#fff,stroke:#fff")
    lines.append("    style integrations fill:#e9c46a,color:#000,stroke:#000")
    lines.append("    style config fill:#264653,color:#fff,stroke:#fff")
    lines.append("    style utils fill:#606c38,color:#fff,stroke:#fff")
    lines.append("    style entry fill:#f4a261,color:#000,stroke:#000")
    
    lines.append("```")
    return "\n".join(lines)


def generate_data_flow_graph(all_files) -> str:
    """Generate a user-request data flow diagram showing how a query flows through the system."""
    return """```mermaid
graph TD
    User["🧑‍🔬 User"] --> UI["ui-pro/api/main.py<br/><i>FastAPI + SSE</i>"]
    UI --> Agent["core/agent.py<br/><i>QuasarAgent</i>"]

    Agent --> RLM["core/rlm.py<br/><i>Complexity Detection</i>"]
    RLM -->|simple| DirectLLM["OpenAI API<br/><i>Direct Response</i>"]
    RLM -->|complex| Conductor["core/conductor.py<br/><i>DAG Orchestration</i>"]

    Conductor --> TaskDAG["core/task_dag.py<br/><i>Dependency Graph</i>"]
    Conductor --> ModelRouter["core/model_router.py<br/><i>Model Selection</i>"]
    Conductor --> Recovery["core/recovery.py<br/><i>Error Recovery</i>"]
    Conductor --> WorkflowMem["core/workflow_memory.py<br/><i>Shared State</i>"]

    Agent --> ToolRegistry["core/tools.py<br/><i>27+ Tools</i>"]
    ToolRegistry --> SearchSvc["services/search.py"]
    ToolRegistry --> RAGSvc["services/rag_service.py"]
    ToolRegistry --> ADSSvc["services/ads_service.py"]
    ToolRegistry --> PlotSvc["services/plotting.py"]
    ToolRegistry --> FITSSvc["services/fits_service.py"]
    ToolRegistry --> NotebookSvc["services/notebook_gen.py"]
    ToolRegistry --> BrowserSvc["services/browser.py"]
    ToolRegistry --> CalcSvc["services/astro_calculators.py"]

    SearchSvc --> TAPClient["integrations/tap.py<br/><i>ALMA TAP</i>"]
    SearchSvc --> MASTClient["integrations/mast_client.py<br/><i>JWST/HST</i>"]
    SearchSvc --> ESOClient["integrations/eso_tap_client.py<br/><i>ESO/VLT</i>"]
    SearchSvc --> IRSAClient["integrations/irsa_client.py<br/><i>WISE/2MASS</i>"]

    ADSSvc --> ADSClient["integrations/ads_client.py<br/><i>NASA ADS</i>"]

    RAGSvc --> VectorDB["services/vector_db.py<br/><i>Qdrant</i>"]

    Agent --> ContextMgr["core/context_manager.py<br/><i>Token Management</i>"]
    Agent --> SessionMem["core/session_memory.py<br/><i>Cross-Session</i>"]
    Agent --> Memory["core/memory.py<br/><i>Conversation Buffer</i>"]

    Agent --> LLMClient["core/llm_client.py<br/><i>Multi-Provider</i>"]
    LLMClient --> OpenAI["OpenAI API"]
    LLMClient --> Claude["Anthropic API"]
    LLMClient --> Gemini["Google API"]

    style User fill:#f4a261,color:#000
    style Agent fill:#e63946,color:#fff
    style Conductor fill:#e63946,color:#fff
    style UI fill:#457b9d,color:#fff
    style RLM fill:#e63946,color:#fff

    style SearchSvc fill:#457b9d,color:#fff
    style RAGSvc fill:#457b9d,color:#fff
    style ADSSvc fill:#457b9d,color:#fff
    style PlotSvc fill:#457b9d,color:#fff
    style FITSSvc fill:#457b9d,color:#fff
    style NotebookSvc fill:#457b9d,color:#fff
    style BrowserSvc fill:#457b9d,color:#fff
    style CalcSvc fill:#457b9d,color:#fff

    style TAPClient fill:#e9c46a,color:#000
    style MASTClient fill:#e9c46a,color:#000
    style ESOClient fill:#e9c46a,color:#000
    style IRSAClient fill:#e9c46a,color:#000
    style ADSClient fill:#e9c46a,color:#000

    style ContextMgr fill:#e63946,color:#fff
    style SessionMem fill:#e63946,color:#fff
    style Memory fill:#e63946,color:#fff
    style LLMClient fill:#e63946,color:#fff
```"""


# ── Output ───────────────────────────────────────────────────────────────────

def write_markdown(filename: str, title: str, description: str, mermaid_content: str):
    """Write a Mermaid diagram to a Markdown file."""
    output_path = OUTPUT_DIR / filename
    
    content = f"""# {title}

> Auto-generated by `scripts/generate_mermaid_graphs.py` — do not edit manually.
> Re-run: `conda run -n quasar python scripts/generate_mermaid_graphs.py`

{description}

{mermaid_content}
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"  ✅ Wrote: {output_path.relative_to(PROJECT_ROOT)}")


def generate_stats(all_files, edges, inheritance_edges, class_methods) -> str:
    """Generate summary statistics."""
    total_files = len(all_files)
    total_classes = sum(len(d["classes"]) for d in all_files.values())
    total_functions = sum(len(d["functions"]) for d in all_files.values())
    total_methods = sum(len(m) for m in class_methods.values())
    total_edges = len(set((s, t) for s, t, _ in edges))
    total_inheritance = len(inheritance_edges)
    
    pkg_stats = defaultdict(lambda: {"files": 0, "classes": 0, "functions": 0})
    for f, d in all_files.items():
        pkg = get_package(f)
        pkg_stats[pkg]["files"] += 1
        pkg_stats[pkg]["classes"] += len(d["classes"])
        pkg_stats[pkg]["functions"] += len(d["functions"])
    
    lines = ["## Codebase Statistics", ""]
    lines.append("| Package | Files | Classes | Functions |")
    lines.append("| --- | --- | --- | --- |")
    for pkg in sorted(pkg_stats.keys()):
        s = pkg_stats[pkg]
        lines.append(f"| **{pkg}** | {s['files']} | {s['classes']} | {s['functions']} |")
    lines.append(f"| **TOTAL** | **{total_files}** | **{total_classes}** | **{total_functions}** |")
    lines.append("")
    lines.append(f"- **{total_methods}** public methods across all classes")
    lines.append(f"- **{total_edges}** module-level import edges")
    lines.append(f"- **{total_inheritance}** class inheritance relationships")
    
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Scanning Quasar codebase...")
    all_files = scan_project()
    print(f"\n📊 Parsed {len(all_files)} Python files")
    
    print("\n🔗 Building dependency graph...")
    edges = build_dependency_edges(all_files)
    print(f"   Found {len(edges)} import edges")
    
    print("\n🏛️  Building class hierarchy...")
    inheritance_edges, class_methods = build_class_edges(all_files)
    print(f"   Found {len(inheritance_edges)} inheritance relationships")
    print(f"   Found {len(class_methods)} classes")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n📝 Generating Mermaid diagrams...\n")
    
    # 1. High-level architecture
    write_markdown(
        "01_high_level_architecture.md",
        "Quasar — High-Level Architecture",
        "Package-level dependency overview showing how many imports flow between each layer.",
        generate_high_level_architecture(all_files, edges),
    )
    
    # 2. Data flow diagram
    write_markdown(
        "02_data_flow.md",
        "Quasar — Request Data Flow",
        "How a user query flows through the system: from the UI through the agent, "
        "tool registry, services, and external integrations.",
        generate_data_flow_graph(all_files),
    )
    
    # 3. Full module dependency graph
    write_markdown(
        "03_module_dependencies.md",
        "Quasar — Full Module Dependency Graph",
        "Every Python file and every import edge between them, grouped by package.",
        generate_module_dependency_graph(all_files, edges),
    )
    
    # 4. Class hierarchy
    write_markdown(
        "04_class_hierarchy.md",
        "Quasar — Class Hierarchy",
        "All classes, their inheritance relationships, and public methods.",
        generate_class_hierarchy_graph(all_files, inheritance_edges, class_methods),
    )
    
    # 5. Per-package internal wiring
    for pkg in PACKAGES:
        pkg_graph = generate_package_graph(pkg, all_files, edges)
        if pkg_graph:
            write_markdown(
                f"05_{pkg}_internals.md",
                f"Quasar — {pkg.title()} Package Internals",
                f"Internal structure and dependencies within the `{pkg}/` package.",
                pkg_graph,
            )
    
    # 6. Statistics summary
    stats = generate_stats(all_files, edges, inheritance_edges, class_methods)
    
    # Write master index
    index_content = f"""# Quasar Architecture Diagrams

> Auto-generated by `scripts/generate_mermaid_graphs.py`
> Re-run: `conda run -n quasar python scripts/generate_mermaid_graphs.py`

## Diagrams

| # | Diagram | Description |
| --- | --- | --- |
| 1 | [High-Level Architecture](01_high_level_architecture.md) | Package-to-package dependency overview |
| 2 | [Request Data Flow](02_data_flow.md) | How a user query flows through the system |
| 3 | [Module Dependencies](03_module_dependencies.md) | Every file and every import edge |
| 4 | [Class Hierarchy](04_class_hierarchy.md) | All classes and inheritance |
| 5 | Package Internals | Per-package detailed wiring: |
|   | — [Core](05_core_internals.md) | Agent, Conductor, RLM, Memory |
|   | — [Services](05_services_internals.md) | RAG, Search, ADS, Plotting |
|   | — [Agents](05_agents_internals.md) | Sub-agent hierarchy |
|   | — [Integrations](05_integrations_internals.md) | TAP, MAST, ESO, ADS clients |
|   | — [Config](05_config_internals.md) | Settings and configuration |
|   | — [Utils](05_utils_internals.md) | Shared utilities |

{stats}
"""
    
    index_path = OUTPUT_DIR / "README.md"
    index_path.write_text(index_content, encoding="utf-8")
    print(f"  ✅ Wrote: {index_path.relative_to(PROJECT_ROOT)}")
    
    print(f"\n🎉 Done! {len(list(OUTPUT_DIR.glob('*.md')))} files written to docs/architecture/")
    print(f"   View them in any Markdown renderer that supports Mermaid (GitHub, VS Code, etc.)")


if __name__ == "__main__":
    main()
