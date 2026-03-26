import os
import json
import ast
import datetime
import glob

def get_relative_path(path, root):
    return os.path.relpath(path, root).replace("\\", "/")

def build_graph(root_dir):
    nodes = []
    edges = []
    
    # 1. SCAN
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'venv', '.venv', 'node_modules', '.understand-anything', 'Understand-Anything', 'chroma_db', 'docs', 'data', 'tests']]
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(os.path.join(dirpath, f))

    # 2. ANALYZE
    for file_path in py_files:
        rel_path = get_relative_path(file_path, root_dir)
        file_node_id = f"file:{rel_path}"
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            tree = None

        size_lines = len(content.splitlines()) if tree else 0
        
        nodes.append({
            "id": file_node_id,
            "type": "file",
            "name": os.path.basename(rel_path),
            "filePath": rel_path,
            "summary": f"Python source file ({size_lines} lines).",
            "tags": ["python"]
        })
        
        if not tree:
            continue
            
        # Parse tree
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append({
                        "source": file_node_id,
                        "target": f"module:{alias.name.split('.')[0]}",
                        "type": "imports",
                        "weight": 0.7
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    edges.append({
                        "source": file_node_id,
                        "target": f"module:{node.module.split('.')[0]}",
                        "type": "imports",
                        "weight": 0.7
                    })
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_id = f"func:{rel_path}:{node.name}"
                nodes.append({
                    "id": func_id,
                    "type": "function",
                    "name": node.name,
                    "filePath": rel_path,
                    "summary": ast.get_docstring(node) or f"Function {node.name}.",
                    "tags": ["function"]
                })
                edges.append({
                    "source": file_node_id,
                    "target": func_id,
                    "type": "contains",
                    "weight": 1.0
                })
            elif isinstance(node, ast.ClassDef):
                class_id = f"class:{rel_path}:{node.name}"
                nodes.append({
                    "id": class_id,
                    "type": "class",
                    "name": node.name,
                    "filePath": rel_path,
                    "summary": ast.get_docstring(node) or f"Class {node.name}.",
                    "tags": ["class"]
                })
                edges.append({
                    "source": file_node_id,
                    "target": class_id,
                    "type": "contains",
                    "weight": 1.0
                })
                # Methods
                for subnode in ast.iter_child_nodes(node):
                    if isinstance(subnode, ast.FunctionDef) or isinstance(subnode, ast.AsyncFunctionDef):
                        method_id = f"func:{rel_path}:{node.name}.{subnode.name}"
                        nodes.append({
                            "id": method_id,
                            "type": "function",
                            "name": subnode.name,
                            "filePath": rel_path,
                            "summary": ast.get_docstring(subnode) or f"Method {subnode.name}.",
                            "tags": ["method"]
                        })
                        edges.append({
                            "source": class_id,
                            "target": method_id,
                            "type": "contains",
                            "weight": 1.0
                        })

    # Ensure all targets in edges exist in nodes, if not, create mock module nodes
    existing_node_ids = {n['id'] for n in nodes}
    for e in edges:
        if e['target'] not in existing_node_ids and e['target'].startswith('module:'):
            nodes.append({
                "id": e['target'],
                "type": "module",
                "name": e['target'].split(':')[1],
                "filePath": "",
                "summary": "External or untracked module.",
                "tags": ["module"]
            })
            existing_node_ids.add(e['target'])
            
    # Cleanup dangling edges
    valid_edges = [e for e in edges if e['source'] in existing_node_ids and e['target'] in existing_node_ids]
    
    # 4. ARCHITECTURE (Basic heuristics)
    layers = [
        {
            "id": "layer:services",
            "name": "Services",
            "description": "Core business logic and external integrations",
            "nodeIds": [n['id'] for n in nodes if n['type'] == 'file' and 'services/' in n['filePath']]
        },
        {
            "id": "layer:core",
            "name": "Core",
            "description": "Fundamental models, configs, and base classes",
            "nodeIds": [n['id'] for n in nodes if n['type'] == 'file' and 'core/' in n['filePath']]
        },
        {
            "id": "layer:scripts",
            "name": "Scripts",
            "description": "CLI scripts and runnables",
            "nodeIds": [n['id'] for n in nodes if n['type'] == 'file' and 'scripts/' in n['filePath']]
        },
        {
            "id": "layer:utils",
            "name": "Utils",
            "description": "Shared utility functions",
            "nodeIds": [n['id'] for n in nodes if n['type'] == 'file' and 'utils/' in n['filePath']]
        }
    ]
    layers = [l for l in layers if len(l['nodeIds']) > 0]
    
    # 5. TOUR (Basic intro)
    tour = [
        {
            "order": 1,
            "title": "Welcome to Quasar",
            "description": "This is the main entry point for the Quasar Agent framework.",
            "nodeIds": ["file:quasar.py"] if "file:quasar.py" in existing_node_ids else []
        }
    ]

    graph = {
        "version": "1.0.0",
        "project": {
            "name": "Quasar",
            "languages": ["Python"],
            "frameworks": ["FastAPI", "Pydantic"],
            "description": "Quasar AI framework and knowledge processing system.",
            "analyzedAt": datetime.datetime.now().isoformat(),
            "gitCommitHash": "unknown"
        },
        "nodes": nodes,
        "edges": valid_edges,
        "layers": layers,
        "tour": tour
    }

    out_dir = os.path.join(root_dir, '.understand-anything')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'knowledge-graph.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)
        
    print(f"Graph generated at {out_file} with {len(nodes)} nodes and {len(valid_edges)} edges.")

if __name__ == '__main__':
    build_graph(r'c:\Users\adama\Desktop\Quasar-main')
