"""
Script to analyze ALMA science archive notebooks and extract key capabilities
"""
import nbformat
import os

notebooks_dir = 'alma-science-archive-notebooks'

for filename in sorted(os.listdir(notebooks_dir)):
    if filename.endswith('.ipynb'):
        print(f'\n{"="*60}')
        print(f'NOTEBOOK: {filename}')
        print(f'{"="*60}')
        try:
            with open(os.path.join(notebooks_dir, filename), 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            cell_num = 0
            for cell in nb.cells:
                cell_num += 1
                if cell.cell_type == 'markdown':
                    md = cell.source
                    # Only print headers and key descriptions
                    lines = md.split('\n')
                    for line in lines:
                        if line.startswith('#') or 'query' in line.lower() or 'search' in line.lower():
                            print(f'[MD]: {line[:150]}')
                elif cell.cell_type == 'code':
                    code = cell.source
                    if len(code) > 20:
                        # Print first 400 chars of code
                        print(f'[CODE Cell {cell_num}]:')
                        print(code[:600])
                        if len(code) > 600:
                            print('... (truncated)')
                        print()
        except Exception as e:
            print(f'Error reading {filename}: {e}')
