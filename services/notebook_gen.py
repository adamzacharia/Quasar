import json
import uuid

class NotebookGenerator:
    """Service to generate Jupyter Notebooks (.ipynb) dynamically."""
    
    def __init__(self):
        self.notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def _create_cell(self, cell_type: str, source: str) -> dict:
        """Helper to create a notebook cell."""
        # Ensure source ends with newline if required by format, or just split into lists
        source_lines = [line + '\n' for line in source.split('\n')]
        # remove the last newline from the last line
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip('\n')
            
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    def add_markdown_cell(self, text: str):
        """Add a markdown cell."""
        self.notebook["cells"].append(self._create_cell("markdown", text))

    def add_code_cell(self, code: str):
        """Add a Python code cell."""
        self.notebook["cells"].append(self._create_cell("code", code))

    def generate_json(self) -> str:
        """Return the notebook as a JSON string."""
        return json.dumps(self.notebook, indent=2)

    def generate_dict(self) -> dict:
        """Return the notebook as a dictionary."""
        return self.notebook

def generate_analysis_notebook(title: str, steps: list) -> dict:
    """
    Generate an analysis notebook from a list of steps.
    Each step should be a dict: {'type': 'markdown' | 'code', 'content': '...'}
    """
    nb = NotebookGenerator()
    nb.add_markdown_cell(f"# {title}\nAuto-generated QUASAR Analysis Notebook.")
    nb.add_code_cell("import numpy as np\nimport matplotlib.pyplot as plt\nfrom astropy.io import fits\nfrom spectral_cube import SpectralCube\nimport astropy.units as u")
    
    for step in steps:
        if step.get("type") == "markdown":
            nb.add_markdown_cell(step.get("content", ""))
        elif step.get("type") == "code":
            nb.add_code_cell(step.get("content", ""))
            
    return nb.generate_dict()
