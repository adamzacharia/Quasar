import json
from services.notebook_gen import generate_analysis_notebook

steps = [
    {
        "type": "markdown",
        "content": "This is a test of the notebook generator."
    },
    {
        "type": "code",
        "content": "x = np.linspace(0, 10, 100)\nplt.plot(x, np.sin(x))\nplt.show()"
    }
]

nb_dict = generate_analysis_notebook("Test Notebook", steps)

with open("test_output.ipynb", "w") as f:
    json.dump(nb_dict, f, indent=2)

print("Notebook generated successfully. Validating JSON structure...")
# Minimal validation: check if cell types are present
assert "cells" in nb_dict
assert len(nb_dict["cells"]) == 4 # 2 auto-gen + 2 steps
assert nb_dict["cells"][0]["cell_type"] == "markdown"
assert nb_dict["cells"][1]["cell_type"] == "code"
assert nb_dict["cells"][2]["cell_type"] == "markdown"
assert nb_dict["cells"][3]["cell_type"] == "code"

print("All tests passed!")
