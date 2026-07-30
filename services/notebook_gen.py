import json
import uuid
from typing import Any

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


def datalab_notebook_steps(
    *,
    sql=None,
    catalog=None,
    table=None,
    sia=None,
    svo_filters=None,
    citation=None,
):
    """Build reproducible NOIRLab Astro Data Lab notebook steps.

    Returns a list of {'type','content'} step dicts (the format generate_analysis_notebook
    consumes) covering a TAP query (native SQL via queryClient), an optional SIA cutout,
    an optional SVO filter-wavelength lookup, and a data-citation cell.

    sia: optional dict {'ra','dec','fov_deg','endpoint'}.
    svo_filters: optional list of SVO filterIDs (e.g. ['CTIO/DECam.g','WISE/WISE.W1']).
    citation: optional dict {'text','url','doi'} (e.g. datalab_registry.citation(catalog)).
    """
    steps = [{
        "type": "markdown",
        "content": "## NOIRLab Astro Data Lab — reproducible recipe\n"
                   "Run in an environment with the `astro-datalab` client (or use the REST API).",
    }, {
        "type": "code",
        "content": (
            "from dl import queryClient as qc\n"
            "from pyvo.dal import sia\n"
            "from astropy.io import fits\n"
            "from astropy.utils.data import download_file\n"
            "import numpy as np"
        ),
    }]
    if sql:
        steps.append({"type": "markdown", "content": f"### TAP query — {catalog or ''}.{table or ''}".rstrip(".")})
        steps.append({"type": "code", "content": "q = '''" + str(sql).replace("'''", "\\'\\'\\'") + "'''\n"
                                                 "df = qc.query(sql=q, fmt='pandas')\ndf.head()"})
    if isinstance(sia, dict) and sia.get("ra") is not None and sia.get("dec") is not None:
        endpoint = sia.get("endpoint") or "https://datalab.noirlab.edu/sia/coadd_all"
        fov = sia.get("fov_deg", 0.1)
        steps.append({"type": "markdown", "content": "### SIA image cutout (dec-corrected size; deepest Stack)"})
        steps.append({"type": "code", "content": (
            f"svc = sia.SIAService('{endpoint}')\n"
            f"ra, dec, fov = {float(sia['ra'])}, {float(sia['dec'])}, {float(fov)}\n"
            "imgs = svc.search((ra, dec), (fov/np.cos(np.radians(dec)), fov), verbosity=2).to_table()\n"
            "sel = (imgs['proctype']=='Stack') & (imgs['prodtype']=='image')\n"
            "row = imgs[sel][np.argmax(imgs[sel]['exptime'])]  # deepest stack\n"
            "img = fits.getdata(download_file(row['access_url'], cache=True, timeout=180))"
        )})
    if svo_filters:
        steps.append({"type": "markdown", "content": "### Filter effective wavelengths (SVO FPS — not hardcoded)"})
        steps.append({"type": "code", "content": (
            "from astroquery.svo_fps import SvoFps\n"
            f"filter_ids = {list(svo_filters)}\n"
            "# WavelengthEff / WavelengthPivot (Angstrom) come from the SVO service per filterID.\n"
            "waves = {fid: SvoFps.get_filter_list(filterID=fid) for fid in filter_ids}\n"
            "waves"
        )})
    if citation:
        if isinstance(citation, dict):
            text = citation.get("text") or ""
            url = citation.get("url") or citation.get("doi")
        else:
            text, url = str(citation), None
        body = f"### Data citation\n\n{text}" + (f"\n\n{url}" if url else "")
        steps.append({"type": "markdown", "content": body})
    return steps


def generate_conductor_notebook(
    query: str,
    subtasks: list,
    results: dict,
    dag_summary: dict = None,
) -> dict:
    """
    Generate a research-grade Jupyter notebook from a Conductor DAG execution.

    Builds a comprehensive, reproducible notebook containing:
    - The user's original query
    - Requirements / install cell
    - For each subtask: a markdown heading describing the step,
      plus a Python code cell showing how to reproduce it.
    - A summary section with the final synthesis.

    Parameters
    ----------
    query : str
        The user's original question.
    subtasks : list[dict]
        The DAG subtask definitions (id, description, agent_type, depends_on).
    results : dict
        Mapping of task_id → result from the Conductor execution.
    dag_summary : dict, optional
        Output of dag.get_execution_summary() for stats.

    Returns
    -------
    dict
        A Jupyter notebook dict (ready for JSON serialisation).
    """
    nb = NotebookGenerator()

    # ── Title cell ──
    nb.add_markdown_cell(
        f"# 🔭 Quasar Research Notebook\n\n"
        f"**Query:** {query}\n\n"
        f"*Auto-generated by the Quasar Multi-Agent Workforce.*\n\n"
        f"This notebook contains the code to reproduce each step of "
        f"the analysis that was performed."
    )

    # ── Requirements cell ──
    nb.add_code_cell(
        "# Install required packages (uncomment if needed)\n"
        "# !pip install astroquery pyvo astropy matplotlib spectral-cube numpy requests"
    )

    # ── Standard imports ──
    nb.add_code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from astropy.io import fits\n"
        "from astropy.coordinates import SkyCoord\n"
        "import astropy.units as u\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Optional: ALMA archive access\n"
        "try:\n"
        "    from astroquery.alma import Alma\n"
        "    from astroquery.simbad import Simbad\n"
        "except ImportError:\n"
        "    print('Install astroquery: pip install astroquery')\n"
        "\n"
        "# Optional: visualization\n"
        "try:\n"
        "    from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch\n"
        "except ImportError:\n"
        "    pass"
    )

    # ── Instructions ──
    nb.add_markdown_cell(
        "## 📋 Instructions\n\n"
        "1. Start by importing the required Astropy and Astroquery libraries.\n"
        "2. Follow the steps sequentially to reproduce the research logic.\n"
        "3. Replace generic tokens like `TARGET_NAME` or `QUERY` with your specific dataset parameters."
    )

    # ── One section per subtask ──
    for index, st in enumerate(subtasks):
        desc = st.get("description", "Unnamed step")
        agent = st.get("agent_type", "general")
        result = results.get(st.get("id"))

        nb.add_markdown_cell(
            f"### Step {index + 1}: {desc}\n\n"
            f"The following code demonstrates how to perform this analytical step programmatically."
        )

        # Build a code cell template for reproducibility
        code = _build_code_cell(agent, desc, result)
        if code:
            nb.add_code_cell(code)

    nb.add_markdown_cell(
        "---\n\n"
        "*Notebook auto-generated by [Quasar](https://github.com/your-org/quasar). "
        "Feel free to modify and extend.*"
    )

    return nb.generate_dict()


def _build_code_cell(agent_type: str, description: str, result) -> str:
    """
    Build a runnable Python code cell template for a subtask based on its
    agent type and the actual result data.

    (#10) Uses real values from the result to make notebooks executable
    out of the box instead of requiring manual editing.
    """
    desc_lower = description.lower()

    # ── Extract real values from result data ───────────────────────────
    target = _extract_target(description, result)
    band = _extract_band(description, result)

    if agent_type == "archive" or "alma" in desc_lower or "search" in desc_lower:
        code = f"# Step: {description}\n"
        code += "from astroquery.alma import Alma\n"
        code += "import pandas as pd\n\n"
        if target and target != "TARGET_NAME":
            code += f"# Search ALMA archive for {target}\n"
            code += f"rs = Alma.query_object('{target}'"
            if band:
                code += f", band_list=[{band}]"
            code += ")\n"
            code += "df = rs.to_pandas()\n"
            code += f"print(f'Found {{len(df)}} ALMA observations of {target}')\n"
            code += "df[['target_name', 'band_list', 's_resolution', 't_exptime']].head(10)"
        else:
            code += "# rs = Alma.query_object('TARGET_NAME')\n"
            code += "# df = rs.to_pandas()\n"
            code += "# df.head()"
        return code

    if "cadc" in desc_lower or "jwst" in desc_lower or "hst" in desc_lower:
        code = f"# Step: {description}\n"
        code += "import pyvo\n\n"
        code += "# Query CADC TAP service\n"
        code += "tap = pyvo.dal.TAPService('https://ws.cadc-cccs.hia-iha.nrc-cnrc.gc.ca/argus')\n"
        if target and target != "TARGET_NAME":
            code += f"query = \"SELECT TOP 20 * FROM caom2.Observation WHERE target_name='{target}'\"\n"
        else:
            code += "query = \"SELECT TOP 20 * FROM caom2.Observation WHERE target_name='TARGET_NAME'\"\n"
        code += "results = tap.search(query)\n"
        code += "results.to_table().to_pandas().head()"
        return code

    if agent_type == "literature" or "paper" in desc_lower or "ads" in desc_lower:
        code = f"# Step: {description}\n"
        code += "import requests\nimport os\n\n"
        code += "ADS_TOKEN = os.getenv('ADS_DEV_KEY', 'YOUR_ADS_TOKEN')\n"
        search_term = target if target and target != "TARGET_NAME" else "QUERY"
        code += f"query = '{search_term}'\n"
        code += "url = f'https://api.adsabs.harvard.edu/v1/search/query?q={query}&fl=title,author,bibcode,year&rows=10'\n"
        code += "headers = {'Authorization': f'Bearer {ADS_TOKEN}'}\n"
        code += "resp = requests.get(url, headers=headers)\n"
        code += "papers = resp.json().get('response', {}).get('docs', [])\n"
        code += "for p in papers:\n"
        code += "    print(f\"{p.get('year')} - {p.get('title', ['?'])[0][:80]}\")"
        return code

    if agent_type == "viz" or "overlay" in desc_lower or "plot" in desc_lower or "render" in desc_lower:
        code = f"# Step: {description}\n"
        code += "import matplotlib.pyplot as plt\n"
        code += "from astropy.io import fits\n"
        code += "from astropy.visualization import ZScaleInterval, ImageNormalize\n"
        code += "# Default (light) style: readable on paper, in docs, and in print\n"
        code += "plt.style.use('default')\n"
        code += "plt.rcParams.update({'axes.titlesize': 13, 'axes.labelsize': 12})\n\n"
        code += "# Load the FITS file\n"
        code += "# hdul = fits.open('FITS_URL_OR_PATH')\n"
        code += "# data = hdul[0].data\n\n"
        code += "# Plot with ZScale normalization\n"
        code += "# norm = ImageNormalize(data, interval=ZScaleInterval())\n"
        code += "# fig, ax = plt.subplots(figsize=(10, 10))\n"
        code += "# ax.imshow(data, norm=norm, cmap='inferno', origin='lower')\n"
        code += "# ax.set_title('" + (target or "Observation") + "')\n"
        code += "# plt.colorbar(ax.images[0])\n"
        code += "# plt.show()"
        return code

    if agent_type == "analysis" or "splatalogue" in desc_lower or "line" in desc_lower or "co" in desc_lower:
        code = f"# Step: {description}\n"
        code += "from astroquery.splatalogue import Splatalogue\n"
        code += "import astropy.units as u\n\n"
        # Try to extract frequency range
        code += "# Query Splatalogue for molecular lines\n"
        code += "lines = Splatalogue.query_lines(\n"
        code += "    200 * u.GHz,\n"
        code += "    300 * u.GHz,\n"
        code += "    chemical_name=' CO ',\n"
        code += "    energy_max=500,\n"
        code += "    energy_type='eu_k'\n"
        code += ")\n"
        code += "df = lines.to_pandas()\n"
        code += "print(f'Found {len(df)} spectral lines')\n"
        code += "df[['Species', 'Chemical Name', 'Freq-GHz(rest frame,redshifted)']].head(20)"
        return code

    return (
        f"# Perform step: {description}\n"
        "# Configure your analysis below.\n"
        "pass"
    )


def _extract_target(description: str, result: Any) -> str:
    """Extract a target name from description or result data."""
    import re

    # Try result data first
    if isinstance(result, dict):
        for key in ("target_name", "target", "object_name"):
            if key in result:
                return str(result[key])

    # Try description
    patterns = [
        r'\b(M\d{1,3})\b',
        r'\b(NGC\s*\d{1,5})\b',
        r'\b(IC\s*\d{1,5})\b',
        r'\b(Sgr\s*[AB]\*?)\b',
        r'\b(3C\s*\d{1,3})\b',
        r'\b(Sz\s*\d{1,3})\b',
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            return m.group(1)

    return "TARGET_NAME"


def _extract_band(description: str, result: Any) -> str:
    """Extract an ALMA band number from description or result data."""
    import re
    m = re.search(r'[Bb]and\s*(\d+)', description)
    if m:
        return m.group(1)
    if isinstance(result, dict) and "band" in result:
        return str(result["band"])
    return ""
