"""
Quasar Documentation Page
"""

import streamlit as st

st.set_page_config(page_title="Quasar Documentation", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0a1628 !important; }
    h1, h2, h3 { color: #f472b6 !important; }
    p, li { color: #e2e8f0 !important; font-size: 1.05rem; line-height: 1.6; }
    strong { color: #f472b6 !important; }
    code { background: #1a3a5c !important; color: #f472b6 !important; padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); }
    .doc-link { color: #f472b6 !important; font-size: 1.2rem; font-weight: bold; text-decoration: none; border-bottom: 2px solid #f472b6; padding-bottom: 2px; }
    .doc-link:hover { opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

st.title("Quasar Documentation")

st.markdown("""
<div style='margin-bottom: 2rem;'>
    <a href='https://github.com/adamzacharia/Quasar2' target='_blank' class='doc-link'>View Source on GitHub ↗</a>
</div>
""", unsafe_allow_html=True)

st.header("Introduction")
st.markdown("""
Quasar is an advanced AI assistant designed specifically for radio astronomy research. It integrates with the ALMA Science Archive and NASA ADS to provide data retrieval, visualization, and literature search capabilities through a chat interface.
""")

st.divider()

st.header("Core Commands")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📡 ALMA Archive Search")
    st.markdown("""
    Use `@archive` to search for observational data.
    
    **Examples:**
    *   `@archive Find Band 6 data for Sz65`
    *   `@archive Show me observations of Orion Nebula from 2023`
    *   `@archive What gives the best angular resolution for HL Tau?`
    """)

with col2:
    st.markdown("### 📄 Literature Search")
    st.markdown("""
    Use `@paper` to find relevant research papers via NASA ADS.
    
    **Examples:**
    *   `@paper Recent papers on protoplanetary disks`
    *   `@paper Find papers by author 'Smith' on pulsars`
    *   `@paper Key papers on ALMA calibration`
    """)

st.markdown("### 🔍 Knowledge Base")
st.markdown("""
Use `@search` to query your personal documents and general knowledge.

*   `@search How do I reduce ALMA data?`
*   `@search Summarize the uploaded PDF about star formation`
""")

st.divider()

st.header("🔧 Adding Custom Tools")

st.markdown("""
Quasar allows you to extend its functionality by determining your own custom tools using Python. This feature is powerful for integrating specific calculations or external APIs.

### How it Works
1.  **Define**: You write a Python function in the "Add Tools" sidebar section.
2.  **Save**: Quasar saves this code to your user profile (`user_tools/`).
3.  **Load**: On restart, the AI agent loads your function and understands its capabilities.
4.  **Use**: You can ask the agent to perform tasks that use your tool.

### Step-by-Step Guide
1.  Expand the **Add Tools** section in the sidebar.
2.  Enter a **Tool Name** (e.g., `calculate_redshift`).
3.  Enter a **Description**. This is crucial—it tells the AI *when* to use your tool.
4.  Write the **Python Code**. The function must return a dictionary (`dict`) to be interpreted by the agent.

### Example: Flux Calculator

**Tool Name:** `calculate_flux`

**Description:**
> Calculates flux density given frequency and brightness temperature.

**Python Code:**
```python
def calculate_flux(frequency_ghz: float, temp_kelvin: float, beam_size_arcsec: float) -> dict:
    \"\"\"
    Calculate flux density in Jansky.
    Args:
        frequency_ghz: Frequency in GHz
        temp_kelvin: Brightness temperature in K
        beam_size_arcsec: Beam size in arcseconds
    \"\"\"
    # Constants
    k_b = 1.38e-23
    c = 3e8
    
    # Conversion logic (simplified example)
    freq_hz = frequency_ghz * 1e9
    wavelength = c / freq_hz
    beam_sr = (beam_size_arcsec / 206265) ** 2
    
    flux_density = (2 * k_b * temp_kelvin * freq_hz**2 / c**2) * beam_sr * 1e26
    
    return {
        "flux_mjy": flux_density * 1000,
        "frequency": f"{frequency_ghz} GHz",
        "temperature": f"{temp_kelvin} K"
    }
```

### Tips for Success
*   **Type Hints:** Always use type hints (e.g., `: float`, `: str`) so the AI knows what arguments to pass.
*   **Docstrings:** Add a clear docstring explaining parameters.
*   **Return Dict:** Always return a dictionary with the results.
*   **Restart:** You must restart the app (or start a new session) after adding a tool for it to be active.
""")

st.divider()

st.header("📚 Personal Knowledge Base")
st.markdown("""
You can make Quasar an expert on your specific research by uploading documents.

1.  Top open the **My Documents** section in the sidebar.
2.  Select **Browse files** and choose your PDF or TXT files (e.g., draft papers, lab notes).
3.  Click **Upload**.
4.  Wait for the confirmation.

Once uploaded, you can ask questions like:
*   *"What does my draft paper say about the error analysis?"*
*   *"Compare the results in the uploaded PDF with known literature."*
""")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Quasar v2.0 - Built for the Radio Astronomy Community")
