# ALMA MCP Server - Astronomical Data Access via Natural Language

A Model Context Protocol (MCP) server that provides comprehensive access to the ALMA (Atacama Large Millimeter/submillimeter Array) archive through a clean, extensible architecture.

## Vision

This MCP server transforms ALMA archive queries from a software engineering problem into a **natural language conversation**. Instead of learning TAP/ADQL syntax and archive APIs, researchers simply ask for what they need and get clean, analysis-ready results.


**The result**: AI assistants that can seamlessly search ALMA data by target, position, frequency, resolution, or any custom criteria 
---

## Quick Setup for Claude Desktop

### 1. Clone/Copy and Setup Environment

> **IMPORTANT: Change the paths below according to YOUR installation location!**

```bash
# Clone the repository (or copy the ALMA_MCP folder)
git clone https://github.com/adamzacharia/Quasar2.git
cd Quasar2/ALMA_MCP

# Create a dedicated conda environment with Python 3.11+
conda create -n alma_mcp python=3.11
conda activate alma_mcp

# Install dependencies
pip install -r requirements.txt

# Install astronomical libraries for full functionality
pip install fastmcp alminer pyvo astroquery astropy pandas
```

**Alternative: Using venv instead of conda:**
```bash
# Navigate to the ALMA_MCP folder
cd c:\Users\Asus\Desktop\Quasar-main\ALMA_MCP

# Create a dedicated venv environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Server

```bash
# Test basic functionality
python test_server.py

# Quick test (optional)
python -c "
from server import get_alma_info, ALMINER_AVAILABLE, PYVO_AVAILABLE
print(' Server loads successfully')
print(f' alminer available: {ALMINER_AVAILABLE}')
print(f' pyvo available: {PYVO_AVAILABLE}')
"
```

### 3. Configure Claude Desktop

Find and edit the Claude Desktop MCP configuration file:

1. Navigate to your Claude Desktop AppData folder:
   - **Windows**: Open File Explorer and go to `%APPDATA%\Claude\`
   - **macOS**: `~/Library/Application Support/Claude/`

2. Find the file named `config.json` 

3. Add the following configuration to the file:

> **IMPORTANT: Change the path below according to YOUR installation location!**

```json
{
  "mcpServers": {
    "alma": {
      "command": "python",
      "args": ["c:/Users/Asus/Desktop/Quasar-main/ALMA_MCP/server.py"]
    }
  }
}
```

> **Note**: Use forward slashes `/` in paths even on Windows.

#### Using a Custom Environment (Conda or venv)

If you installed the dependencies in a **conda environment** or **venv**, you MUST specify the full path to that environment's Python executable. Otherwise the system won't find the installed packages!

**For Conda environment:**
```json
{
  "mcpServers": {
    "alma": {
      "command": "C:/Users/YourName/anaconda3/envs/alma_mcp/python.exe",
      "args": ["c:/path/to/ALMA_MCP/server.py"],
      "cwd": "c:/path/to/ALMA_MCP",
      "env": {}
    }
  }
}
```

To find your conda environment's Python path, run:
```bash
conda activate alma_mcp
where python    # Windows
which python    # Mac/Linux
```

**For venv:**
```json
{
  "mcpServers": {
    "alma": {
      "command": "c:/path/to/ALMA_MCP/venv/Scripts/python.exe",
      "args": ["c:/path/to/ALMA_MCP/server.py"],
      "cwd": "c:/path/to/ALMA_MCP",
      "env": {}
    }
  }
}
```


### 4. Restart and Test

1. **Quit Claude Desktop completely** (right-click system tray → Quit)
2. **Check Task Manager** - If Claude is still running in the background, end the task
3. **Reopen Claude Desktop**
4. **Test with a query** like:
   - "Search ALMA for observations of M87"
   - "Find high-resolution ALMA data with resolution under 0.5 arcseconds"
   - "What are ALMA's frequency bands?"

### 5. Troubleshooting

**Server won't start:**
```bash
# Check Python environment
python --version  # Should be 3.10+

# Test server manually
python server.py
# Should start without errors
```

**MCP connection issues:**
- Verify the Python path in your config is correct
- Ensure all dependencies are installed
- Check that `server.py` exists at the specified path

**Missing dependencies:**
```bash
pip install fastmcp alminer pyvo astroquery astropy pandas
```

---

## Usage Examples

Once configured, you can ask natural language questions about ALMA data:

### Basic Searches
- "Find observations of Orion KL"
- "Search ALMA for M87 within 1 arcminute"
- "What ALMA data exists for NGC 1234?"

### Position-Based Searches  
- "Search ALMA at RA=83.6, Dec=-5.4"
- "Find observations near the Galactic Center"
- "Cone search at coordinates 150.5, 2.2 with 5 arcmin radius"

### Frequency Searches
- "Find ALMA data between 230 and 250 GHz"
- "Search for Band 6 observations"
- "What observations cover the CO(2-1) line at 230.5 GHz?"

### Resolution Searches
- "Find high-resolution data with resolution under 0.1 arcseconds"
- "Search for ALMA observations better than 0.5 arcsec resolution"

### Proposal/PI Searches
- "Find ALMA proposals by Adele Plunkett"
- "Get data for proposal 2023.1.00001.S"
- "Search for galaxy evolution proposals"

### Spectral Line Coverage
- "Do any M87 observations cover the CO(2-1) line?"
- "Check if Orion data covers HCN(1-0) at 88.6 GHz"
- "Find observations that cover 115 GHz for a source at z=0.5"

### Custom SQL Queries
- "Run this SQL: SELECT target_name, band_list FROM ivoa.obscore WHERE target_name LIKE '%M87%'"
- "Query ALMA for all Band 7 observations from 2023"

### ALMA Information
- "What are ALMA's frequency bands?"
- "List common spectral lines in the mm range"
- "What science categories does ALMA support?"

---

### Bibliography & Publication Searches
- "Find ALMA observations published in Nature"
- "Search for data used in papers by Smith published in 2023"
- "What ALMA data has the bibcode 2017ApJ...834..140R"

### Science Keyword Searches
- "Find all ALMA observations tagged with 'Quasars'"
- "Search for Sub-mm Galaxies (SMG) observations"
- "What observations are categorized under 'Active Galactic Nuclei'?"

### Data Type Searches
- "Find all spectral cubes for M87"
- "Search for continuum images tagged with Exoplanets"
- "Show me spectral line observations in Band 6"

### Abstract Searches
- "Find proposals mentioning black holes in their abstract"
- "Search for observations from proposals about star formation"
- "What proposals mention the cosmic microwave background?"

### Sensitivity Searches
- "Find observations with continuum sensitivity better than 0.1 mJy/beam"
- "Search for deep ALMA observations under 0.05 mJy sensitivity"

### Batch Multi-Source Queries
- "Check if ALMA has observed M31, M51, M82, and NGC 1068"
- "Query ALMA for these sources: Cen A, M83, and NGC 1234"

---

## Architecture

```
ALMA_MCP/
├── server.py           # Main MCP server with 16 tools
├── requirements.txt    # Python dependencies
├── test_server.py      # Test suite
└── README.md           # This file
```

---

## Features

### ALMA Archive Access
- **Target Search**: Resolve object names via SIMBAD and search ALMA
- **Position Search**: Cone search by RA/Dec coordinates
- **Proposal Search**: Find by PI name, proposal ID, or science category
- **Frequency Search**: Query by frequency range (GHz)
- **Resolution Search**: Filter by angular resolution (arcsec)
- **Custom SQL**: Run any ADQL query against ALMA TAP

### Core Query Tools (8 original)
| Tool | Description |
|------|-------------|
| `search_alma_by_target` | Search by astronomical object name (SIMBAD resolution) |
| `search_alma_by_position` | Cone search by RA/Dec coordinates |
| `search_alma_by_proposal` | Search by PI name or proposal ID |
| `search_alma_by_frequency` | Search by frequency range (GHz) |
| `search_alma_by_resolution` | Search by angular resolution (arcsec) |
| `check_alma_line_coverage` | Check spectral line coverage with redshift |
| `get_alma_info` | ALMA bands, lines, and capabilities reference |
| `run_alma_tap_query` | Custom SQL/ADQL queries against TAP |

### Extended Query Tools (8 new - from ALMA notebooks)
| Tool | Description |
|------|-------------|
| `search_alma_by_source_name` | Search by PI-specified target name (exact or partial) |
| `search_alma_by_bibliography` | Search by bibcode, journal, author, or publication year |
| `search_alma_by_member_ous` | Search by Member OUS dataset identifier |
| `search_alma_by_data_type` | Search for cubes (spectral) vs images (continuum) |
| `search_alma_by_science_keyword` | Search by ALMA science keywords |
| `search_alma_by_abstract` | Full-text search in proposal/publication abstracts |
| `search_alma_by_sensitivity` | Search by continuum or line sensitivity (mJy/beam) |
| `query_alma_multiple_sources` | Batch query for multiple sources at once |

### Multi-Backend Support
- **alminer**: Advanced ALMA queries with spectral line tools
- **pyvo**: Direct TAP/ADQL access to ALMA archive
- **astroquery**: SIMBAD name resolution


---

## Dependencies

### Core Requirements
```
fastmcp>=2.0.0      # MCP framework
pandas>=1.5.0       # Data manipulation
```

### Astronomical Libraries
```
alminer>=0.2.0      # Advanced ALMA queries
pyvo>=1.4.0         # TAP/ADQL access
astroquery>=0.4.0   # SIMBAD, VizieR, etc.
astropy>=5.0.0      # Astronomical utilities
```

---

## Roadmap

### Current (v1.0) ✅
- [x] ALMA target name search with SIMBAD resolution
- [x] Position-based cone search
- [x] Frequency range search
- [x] Angular resolution filtering
- [x] Proposal/PI search
- [x] Spectral line coverage check
- [x] Custom SQL/TAP queries
- [x] ALMA info and band reference
- [x] Source name search (PI-specified names, exact/partial)
- [x] Bibliography/publication search (bibcode, journal, author, year)
- [x] Member OUS ID search (dataset identifier)
- [x] Data type filtering (cubes vs images)
- [x] Science keyword search with filters
- [x] Abstract full-text search (proposal and publication)
- [x] Sensitivity-based search (continuum or line)
- [x] Batch multi-source queries

### Planned (v2.0)
- [ ] VLA archive integration
- [ ] GBT archive integration  
- [ ] Result caching for faster queries
- [ ] FITS file download support
- [ ] Cross-archive object matching (ALMA + VLA + optical)
- [ ] Visualization tools (sky plots, spectra)
- [ ] Data download workflow

---

## Moving to Standalone Location

This folder is designed to be portable. To move it:

1. Copy the entire `ALMA_MCP/` folder to your desired location
2. Update the path in `config.json`
3. Restart Claude Desktop

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/vla-support`)
3. Add your data source following existing patterns
4. Write tests for new functionality
5. Submit a pull request

## Authors

- **Adam Zacharia Anil** - Lead Developer
- **Adele Plunkett** - Scientific Advisor

---

## Acknowledgments

Special thanks to:

- **Adele Plunkett** - For valuable guidance and advice throughout this project
- **NRAO (National Radio Astronomy Observatory)** - For supporting astronomical research and data access
- **Brian Mason** - For technical expertise and support
- **Cosmic AI / Stella Offner** - For inspiration in applying AI to astronomical research

---

## License

Part of the [Quasar Project](https://github.com/adamzacharia/Quasar2)

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{alma_mcp,
  title={ALMA MCP Server: Astronomical Data Access for AI Agents},
  author={Adam Zacharia Anil and Adele Plunkett},
  year={2025},
  url={https://github.com/adamzacharia/Quasar2}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/adamzacharia/Quasar2/issues)
- **Documentation**: This README and inline code comments
- **Discussions**: [GitHub Discussions](https://github.com/adamzacharia/Quasar2/discussions)
