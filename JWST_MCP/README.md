# Quasar JWST MCP

This folder packages Quasar's JWST archive capabilities as a standalone MCP server backed by MAST.

## Tools

- `jwst_search_by_target`: JWST target-name search through MAST.
- `jwst_search_by_position`: JWST cone search by RA/Dec.
- `jwst_search_by_criteria`: program, instrument, filter, target, date, and product-type search.
- `jwst_get_products`: file-level product listing for the last JWST search.
- `jwst_observation_summary`: grounded summary over rows from a previous MCP call.

## Run

```bash
pip install -r JWST_MCP/requirements.txt
python JWST_MCP/server.py
```

## Example MCP Client Config

```json
{
  "mcpServers": {
    "quasar-jwst": {
      "command": "python",
      "args": ["C:/Users/adama/Desktop/Quasar-main/JWST_MCP/server.py"]
    }
  }
}
```

## Recommended Use Cases

- JWST archive discovery for named targets.
- NIRCam, NIRSpec, MIRI, NIRISS, and FGS filtering.
- Program-ID and filter-specific searches.
- Product-list retrieval before downloading FITS data.
