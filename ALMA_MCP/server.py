"""
ALMA MCP Server
Provides ALMA archive access via Model Context Protocol
Connect this to Claude Desktop to search ALMA from natural language!

Author: Quasar Project
Repository: https://github.com/adamzacharia/Quasar2

NOTE: This folder can be moved outside of Quasar-main to be a standalone project.
Just copy the entire ALMA_MCP folder wherever you want it.
"""

from fastmcp import FastMCP
from typing import Optional, Dict, Any, List
import pandas as pd

# Create the MCP server instance
mcp = FastMCP("ALMA Archive Server")

# Import alminer (for advanced queries)
try:
    import alminer
    ALMINER_AVAILABLE = True
except ImportError:
    ALMINER_AVAILABLE = False
    print("Note: alminer not installed. Some features may be limited.")

# Import pyvo (for TAP queries)
try:
    import pyvo
    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False
    print("Note: pyvo not installed. TAP queries will fail.")

# Import astroquery for name resolution
try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    SIMBAD_AVAILABLE = True
except ImportError:
    SIMBAD_AVAILABLE = False
    print("Note: astroquery not installed. Target name resolution will fail.")


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL LINE DATABASE - Rest frequencies in GHz
# ═══════════════════════════════════════════════════════════════════════════════

SPECTRAL_LINES = {
    # CO isotopologues
    "CO(1-0)": 115.271, "CO(2-1)": 230.538, "CO(3-2)": 345.796,
    "CO(4-3)": 461.041, "CO(5-4)": 576.268, "CO(6-5)": 691.473,
    "CO(7-6)": 806.652, "CO(8-7)": 921.800,
    "13CO(1-0)": 110.201, "13CO(2-1)": 220.399, "13CO(3-2)": 330.588,
    "C18O(1-0)": 109.782, "C18O(2-1)": 219.560, "C18O(3-2)": 329.331,
    "C17O(1-0)": 112.359, "C17O(2-1)": 224.714, "C17O(3-2)": 337.061,
    # HCN and HCO+
    "HCN(1-0)": 88.632, "HCN(2-1)": 177.261, "HCN(3-2)": 265.886, "HCN(4-3)": 354.505,
    "HCO+(1-0)": 89.189, "HCO+(2-1)": 178.375, "HCO+(3-2)": 267.558, "HCO+(4-3)": 356.734,
    "H13CN(1-0)": 86.340, "H13CO+(1-0)": 86.754,
    # Other common molecules
    "CS(2-1)": 97.981, "CS(3-2)": 146.969, "CS(5-4)": 244.936, "CS(7-6)": 342.883,
    "SiO(2-1)": 86.847, "SiO(5-4)": 217.105, "SiO(6-5)": 260.518,
    "N2H+(1-0)": 93.174, "N2H+(3-2)": 279.512,
    "HNC(1-0)": 90.664, "HNC(3-2)": 271.981,
    "CN(1-0)": 113.491, "CN(2-1)": 226.875,
    "CCH(1-0)": 87.317, "CCH(3-2)": 262.004,
    # Water and related
    "H2O(183)": 183.310, "H2O(325)": 325.153, "H2O(380)": 380.197,
    "HDO(1-0)": 80.578, "HDO(2-1)": 241.562,
    # Formaldehyde
    "H2CO(2-1)": 140.840, "H2CO(3-2)": 218.222, "H2CO(4-3)": 290.623,
    # Methanol
    "CH3OH(2-1)": 96.741, "CH3OH(5-4)": 241.791,
    # Atomic/ionic
    "[CI](1-0)": 492.161, "[CI](2-1)": 809.342,
    "[CII]": 1900.537, "[NII]122": 2459.380, "[NII]205": 1461.134,
    "[OI]63": 4744.778, "[OI]145": 2060.069,
    "[OIII]52": 5785.880, "[OIII]88": 3393.006,
    # Continuum placeholders (for reference)
    "continuum_band3": 100.0, "continuum_band6": 230.0, "continuum_band7": 345.0,
}

# Speed of light in km/s for redshift calculations
SPEED_OF_LIGHT_KMS = 299792.458


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Search ALMA by Target Name
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_target(
    target_name: str,
    radius_arcmin: float = 1.0,
    public_only: bool = True
) -> dict:
    """
    Search the ALMA archive for observations of a specific astronomical target.
    
    Args:
        target_name: Name of the astronomical object (e.g., "M87", "Orion KL", "NGC 1234")
        radius_arcmin: Search radius around the target in arcminutes (default: 1.0)
        public_only: If True, only return publicly available data
        
    Returns:
        Dictionary containing:
        - count: Number of observations found
        - observations: List of observation details (target, RA, Dec, bands, proposal)
        - summary: Text summary of the results
    """
    if not SIMBAD_AVAILABLE:
        return {"error": "astroquery not installed - cannot resolve target name"}
    
    # Resolve target name to coordinates
    try:
        result = Simbad.query_object(target_name)
        if result is None or len(result) == 0:
            return {"error": f"Could not resolve '{target_name}' - not found in SIMBAD"}
        
        ra_str = result['RA'][0]
        dec_str = result['DEC'][0]
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
    except Exception as e:
        return {"error": f"Failed to resolve target: {str(e)}"}
    
    # Search ALMA archive using alminer if available, else TAP
    if ALMINER_AVAILABLE:
        try:
            radius_deg = radius_arcmin / 60.0
            df = alminer.conesearch(ra_deg, dec_deg, search_radius=radius_deg, 
                                    public=public_only, print_targets=False)
            
            if df is None or df.empty:
                return {
                    "count": 0,
                    "target_resolved_to": {"ra": round(ra_deg, 4), "dec": round(dec_deg, 4)},
                    "observations": [],
                    "summary": f"No ALMA observations found for {target_name}"
                }
            
            # Format results
            observations = []
            for _, row in df.head(20).iterrows():
                obs = {
                    "target": row.get("target_name", "Unknown"),
                    "ra": round(row.get("s_ra", row.get("ra", 0)), 4),
                    "dec": round(row.get("s_dec", row.get("dec", 0)), 4),
                    "band": str(row.get("band_list", row.get("band", "Unknown"))),
                    "proposal_id": row.get("proposal_id", row.get("project_code", "Unknown")),
                    "integration_time_sec": float(row.get("t_exptime", row.get("integration_time", 0)) or 0)
                }
                observations.append(obs)
            
            return {
                "count": len(df),
                "showing": min(20, len(df)),
                "target_resolved_to": {"ra": round(ra_deg, 4), "dec": round(dec_deg, 4)},
                "observations": observations,
                "summary": f"Found {len(df)} ALMA observations for {target_name}"
            }
            
        except Exception as e:
            return {"error": f"ALMA search failed: {str(e)}"}
    
    # Fallback to TAP
    return search_alma_by_position(ra_deg, dec_deg, radius_arcmin, public_only)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Search ALMA by Coordinates
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_position(
    ra_degrees: float,
    dec_degrees: float,
    radius_arcmin: float = 1.0,
    public_only: bool = True
) -> dict:
    """
    Search the ALMA archive by sky coordinates (cone search).
    
    Args:
        ra_degrees: Right Ascension in degrees (0-360)
        dec_degrees: Declination in degrees (-90 to +90)
        radius_arcmin: Search radius in arcminutes (default: 1.0)
        public_only: If True, only return publicly available data
        
    Returns:
        Dictionary with observation count and details
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        radius_deg = radius_arcmin / 60.0
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            frequency, bandwidth, t_exptime, s_resolution
        FROM ivoa.obscore 
        WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), 
                       CIRCLE('ICRS', {ra_degrees}, {dec_degrees}, {radius_deg})) = 1
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "position": {"ra": ra_degrees, "dec": dec_degrees},
                "observations": [],
                "summary": f"No ALMA observations at RA={ra_degrees:.4f}, Dec={dec_degrees:.4f}"
            }
        
        # Format results
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "ra": round(float(row.get("s_ra", 0)), 4),
                "dec": round(float(row.get("s_dec", 0)), 4),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "position": {"ra": ra_degrees, "dec": dec_degrees},
            "observations": observations,
            "summary": f"Found {len(df)} ALMA observations near RA={ra_degrees:.4f}, Dec={dec_degrees:.4f}"
        }
        
    except Exception as e:
        return {"error": f"Position search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: Search by Proposal/PI
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_proposal(
    proposal_id: str = None,
    pi_name: str = None,
    science_category: str = None
) -> dict:
    """
    Search ALMA archive by proposal metadata.
    
    Args:
        proposal_id: ALMA proposal ID (e.g., "2023.1.00001.S")
        pi_name: Principal Investigator name (partial match)
        science_category: Science category like "Galaxy evolution", "Star formation"
        
    Returns:
        Dictionary with matching observations
    """
    if not any([proposal_id, pi_name, science_category]):
        return {"error": "Must provide at least one search parameter (proposal_id, pi_name, or science_category)"}
    
    # Try alminer first if available
    if ALMINER_AVAILABLE:
        try:
            search_dict = {}
            if proposal_id:
                search_dict["proposal_id"] = [proposal_id]
            if pi_name:
                search_dict["pi_name"] = [pi_name]
            if science_category:
                search_dict["scientific_category"] = [science_category]
            
            df = alminer.keysearch(search_dict, public=True, print_targets=False)
            
            if df is None or df.empty:
                return {"count": 0, "observations": [], "summary": "No matching observations found"}
            
            # Format results
            observations = []
            for _, row in df.head(20).iterrows():
                obs = {
                    "target": row.get("target_name", "Unknown"),
                    "proposal_id": row.get("proposal_id", row.get("project_code", "Unknown")),
                    "pi": row.get("obs_creator_name", "Unknown"),
                    "band": str(row.get("band_list", row.get("band", "Unknown"))),
                }
                observations.append(obs)
            
            return {
                "count": len(df),
                "showing": min(20, len(df)),
                "observations": observations,
                "summary": f"Found {len(df)} observations matching criteria"
            }
            
        except Exception as e:
            pass  # Fall through to TAP
    
    # Fallback to TAP query
    if not PYVO_AVAILABLE:
        return {"error": "Neither alminer nor pyvo available"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        conditions = []
        if proposal_id:
            conditions.append(f"proposal_id LIKE '%{proposal_id}%'")
        if pi_name:
            conditions.append(f"LOWER(obs_creator_name) LIKE LOWER('%{pi_name}%')")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id, obs_creator_name
        FROM ivoa.obscore 
        WHERE {where_clause}
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "pi": row.get("obs_creator_name", "Unknown"),
                "band": str(row.get("band_list", "Unknown")),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "observations": observations,
            "summary": f"Found {len(df)} observations matching criteria"
        }
        
    except Exception as e:
        return {"error": f"Proposal search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: Check Line Coverage
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def check_alma_line_coverage(
    target_name: str,
    line_frequency_ghz: float,
    redshift: float = 0.0
) -> dict:
    """
    Check if ALMA observations of a target cover a specific spectral line.
    
    Args:
        target_name: Name of the target (will search ALMA first)
        line_frequency_ghz: Rest frequency of the line in GHz
        redshift: Source redshift for Doppler correction (default: 0)
        
    Returns:
        Dictionary with observations that cover the specified line
    """
    if not ALMINER_AVAILABLE:
        return {"error": "alminer library required for line coverage check"}
    
    if not SIMBAD_AVAILABLE:
        return {"error": "astroquery required for target resolution"}
    
    try:
        # Resolve target
        result = Simbad.query_object(target_name)
        if result is None or len(result) == 0:
            return {"error": f"Could not resolve '{target_name}'"}
        
        ra_str = result['RA'][0]
        dec_str = result['DEC'][0]
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        
        # Search for observations
        df = alminer.conesearch(coord.ra.deg, coord.dec.deg, search_radius=0.016, print_targets=False)
        
        if df is None or df.empty:
            return {
                "total_observations": 0,
                "covering_line": 0,
                "summary": f"No ALMA observations found for {target_name}"
            }
        
        # Check line coverage
        covered_df = alminer.line_coverage(df, line_freq=line_frequency_ghz, z=redshift, print_targets=False)
        
        if covered_df is None or covered_df.empty:
            return {
                "total_observations": len(df),
                "covering_line": 0,
                "line_frequency_ghz": line_frequency_ghz,
                "redshift": redshift,
                "observed_frequency_ghz": round(line_frequency_ghz / (1 + redshift), 4),
                "summary": f"None of the {len(df)} observations cover {line_frequency_ghz} GHz at z={redshift}"
            }
        
        return {
            "total_observations": len(df),
            "covering_line": len(covered_df),
            "line_frequency_ghz": line_frequency_ghz,
            "redshift": redshift,
            "observed_frequency_ghz": round(line_frequency_ghz / (1 + redshift), 4),
            "summary": f"{len(covered_df)} of {len(df)} observations cover the line"
        }
        
    except Exception as e:
        return {"error": f"Line coverage check failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4b: Get Source Redshift from SIMBAD
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_source_redshift(target_name: str) -> dict:
    """
    Get the redshift or radial velocity of an astronomical source from SIMBAD.
    
    Args:
        target_name: Name of the astronomical object (e.g., "M87", "NGC 1068", "3C 273")
        
    Returns:
        Dictionary containing:
        - target: The queried target name
        - redshift: Redshift value (z) if available
        - radial_velocity_kms: Radial velocity in km/s if available
        - velocity_type: Type of velocity measurement (optical, radio, etc.)
        - source: Data source (SIMBAD)
    """
    if not SIMBAD_AVAILABLE:
        return {"error": "astroquery not installed - cannot query SIMBAD"}
    
    try:
        # Create custom SIMBAD query with velocity fields
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('velocity', 'rvz_radvel', 'rvz_type', 'rvz_error')
        
        result = custom_simbad.query_object(target_name)
        
        if result is None or len(result) == 0:
            return {"error": f"Could not find '{target_name}' in SIMBAD"}
        
        # Extract velocity information
        row = result[0]
        
        # Try to get radial velocity (in km/s)
        radial_velocity = None
        velocity_type = None
        redshift = None
        
        # Check rvz_radvel column (radial velocity)
        if 'rvz_radvel' in result.colnames:
            rv_val = row['rvz_radvel']
            if rv_val is not None and not (hasattr(rv_val, 'mask') and rv_val.mask):
                radial_velocity = float(rv_val)
                # Convert to redshift: z = v/c (non-relativistic approximation, good for v << c)
                redshift = radial_velocity / SPEED_OF_LIGHT_KMS
        
        # Check velocity type
        if 'rvz_type' in result.colnames:
            vt_val = row['rvz_type']
            if vt_val is not None and not (hasattr(vt_val, 'mask') and vt_val.mask):
                velocity_type = str(vt_val)
        
        # If no radial velocity found, return what we have
        if radial_velocity is None:
            return {
                "target": target_name,
                "redshift": None,
                "radial_velocity_kms": None,
                "velocity_type": None,
                "source": "SIMBAD",
                "note": f"No radial velocity data available for {target_name} in SIMBAD. You may need to specify redshift manually."
            }
        
        return {
            "target": target_name,
            "redshift": round(redshift, 6),
            "radial_velocity_kms": round(radial_velocity, 2),
            "velocity_type": velocity_type,
            "source": "SIMBAD",
            "summary": f"{target_name} has z = {redshift:.6f} (v = {radial_velocity:.1f} km/s)"
        }
        
    except Exception as e:
        return {"error": f"Failed to query SIMBAD: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4c: Smart Spectral Line Coverage Check (Auto-resolves line name & redshift)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def check_spectral_line_coverage(
    target_name: str,
    line_name: str,
    redshift: Optional[float] = None
) -> dict:
    """
    Check if ALMA observations of a target cover a specific spectral line.
    This is the "smart" version that automatically:
    1. Resolves the line name to its rest frequency (e.g., "CO(2-1)" -> 230.538 GHz)
    2. Looks up the source redshift from SIMBAD if not provided
    3. Calculates the observed frequency and checks coverage
    
    Args:
        target_name: Name of the target (e.g., "M87", "NGC 1068", "Orion KL")
        line_name: Name of the spectral line (e.g., "CO(2-1)", "HCN(1-0)", "[CII]")
        redshift: Optional source redshift. If not provided, will be looked up from SIMBAD.
        
    Returns:
        Dictionary with line coverage results including:
        - rest_frequency_ghz: Rest frequency of the line
        - observed_frequency_ghz: Doppler-shifted frequency at the source
        - redshift_used: The redshift value used (auto or manual)
        - total_observations: Total ALMA observations of the target
        - covering_line: Number of observations covering the line
    
    Examples:
        - check_spectral_line_coverage("M87", "CO(2-1)")
        - check_spectral_line_coverage("NGC 1068", "HCN(1-0)")
        - check_spectral_line_coverage("3C 273", "[CII]", redshift=0.158)
    """
    if not ALMINER_AVAILABLE:
        return {"error": "alminer library required for line coverage check"}
    
    if not SIMBAD_AVAILABLE:
        return {"error": "astroquery required for target resolution and redshift lookup"}
    
    # Step 1: Resolve line name to frequency
    line_name_upper = line_name.upper().replace(" ", "")
    line_name_clean = line_name.replace(" ", "")
    
    rest_frequency = None
    matched_line = None
    
    # Try exact match first
    for known_line, freq in SPECTRAL_LINES.items():
        if known_line.upper().replace(" ", "") == line_name_upper:
            rest_frequency = freq
            matched_line = known_line
            break
    
    # Try partial match
    if rest_frequency is None:
        for known_line, freq in SPECTRAL_LINES.items():
            if line_name_upper in known_line.upper().replace(" ", ""):
                rest_frequency = freq
                matched_line = known_line
                break
    
    if rest_frequency is None:
        # List available lines for the user
        available = ", ".join(list(SPECTRAL_LINES.keys())[:20]) + "..."
        return {
            "error": f"Unknown spectral line '{line_name}'. Please use a known line name or provide the frequency directly.",
            "hint": f"Available lines include: {available}",
            "available_lines": list(SPECTRAL_LINES.keys())
        }
    
    # Step 2: Get redshift if not provided
    redshift_source = "user-provided"
    if redshift is None:
        redshift_result = get_source_redshift(target_name)
        if "error" in redshift_result:
            # No redshift available, use z=0 with a warning
            redshift = 0.0
            redshift_source = "default (z=0, no SIMBAD data)"
        elif redshift_result.get("redshift") is None:
            redshift = 0.0
            redshift_source = "default (z=0, no velocity in SIMBAD)"
        else:
            redshift = redshift_result["redshift"]
            redshift_source = f"SIMBAD (v={redshift_result.get('radial_velocity_kms', 'N/A')} km/s)"
    
    # Step 3: Calculate observed frequency
    observed_frequency = rest_frequency / (1 + redshift)
    
    # Step 4: Search for ALMA observations of the target
    try:
        result = Simbad.query_object(target_name)
        if result is None or len(result) == 0:
            return {"error": f"Could not resolve '{target_name}' in SIMBAD"}
        
        ra_str = result['RA'][0]
        dec_str = result['DEC'][0]
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        
        # Search for observations
        df = alminer.conesearch(coord.ra.deg, coord.dec.deg, search_radius=0.016, print_targets=False)
        
        if df is None or df.empty:
            return {
                "target": target_name,
                "line_name": matched_line,
                "rest_frequency_ghz": rest_frequency,
                "observed_frequency_ghz": round(observed_frequency, 4),
                "redshift_used": redshift,
                "redshift_source": redshift_source,
                "total_observations": 0,
                "covering_line": 0,
                "summary": f"No ALMA observations found for {target_name}"
            }
        
        # Step 5: Check line coverage
        covered_df = alminer.line_coverage(df, line_freq=rest_frequency, z=redshift, print_targets=False)
        
        covering_count = 0 if covered_df is None or covered_df.empty else len(covered_df)
        
        return {
            "target": target_name,
            "line_name": matched_line,
            "rest_frequency_ghz": rest_frequency,
            "observed_frequency_ghz": round(observed_frequency, 4),
            "redshift_used": round(redshift, 6),
            "redshift_source": redshift_source,
            "total_observations": len(df),
            "covering_line": covering_count,
            "summary": f"{covering_count} of {len(df)} ALMA observations of {target_name} cover {matched_line} (rest: {rest_frequency} GHz, observed: {observed_frequency:.3f} GHz at z={redshift:.6f})"
        }
        
    except Exception as e:
        return {"error": f"Spectral line coverage check failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: Get ALMA Info
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_alma_info() -> dict:
    """
    Get information about ALMA capabilities and frequency bands.
    
    Returns:
        Dictionary with ALMA band information and capabilities
    """
    return {
        "telescope": "Atacama Large Millimeter/submillimeter Array",
        "location": "Atacama Desert, Chile (5000m altitude)",
        "operator": "NRAO, ESO, NAOJ",
        "antennas": "66 high-precision antennas (54 x 12m + 12 x 7m)",
        "bands": {
            "Band 3": {"frequency_ghz": "84-116", "wavelength_mm": "2.6-3.6"},
            "Band 4": {"frequency_ghz": "125-163", "wavelength_mm": "1.8-2.4"},
            "Band 5": {"frequency_ghz": "163-211", "wavelength_mm": "1.4-1.8"},
            "Band 6": {"frequency_ghz": "211-275", "wavelength_mm": "1.1-1.4"},
            "Band 7": {"frequency_ghz": "275-373", "wavelength_mm": "0.8-1.1"},
            "Band 8": {"frequency_ghz": "385-500", "wavelength_mm": "0.6-0.8"},
            "Band 9": {"frequency_ghz": "602-720", "wavelength_mm": "0.4-0.5"},
            "Band 10": {"frequency_ghz": "787-950", "wavelength_mm": "0.3-0.4"}
        },
        "common_lines": {
            "CO(1-0)": "115.271 GHz",
            "CO(2-1)": "230.538 GHz",
            "CO(3-2)": "345.796 GHz",
            "13CO(1-0)": "110.201 GHz",
            "13CO(2-1)": "220.399 GHz",
            "HCN(1-0)": "88.632 GHz",
            "HCO+(1-0)": "89.189 GHz",
            "CS(2-1)": "97.981 GHz",
            "SiO(2-1)": "86.847 GHz",
            "N2H+(1-0)": "93.174 GHz"
        },
        "science_categories": [
            "Cosmology",
            "Galaxy evolution", 
            "ISM and star formation",
            "Disks and planet formation",
            "Stars and stellar evolution",
            "Solar system",
            "Sun"
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6: Search by Frequency Range
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_frequency(
    min_freq_ghz: float,
    max_freq_ghz: float,
    target_name: str = None
) -> dict:
    """
    Search ALMA archive for observations covering a specific frequency range.
    
    Args:
        min_freq_ghz: Minimum frequency in GHz
        max_freq_ghz: Maximum frequency in GHz
        target_name: Optional target name to narrow search
        
    Returns:
        Dictionary with observations covering the frequency range
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # Build query for frequency overlap (frequency in Hz in ALMA TAP)
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            frequency, bandwidth, t_exptime, s_resolution
        FROM ivoa.obscore 
        WHERE frequency >= {min_freq_ghz * 1e9} 
          AND frequency <= {max_freq_ghz * 1e9}
        '''
        
        if target_name:
            query += f" AND LOWER(target_name) LIKE LOWER('%{target_name}%')"
        
        query += " ORDER BY frequency"
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "frequency_range_ghz": [min_freq_ghz, max_freq_ghz],
                "observations": [],
                "summary": f"No observations found in {min_freq_ghz}-{max_freq_ghz} GHz range"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "frequency_ghz": round(float(row.get("frequency", 0)) / 1e9, 2),
                "bandwidth_ghz": round(float(row.get("bandwidth", 0)) / 1e9, 2),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "frequency_range_ghz": [min_freq_ghz, max_freq_ghz],
            "observations": observations,
            "summary": f"Found {len(df)} observations in {min_freq_ghz}-{max_freq_ghz} GHz range"
        }
        
    except Exception as e:
        return {"error": f"Frequency search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7: Search by Angular Resolution
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_resolution(
    max_resolution_arcsec: float,
    min_resolution_arcsec: float = 0.0,
    target_name: str = None
) -> dict:
    """
    Search ALMA archive for observations with specific angular resolution.
    
    Args:
        max_resolution_arcsec: Maximum angular resolution in arcseconds
        min_resolution_arcsec: Minimum angular resolution in arcseconds (default: 0)
        target_name: Optional target name to narrow search
        
    Returns:
        Dictionary with observations matching resolution criteria
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # Resolution in ALMA TAP is in degrees, convert from arcsec
        max_res_deg = max_resolution_arcsec / 3600.0
        min_res_deg = min_resolution_arcsec / 3600.0
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            s_resolution, frequency, t_exptime
        FROM ivoa.obscore 
        WHERE s_resolution <= {max_res_deg}
          AND s_resolution >= {min_res_deg}
        '''
        
        if target_name:
            query += f" AND LOWER(target_name) LIKE LOWER('%{target_name}%')"
        
        query += " ORDER BY s_resolution"
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "resolution_range_arcsec": [min_resolution_arcsec, max_resolution_arcsec],
                "observations": [],
                "summary": f"No observations found with resolution {min_resolution_arcsec}-{max_resolution_arcsec} arcsec"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            res_arcsec = float(row.get("s_resolution", 0)) * 3600  # Convert back to arcsec
            obs = {
                "target": row.get("target_name", "Unknown"),
                "resolution_arcsec": round(res_arcsec, 3),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "resolution_range_arcsec": [min_resolution_arcsec, max_resolution_arcsec],
            "observations": observations,
            "summary": f"Found {len(df)} observations with resolution < {max_resolution_arcsec} arcsec"
        }
        
    except Exception as e:
        return {"error": f"Resolution search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8: Custom SQL/TAP Query
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def run_alma_tap_query(
    sql_query: str,
    max_rows: int = 100
) -> dict:
    """
    Run a custom SQL/ADQL query against the ALMA TAP service.
    
    Args:
        sql_query: ADQL/SQL query to execute. Query from 'ivoa.obscore' table.
                   Available columns include: target_name, s_ra, s_dec, band_list,
                   proposal_id, frequency, bandwidth, t_exptime, s_resolution,
                   obs_publisher_did, member_ous_uid, access_url, obs_creator_name
        max_rows: Maximum rows to return (default: 100, max: 1000)
                   
    Returns:
        Dictionary with query results or error message
        
    Example queries:
        - "SELECT target_name, band_list FROM ivoa.obscore WHERE target_name LIKE '%M87%'"
        - "SELECT DISTINCT proposal_id FROM ivoa.obscore WHERE band_list LIKE '%6%'"
        - "SELECT COUNT(*) FROM ivoa.obscore WHERE s_resolution < 0.0001"
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        # Safety limits
        max_rows = min(max_rows, 1000)
        
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # Add TOP clause if not present for SELECT queries
        sql_upper = sql_query.upper().strip()
        if 'TOP' not in sql_upper and sql_upper.startswith('SELECT') and 'COUNT' not in sql_upper:
            sql_query = sql_query.replace('SELECT', f'SELECT TOP {max_rows}', 1)
        
        result = service.search(sql_query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "columns": list(df.columns),
                "rows": [],
                "summary": "Query returned no results"
            }
        
        # Convert DataFrame to list of dicts, handling numpy types
        rows = []
        for _, row in df.head(max_rows).iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                # Convert numpy types to Python types for JSON serialization
                if hasattr(val, 'item'):
                    val = val.item()
                row_dict[col] = val
            rows.append(row_dict)
        
        return {
            "count": len(df),
            "showing": min(max_rows, len(df)),
            "columns": list(df.columns),
            "rows": rows,
            "summary": f"Query returned {len(df)} rows"
        }
        
    except Exception as e:
        return {
            "error": f"TAP query failed: {str(e)}",
            "hint": "Check your SQL syntax. Query the 'ivoa.obscore' table. "
                    "Use single quotes for string values."
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9: Search by ALMA Source Name (as given by PI)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_source_name(
    source_name: str,
    exact_match: bool = False
) -> dict:
    """
    Search ALMA archive by the target name as specified by the PI in the proposal.
    This searches the 'target_name' field directly, not using name resolution.
    
    Args:
        source_name: The source name to search for (e.g., "Centaurus A", "NGC 1234", "SPT0311-58")
        exact_match: If True, search for exact name match; if False, partial/substring match
        
    Returns:
        Dictionary with matching observations
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        if exact_match:
            query = f'''
            SELECT TOP 100
                target_name, s_ra, s_dec, band_list, proposal_id,
                frequency, t_exptime, s_resolution, dataproduct_type
            FROM ivoa.obscore 
            WHERE target_name = '{source_name}'
            '''
        else:
            query = f'''
            SELECT TOP 100
                target_name, s_ra, s_dec, band_list, proposal_id,
                frequency, t_exptime, s_resolution, dataproduct_type
            FROM ivoa.obscore 
            WHERE LOWER(target_name) LIKE LOWER('%{source_name}%')
            '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "observations": [],
                "summary": f"No ALMA observations found with source name '{source_name}'"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "ra": round(float(row.get("s_ra", 0)), 4),
                "dec": round(float(row.get("s_dec", 0)), 4),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "data_type": row.get("dataproduct_type", "Unknown"),
            }
            observations.append(obs)
        
        # Get unique target names found
        unique_targets = df['target_name'].unique().tolist()[:10]
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "unique_targets_found": unique_targets,
            "observations": observations,
            "summary": f"Found {len(df)} observations matching source name '{source_name}'"
        }
        
    except Exception as e:
        return {"error": f"Source name search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10: Search by Bibliography/Publication
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_bibliography(
    bibcode: str = None,
    journal_name: str = None,
    first_author: str = None,
    publication_year: int = None
) -> dict:
    """
    Search ALMA archive for observations used in specific publications.
    
    Args:
        bibcode: NASA ADS bibcode (e.g., "2017ApJ...834..140R") or partial match
        journal_name: Journal name substring (e.g., "Nature", "ApJ", "A&A")
        first_author: First author name (partial match)
        publication_year: Year of publication
        
    Returns:
        Dictionary with observations linked to publications
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    if not any([bibcode, journal_name, first_author, publication_year]):
        return {"error": "Must provide at least one search parameter"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        conditions = []
        if bibcode:
            conditions.append(f"bib_reference LIKE '%{bibcode}%'")
        if journal_name:
            conditions.append(f"bib_reference LIKE '%{journal_name}%'")
        if first_author:
            conditions.append(f"LOWER(first_author) LIKE LOWER('%{first_author}%')")
        if publication_year:
            conditions.append(f"publication_year = {publication_year}")
        
        where_clause = " AND ".join(conditions)
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            bib_reference, first_author, publication_year, pub_title
        FROM ivoa.obscore 
        WHERE {where_clause}
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "observations": [],
                "summary": "No ALMA observations found with matching bibliography"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "bibcode": row.get("bib_reference", "Unknown"),
                "first_author": row.get("first_author", "Unknown"),
                "pub_year": row.get("publication_year", "Unknown"),
                "pub_title": str(row.get("pub_title", "Unknown"))[:100],
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "observations": observations,
            "summary": f"Found {len(df)} observations with matching publications"
        }
        
    except Exception as e:
        return {"error": f"Bibliography search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 11: Search by Member OUS ID
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_member_ous(
    member_ous_id: str
) -> dict:
    """
    Search ALMA archive by Member OUS dataset identifier.
    
    Args:
        member_ous_id: ALMA Member OUS ID in the form "uid://A001/X123/X456" 
                       or "uid___A001_X123_X456"
                       
    Returns:
        Dictionary with matching dataset details
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # Normalize the UID format
        normalized_uid = member_ous_id.replace("___", "://").replace("_", "/")
        
        query = f'''
        SELECT *
        FROM ivoa.obscore 
        WHERE member_ous_uid = '{normalized_uid}'
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "member_ous_id": normalized_uid,
                "observations": [],
                "summary": f"No ALMA data found for Member OUS ID '{normalized_uid}'"
            }
        
        observations = []
        for _, row in df.iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "ra": round(float(row.get("s_ra", 0)), 4),
                "dec": round(float(row.get("s_dec", 0)), 4),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "frequency_ghz": round(float(row.get("frequency", 0)), 2),
                "resolution_arcsec": round(float(row.get("s_resolution", 0)), 3),
                "data_type": row.get("dataproduct_type", "Unknown"),
                "access_url": row.get("access_url", "Unknown"),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "member_ous_id": normalized_uid,
            "observations": observations,
            "summary": f"Found {len(df)} data products for Member OUS ID"
        }
        
    except Exception as e:
        return {"error": f"Member OUS search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 12: Search by Data Type (cube vs image)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_data_type(
    data_type: str,
    target_name: str = None,
    science_keyword: str = None,
    band: int = None
) -> dict:
    """
    Search ALMA archive by data product type (spectral cubes vs continuum images).
    
    Args:
        data_type: "cube" for spectral line data or "image" for continuum
        target_name: Optional target name filter (partial match)
        science_keyword: Optional science keyword filter
        band: Optional ALMA band number (3-10)
        
    Returns:
        Dictionary with observations of specified data type
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    if data_type.lower() not in ["cube", "image"]:
        return {"error": "data_type must be 'cube' or 'image'"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        conditions = [f"dataproduct_type = '{data_type.lower()}'"]
        conditions.append("science_observation = 'T'")
        
        if target_name:
            conditions.append(f"LOWER(target_name) LIKE LOWER('%{target_name}%')")
        if science_keyword:
            conditions.append(f"LOWER(science_keyword) LIKE LOWER('%{science_keyword}%')")
        if band:
            conditions.append(f"band_list LIKE '%{band}%'")
        
        where_clause = " AND ".join(conditions)
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            frequency, t_exptime, s_resolution, science_keyword
        FROM ivoa.obscore 
        WHERE {where_clause}
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "data_type": data_type,
                "observations": [],
                "summary": f"No {data_type} observations found matching criteria"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "frequency_ghz": round(float(row.get("frequency", 0)), 2),
                "resolution_arcsec": round(float(row.get("s_resolution", 0)), 3),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "data_type": data_type,
            "observations": observations,
            "summary": f"Found {len(df)} {data_type} observations"
        }
        
    except Exception as e:
        return {"error": f"Data type search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 13: Search by Science Keyword (enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_science_keyword(
    science_keyword: str,
    data_type: str = None,
    band: int = None,
    science_observation_only: bool = True
) -> dict:
    """
    Search ALMA archive by science keyword with optional filters.
    
    Args:
        science_keyword: ALMA science keyword (e.g., "Quasars", "Sub-mm Galaxies (SMG)", 
                        "Exoplanets", "High-z Universe", "Active Galactic Nuclei")
        data_type: Optional filter: "cube" for spectral or "image" for continuum
        band: Optional ALMA band number (3-10)
        science_observation_only: If True, only return science observations (default: True)
        
    Returns:
        Dictionary with observations matching the science keyword
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        conditions = [f"LOWER(science_keyword) LIKE LOWER('%{science_keyword}%')"]
        
        if science_observation_only:
            conditions.append("science_observation = 'T'")
        if data_type and data_type.lower() in ["cube", "image"]:
            conditions.append(f"dataproduct_type = '{data_type.lower()}'")
        if band:
            conditions.append(f"band_list LIKE '%{band}%'")
        
        where_clause = " AND ".join(conditions)
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            frequency, t_exptime, s_resolution, science_keyword, dataproduct_type
        FROM ivoa.obscore 
        WHERE {where_clause}
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "science_keyword": science_keyword,
                "observations": [],
                "summary": f"No observations found with science keyword '{science_keyword}'"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "band": str(row.get("band_list", "Unknown")),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "data_type": row.get("dataproduct_type", "Unknown"),
                "science_keyword": row.get("science_keyword", "Unknown"),
            }
            observations.append(obs)
        
        # Get unique targets
        unique_targets = df['target_name'].unique().tolist()[:10]
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "science_keyword": science_keyword,
            "unique_targets": unique_targets,
            "observations": observations,
            "summary": f"Found {len(df)} observations with science keyword '{science_keyword}'"
        }
        
    except Exception as e:
        return {"error": f"Science keyword search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 14: Search by Proposal Abstract
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_abstract(
    search_terms: str,
    search_pub_abstract: bool = False
) -> dict:
    """
    Search ALMA proposals by keywords in the abstract text.
    
    Args:
        search_terms: Keywords to search for in proposal abstracts (case-insensitive)
        search_pub_abstract: If True, also search publication abstracts
        
    Returns:
        Dictionary with observations from matching proposals
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        if search_pub_abstract:
            abstract_condition = f"""
                (LOWER(proposal_abstract) LIKE LOWER('%{search_terms}%')
                 OR LOWER(pub_abstract) LIKE LOWER('%{search_terms}%'))
            """
        else:
            abstract_condition = f"LOWER(proposal_abstract) LIKE LOWER('%{search_terms}%')"
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            obs_creator_name, science_keyword, t_exptime
        FROM ivoa.obscore 
        WHERE {abstract_condition}
        AND science_observation = 'T'
        GROUP BY target_name, s_ra, s_dec, band_list, proposal_id,
                 obs_creator_name, science_keyword, t_exptime
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "search_terms": search_terms,
                "observations": [],
                "summary": f"No proposals found with abstract containing '{search_terms}'"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "proposal_id": row.get("proposal_id", "Unknown"),
                "pi": row.get("obs_creator_name", "Unknown"),
                "band": str(row.get("band_list", "Unknown")),
                "science_keyword": row.get("science_keyword", "Unknown"),
            }
            observations.append(obs)
        
        # Get unique proposals
        unique_proposals = df['proposal_id'].unique().tolist()[:10]
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "search_terms": search_terms,
            "unique_proposals": unique_proposals,
            "observations": observations,
            "summary": f"Found {len(df)} observations from proposals mentioning '{search_terms}'"
        }
        
    except Exception as e:
        return {"error": f"Abstract search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 15: Search by Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_alma_by_sensitivity(
    max_sensitivity_mjy: float,
    sensitivity_type: str = "continuum",
    target_name: str = None,
    band: int = None
) -> dict:
    """
    Search ALMA archive for observations with specific sensitivity limits.
    
    Args:
        max_sensitivity_mjy: Maximum sensitivity in mJy/beam
        sensitivity_type: "continuum" for cont_sensitivity_bandwidth or 
                         "line" for sensitivity_10kms (line at 10 km/s)
        target_name: Optional target name filter
        band: Optional ALMA band number (3-10)
        
    Returns:
        Dictionary with observations meeting sensitivity criteria
    """
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    try:
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        if sensitivity_type.lower() == "continuum":
            sens_column = "cont_sensitivity_bandwidth"
        else:
            sens_column = "sensitivity_10kms"
        
        conditions = [
            f"{sens_column} <= {max_sensitivity_mjy}",
            f"{sens_column} > 0",
            "science_observation = 'T'"
        ]
        
        if target_name:
            conditions.append(f"LOWER(target_name) LIKE LOWER('%{target_name}%')")
        if band:
            conditions.append(f"band_list LIKE '%{band}%'")
        
        where_clause = " AND ".join(conditions)
        
        query = f'''
        SELECT TOP 100
            target_name, s_ra, s_dec, band_list, proposal_id,
            {sens_column} as sensitivity, s_resolution, frequency
        FROM ivoa.obscore 
        WHERE {where_clause}
        ORDER BY {sens_column}
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if df.empty:
            return {
                "count": 0,
                "max_sensitivity_mjy": max_sensitivity_mjy,
                "sensitivity_type": sensitivity_type,
                "observations": [],
                "summary": f"No observations found with {sensitivity_type} sensitivity <= {max_sensitivity_mjy} mJy/beam"
            }
        
        observations = []
        for _, row in df.head(20).iterrows():
            obs = {
                "target": row.get("target_name", "Unknown"),
                "sensitivity_mjy": round(float(row.get("sensitivity", 0)), 4),
                "band": str(row.get("band_list", "Unknown")),
                "resolution_arcsec": round(float(row.get("s_resolution", 0)), 3),
                "proposal_id": row.get("proposal_id", "Unknown"),
            }
            observations.append(obs)
        
        return {
            "count": len(df),
            "showing": min(20, len(df)),
            "max_sensitivity_mjy": max_sensitivity_mjy,
            "sensitivity_type": sensitivity_type,
            "observations": observations,
            "summary": f"Found {len(df)} observations with {sensitivity_type} sensitivity <= {max_sensitivity_mjy} mJy/beam"
        }
        
    except Exception as e:
        return {"error": f"Sensitivity search failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 16: Query Multiple Sources (Batch)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_alma_multiple_sources(
    source_names: List[str],
    radius_arcmin: float = 1.0
) -> dict:
    """
    Query ALMA archive for multiple sources at once (batch query).
    Resolves each source name and searches the archive.
    
    Args:
        source_names: List of source names to search for (max 20 per query)
        radius_arcmin: Search radius in arcminutes (default: 1.0)
        
    Returns:
        Dictionary with observations for each source
    """
    if not SIMBAD_AVAILABLE:
        return {"error": "astroquery not installed - cannot resolve target names"}
    if not PYVO_AVAILABLE:
        return {"error": "pyvo library not installed"}
    
    # Limit batch size
    source_names = source_names[:20]
    
    try:
        results_by_source = {}
        total_observations = 0
        
        for source_name in source_names:
            try:
                # Resolve source name
                result = Simbad.query_object(source_name)
                if result is None or len(result) == 0:
                    results_by_source[source_name] = {
                        "status": "not_resolved",
                        "count": 0,
                        "message": f"Could not resolve '{source_name}' in SIMBAD"
                    }
                    continue
                
                ra_str = result['RA'][0]
                dec_str = result['DEC'][0]
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra_deg = coord.ra.deg
                dec_deg = coord.dec.deg
                
                # Search ALMA
                radius_deg = radius_arcmin / 60.0
                tap_url = 'https://almascience.nrao.edu/tap'
                service = pyvo.dal.TAPService(tap_url)
                
                query = f'''
                SELECT TOP 50
                    target_name, band_list, proposal_id, t_exptime
                FROM ivoa.obscore 
                WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), 
                               CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1
                '''
                
                tap_result = service.search(query)
                df = tap_result.to_table().to_pandas()
                
                if df.empty:
                    results_by_source[source_name] = {
                        "status": "no_data",
                        "count": 0,
                        "coordinates": {"ra": round(ra_deg, 4), "dec": round(dec_deg, 4)},
                        "message": "No ALMA observations found"
                    }
                else:
                    total_observations += len(df)
                    unique_bands = df['band_list'].unique().tolist()
                    results_by_source[source_name] = {
                        "status": "found",
                        "count": len(df),
                        "coordinates": {"ra": round(ra_deg, 4), "dec": round(dec_deg, 4)},
                        "bands_observed": unique_bands,
                    }
                    
            except Exception as e:
                results_by_source[source_name] = {
                    "status": "error",
                    "count": 0,
                    "message": str(e)
                }
        
        # Summary statistics
        sources_with_data = sum(1 for s in results_by_source.values() if s.get("status") == "found")
        
        return {
            "total_sources_queried": len(source_names),
            "sources_with_alma_data": sources_with_data,
            "total_observations": total_observations,
            "results_by_source": results_by_source,
            "summary": f"{sources_with_data} of {len(source_names)} sources have ALMA observations"
        }
        
    except Exception as e:
        return {"error": f"Batch query failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Run the server
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("ALMA MCP Server")
    print("=" * 60)
    print(f"alminer available: {ALMINER_AVAILABLE}")
    print(f"pyvo available: {PYVO_AVAILABLE}")
    print(f"astroquery available: {SIMBAD_AVAILABLE}")
    print("=" * 60)
    
    # Check for SSE mode (for Responses API integration)
    if "--sse" in sys.argv:
        port = 8000
        # Check for custom port
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        print(f"Starting server in SSE mode on port {port}...")
        print("Connect via: http://localhost:{port}/sse")
        print("=" * 60)
        mcp.run(transport="sse", port=port)
    else:
        print("Starting server in stdio mode...")
        print("Connect via Claude Desktop!")
        print("(Use --sse for Responses API integration)")
        print("=" * 60)
        mcp.run()

