"""
Test script for the Advanced Visualization Service
Run this to verify all visualization functions work correctly
"""

import sys
import os
import types

# Python 3.13 compatibility fix
if sys.version_info >= (3, 13):
    if "cgi" not in sys.modules:
        mock_cgi = types.ModuleType("cgi")
        mock_cgi.parse_header = lambda x: (x, {})
        sys.modules["cgi"] = mock_cgi

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Test imports
print("=" * 60)
print("TESTING VISUALIZATION SERVICE")
print("=" * 60)

try:
    from services.visualization_service import (
        get_visualization_service,
        SPECTRAL_LINES,
        ALMA_BANDS,
        AdvancedVisualizationService
    )
    print("✓ Visualization service imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create test data
print("\n--- Creating Test Data ---")
np.random.seed(42)
n_obs = 50

test_df = pd.DataFrame({
    'target_name': [f'Source_{i}' for i in range(n_obs)],
    's_ra': np.random.uniform(0, 360, n_obs),
    's_dec': np.random.uniform(-90, 90, n_obs),
    'Band': np.random.choice([3, 4, 5, 6, 7], n_obs),
    'freq_min': np.random.uniform(84, 400, n_obs),
    'freq_max': np.random.uniform(100, 450, n_obs),
    't_exptime': np.random.uniform(60, 36000, n_obs),
    'sensitivity': np.random.uniform(0.01, 1, n_obs),
    'project_code': [f'2024.1.{i:05d}.S' for i in range(n_obs)],
    'telescope': ['ALMA'] * n_obs,
    'obs_date': pd.date_range('2020-01-01', periods=n_obs, freq='W')
})

# Ensure freq_max > freq_min
test_df['freq_max'] = test_df['freq_min'] + np.random.uniform(5, 20, n_obs)

print(f"✓ Created test DataFrame with {len(test_df)} observations")

# Get visualization service
viz = get_visualization_service()
print("✓ Got visualization service instance")

# Test each visualization type
print("\n--- Testing Visualizations ---")

# 1. Interactive Sky Plot
print("\n1. Interactive Sky Plot...")
try:
    fig = viz.create_interactive_sky_plot(test_df, color_by='Band')
    if fig:
        print(f"   ✓ Created sky plot with {len(fig.data)} traces")
    else:
        print("   ✗ Sky plot returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 2. Spectral Line Heatmap
print("\n2. Spectral Line Heatmap...")
try:
    fig = viz.create_spectral_line_heatmap(test_df, redshift=0.0)
    if fig:
        print(f"   ✓ Created heatmap")
    else:
        print("   ✗ Heatmap returned None (may need freq columns)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 3. Timeline View
print("\n3. Timeline View...")
try:
    fig = viz.create_timeline_view(test_df, color_by='Band')
    if fig:
        print(f"   ✓ Created timeline with {len(fig.data)} traces")
    else:
        print("   ✗ Timeline returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. UV Coverage
print("\n4. UV Coverage Estimator...")
try:
    fig = viz.estimate_uv_coverage(declination_deg=-23.0, config="C43-5")
    if fig:
        print(f"   ✓ Created UV coverage plot")
    else:
        print("   ✗ UV plot returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 5. Sensitivity Analysis
print("\n5. Sensitivity Analysis...")
try:
    fig = viz.create_sensitivity_plot(test_df)
    if fig:
        print(f"   ✓ Created sensitivity plot")
    else:
        print("   ✗ Sensitivity plot returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 6. Sensitivity Calculator
print("\n6. Sensitivity Calculator...")
try:
    result = viz.calculate_alma_sensitivity(
        frequency_ghz=230,
        integration_time_hours=1.0,
        bandwidth_ghz=7.5
    )
    print(f"   ✓ Calculated sensitivity: {result['continuum_rms_ujy']:.1f} μJy for 1 hour at 230 GHz")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Frequency Distribution
print("\n7. Frequency Distribution...")
try:
    fig = viz.create_frequency_distribution(test_df)
    if fig:
        print(f"   ✓ Created frequency distribution plot")
    else:
        print("   ✗ Frequency plot returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 8. Summary Dashboard
print("\n8. Summary Dashboard...")
try:
    fig = viz.create_summary_dashboard(test_df)
    if fig:
        print(f"   ✓ Created summary dashboard with 4 subplots")
    else:
        print("   ✗ Dashboard returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test spectral lines database
print("\n--- Spectral Lines Database ---")
print(f"✓ {len(SPECTRAL_LINES)} spectral lines available")
print(f"   CO lines: {[l for l in SPECTRAL_LINES if l.startswith('CO')]}")

# Test ALMA bands
print("\n--- ALMA Bands ---")
for band, (fmin, fmax) in ALMA_BANDS.items():
    print(f"   {band}: {fmin}-{fmax} GHz")

print("\n" + "=" * 60)
print("VISUALIZATION SERVICE TEST COMPLETE")
print("=" * 60)

# Save one plot to verify
print("\nSaving test plot to test_sky_plot.html...")
try:
    fig = viz.create_interactive_sky_plot(test_df, color_by='Band')
    if fig:
        fig.write_html("test_sky_plot.html")
        print("✓ Saved to test_sky_plot.html - open in browser to view")
except Exception as e:
    print(f"✗ Could not save: {e}")
