from astropy.coordinates import SkyCoord
from astroquery.alma import Alma
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

target = "Sz65"
print(f"Testing resolution for: {target}")

# 1. Try SkyCoord (SIMBAD/SESAME)
try:
    coord = SkyCoord.from_name(target)
    print(f"SUCCESS: SkyCoord resolved {target} to {coord}")
except Exception as e:
    print(f"FAILURE: SkyCoord could not resolve {target}: {e}")

# 2. Try ALMA Query
try:
    print("Attempting ALMA query...")
    results = Alma.query_object(target, public=True, science=True)
    print(f"SUCCESS: ALMA found {len(results)} results for {target}")
except Exception as e:
    print(f"FAILURE: ALMA query failed for {target}: {e}")
