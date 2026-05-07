"""Quick smoke test for SkyView + all new archive clients."""
from integrations.skyview_client import SkyViewClient
from integrations.mast_client import MASTClient
from integrations.eso_tap_client import ESOTAPClient
from integrations.irsa_client import IRSAClient

print("=== SkyView DSS2 Test ===")
sv = SkyViewClient()
r = sv.get_image(target="M87", survey="dss2", radius_arcmin=3)
print(f"  Success: {r.get('success')}")
print(f"  Survey: {r.get('survey')}")
print(f"  Shape: {r.get('image_shape')}")
print(f"  FITS: {r.get('fits_path', 'N/A')}")
print(f"  Preview: {r.get('preview_path', 'N/A')}")

print("\n=== All imports OK ===")
print(f"  MASTClient: {MASTClient.SUPPORTED_MISSIONS}")
print(f"  ESOTAPClient: {ESOTAPClient.INSTRUMENTS[:3]}")
print(f"  IRSAClient catalogs: {list(IRSAClient.POPULAR_CATALOGS.keys())[:4]}")
print(f"  SkyViewClient surveys: {list(SkyViewClient.ALIASES.keys())[:5]}")
