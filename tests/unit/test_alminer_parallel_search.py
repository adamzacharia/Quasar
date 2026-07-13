import pandas as pd

from integrations.alminer_client import ALminerClient, _CANONICAL_ALMA_COLUMNS


def test_standardize_columns_canonical_on_tap_shape():
    client = ALminerClient()
    tap_df = pd.DataFrame({"s_ra": [10.0], "s_dec": [20.0],
                           "member_ous_uid": ["uid://A001/X1/1"]})
    out = client._standardize_columns(tap_df)
    for col in _CANONICAL_ALMA_COLUMNS:
        assert col in out.columns, f"missing {col}"


def test_standardize_columns_canonical_on_alminer_shape():
    client = ALminerClient()
    alminer_df = pd.DataFrame({"ra": [10.0], "dec": [20.0], "band_number": [6],
                               "min_freq_ghz": [230.0], "max_freq_ghz": [232.0]})
    out = client._standardize_columns(alminer_df)
    for col in _CANONICAL_ALMA_COLUMNS:
        assert col in out.columns, f"missing {col}"


def test_both_shapes_share_the_canonical_core():
    client = ALminerClient()
    tap_out = client._standardize_columns(pd.DataFrame({"s_ra": [1.0], "s_dec": [2.0]}))
    alminer_out = client._standardize_columns(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))
    canonical = set(_CANONICAL_ALMA_COLUMNS)
    assert canonical.issubset(set(tap_out.columns))
    assert canonical.issubset(set(alminer_out.columns))
