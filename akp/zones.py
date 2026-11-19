"""Coarse probe boxes (WGS84 degrees) used by the setup probes.

Claims-derived and deliberately loose: they only bound metadata queries
(station counts, tile counts/sizes). The P2a prior map supersedes them for any
analysis. Anchors are approximate gazetteer positions: Denali (Mt McKinley)
~63.07 N 151.01 W, Nome ~64.50 N 165.41 W, Valdez ~61.13 N 146.35 W.
Coordinates policy is about candidate sites, not these boxes.
"""
BOXES = {
    # "between McKinley and Nome" family of claim versions
    "mckinley_nome": dict(minlatitude=61.5, maxlatitude=66.0, minlongitude=-166.5, maxlongitude=-149.0),
    # Valdez version (its own zone; mountainous, bedrock near surface)
    "valdez": dict(minlatitude=60.8, maxlatitude=61.6, minlongitude=-147.5, maxlongitude=-145.5),
}

# Claim anchors (approximate gazetteer positions; km-scale use only): (lat, lon)
ANCHORS = {
    "denali": (63.0690, -151.0063),
    "nome": (64.5011, -165.4064),
    "valdez": (61.1308, -146.3483),
}

# Size-probe area for DEM access tests: outside every box above (near Fairbanks).
# Not a calibration control area.
DEM_PROBE_BBOX = (-148.0, 64.7, -147.9, 64.8)  # (minlon, minlat, maxlon, maxlat)


def in_recording_region(lat: float, lon: float) -> bool:
    """Alaska (incl. Aleutians across 180) + NW Canada (Yukon, northern BC, western NWT).

    Lat/lon box with two carve-outs: Chukotka west of the US-Russia line (lon < -168.97, lat > 64),
    and east-longitude land north of the western Aleutians (Koryak/Kamchatka).
    simplification: hand rule, not a coastline polygon; swap in a land polygon if stations near the line matter.
    """
    if 51.0 <= lat <= 72.0 and -180.0 <= lon <= -120.0:
        return not (lon < -168.97 and lat > 64.0)
    return 51.0 <= lat < 56.0 and lon >= 170.0


def in_box(lat: float, lon: float, box: dict) -> bool:
    return (box["minlatitude"] <= lat <= box["maxlatitude"]
            and box["minlongitude"] <= lon <= box["maxlongitude"])


if __name__ == "__main__":
    assert in_recording_region(64.84, -147.72)      # Fairbanks
    assert in_recording_region(52.9, 173.2)         # Attu (Aleutians, east longitude)
    assert not in_recording_region(41.6, 88.8)      # Lop Nor
    assert not in_recording_region(64.38, -173.3)   # Provideniya, Chukotka
    assert not in_recording_region(67.73, -178.27)  # ISC "ILT" estimate, Chukotka
    assert in_recording_region(63.78, -171.74)      # Gambell, St Lawrence Island (AK)
    assert in_recording_region(65.75, -168.93)      # Little Diomede (AK)
    assert not in_recording_region(53.0, 158.6)     # Kamchatka
    lon0, lat0, lon1, lat1 = DEM_PROBE_BBOX
    assert not any(in_box(lat, lon, b) for b in BOXES.values()
                   for lat in (lat0, lat1) for lon in (lon0, lon1))
    print("akp.zones self-check OK")
