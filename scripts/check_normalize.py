from src.cvpipe.normalize import plan_normalization

cases = [
    ({"field_order": "progressive", "is_vfr": False}, []),
    ({"field_order": "unknown",     "is_vfr": False}, []),
    ({"field_order": "tt",          "is_vfr": False}, ["deinterlace"]),
    ({"field_order": "bb",          "is_vfr": True},  ["deinterlace", "constant_fps"]),
    ({"field_order": "progressive", "is_vfr": True},  ["constant_fps"]),
]
for meta, expected in cases:
    got = plan_normalization(meta)
    print("PASS" if got == expected else "FAIL", meta, "->", got)
