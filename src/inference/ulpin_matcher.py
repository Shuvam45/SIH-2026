import math
import pandas as pd


MATCH_THRESHOLD = 100.0


# ============================================================
# PIXEL DISTANCE
# ============================================================

def pixel_distance(x1, y1, x2, y2):

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


# ============================================================
# MATCH ONE ROOF TO BUILDING
# ============================================================

def find_best_match(
    roof,
    mapping_df,
    master_df
):

    if mapping_df.empty:
        return None

    best_match = None
    best_distance = float("inf")

    roof_x = roof["pixel_x"]
    roof_y = roof["pixel_y"]

    for _, row in mapping_df.iterrows():

        distance = pixel_distance(
            roof_x,
            roof_y,
            float(row["Scene3_Pixel_X"]),
            float(row["Scene3_Pixel_Y"])
        )

        if distance < best_distance:

            best_distance = distance
            best_match = row

    if best_match is None:
        return None

    # Same formula used in your Step 66
    match_score = max(
        0.0,
        1.0 - (best_distance / MATCH_THRESHOLD)
    )

    accepted = (
        best_distance <= MATCH_THRESHOLD
        and match_score > 0
    )

    ulpin_id = best_match["ULPIN_ID"]

    building_rows = master_df[
        master_df["ULPIN_ID"].astype(str)
        == str(ulpin_id)
    ]

    if building_rows.empty:
        return {
            "accepted": False,
            "reason": "ULPIN not found in master dataset"
        }

    building = building_rows.iloc[0].to_dict()

    return {
        "accepted": bool(accepted),
        "ulpin_id": str(ulpin_id),
        "building_id": int(best_match["Building_ID"]),
        "pixel_distance": float(best_distance),
        "match_score": float(match_score),
        "building": building
    }


# ============================================================
# MATCH ALL ROOFS
# ============================================================

def match_roofs(
    roofs,
    mapping_df,
    master_df
):

    matches = []

    used_ulpins = set()

    # Process strongest detections first
    roofs = sorted(
        roofs,
        key=lambda x: x["confidence"],
        reverse=True
    )

    for roof in roofs:

        match = find_best_match(
            roof,
            mapping_df,
            master_df
        )

        if match is None:
            continue

        if not match.get("accepted", False):
            continue

        ulpin_id = match["ulpin_id"]

        # Prevent duplicate building matches
        if ulpin_id in used_ulpins:
            continue

        used_ulpins.add(ulpin_id)

        matches.append({
            "roof": roof,
            "match": match
        })

    return matches