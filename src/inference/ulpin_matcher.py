import math
import pandas as pd


# ============================================================
# MATCHING THRESHOLD
# ============================================================

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

    # --------------------------------------------------------
    # YOLO roof pixel coordinates
    # --------------------------------------------------------

    roof_x = float(roof["pixel_x"])
    roof_y = float(roof["pixel_y"])

    # --------------------------------------------------------
    # ACTUAL YOLO CONFIDENCE
    # --------------------------------------------------------

    yolo_confidence = float(
        roof.get("confidence", 0.0)
    )

    # ========================================================
    # SEARCH NEAREST SCENE 3 BUILDING
    # ========================================================

    for _, row in mapping_df.iterrows():

        scene_x = row.get(
            "3D_Scene3_Pixel_X"
        )

        scene_y = row.get(
            "3D_Scene3_Pixel_Y"
        )

        # ----------------------------------------------------
        # Skip rows without Scene 3 pixel coordinates
        # ----------------------------------------------------

        if pd.isna(scene_x) or pd.isna(scene_y):
            continue

        try:

            scene_x = float(scene_x)
            scene_y = float(scene_y)

        except (ValueError, TypeError):

            continue

        # ----------------------------------------------------
        # Calculate pixel distance
        # ----------------------------------------------------

        distance = pixel_distance(
            roof_x,
            roof_y,
            scene_x,
            scene_y
        )

        # ----------------------------------------------------
        # Keep nearest building
        # ----------------------------------------------------

        if distance < best_distance:

            best_distance = distance
            best_match = row

    # ========================================================
    # NO VALID MATCH
    # ========================================================

    if best_match is None:

        return None

    # ========================================================
    # MATCH SCORE
    # ========================================================

    match_score = max(
        0.0,
        1.0 - (
            best_distance /
            MATCH_THRESHOLD
        )
    )

    # ========================================================
    # ACCEPT / REJECT
    # ========================================================

    accepted = (
        best_distance <= MATCH_THRESHOLD
        and match_score > 0
    )

    # ========================================================
    # ULPIN
    # ========================================================

    ulpin_id = str(
        best_match["ULPIN_ID"]
    )

    # ========================================================
    # BUILDING ID
    # ========================================================

    try:

        building_id = int(
            best_match["Building_ID"]
        )

    except (ValueError, TypeError):

        building_id = None

    # ========================================================
    # FIND BUILDING IN MASTER DATASET
    # ========================================================

    building_rows = master_df[
        master_df["ULPIN_ID"].astype(str)
        == ulpin_id
    ]

    if building_rows.empty:

        return {
            "accepted": False,
            "reason": (
                "ULPIN not found in master dataset"
            ),
            "ulpin_id": ulpin_id,
            "yolo_confidence": yolo_confidence
        }

    # ========================================================
    # BUILDING DATA
    # ========================================================

    building = (
        building_rows
        .iloc[0]
        .to_dict()
    )

    # ========================================================
    # RETURN MATCH
    # ========================================================

    return {

        "accepted": bool(
            accepted
        ),

        "ulpin_id": ulpin_id,

        "building_id": building_id,

        "pixel_distance": float(
            best_distance
        ),

        "match_score": float(
            match_score
        ),

        # ====================================================
        # ACTUAL YOLO CONFIDENCE
        # ====================================================

        "yolo_confidence": float(
            yolo_confidence
        ),

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

    # ========================================================
    # PROCESS HIGHEST CONFIDENCE YOLO DETECTIONS FIRST
    # ========================================================

    roofs = sorted(
        roofs,
        key=lambda x: x.get(
            "confidence",
            0.0
        ),
        reverse=True
    )

    # ========================================================
    # MATCH EACH ROOF
    # ========================================================

    for roof in roofs:

        match = find_best_match(
            roof,
            mapping_df,
            master_df
        )

        # ----------------------------------------------------
        # No possible match
        # ----------------------------------------------------

        if match is None:
            continue

        # ----------------------------------------------------
        # Match outside threshold
        # ----------------------------------------------------

        if not match.get(
            "accepted",
            False
        ):
            continue

        ulpin_id = match["ulpin_id"]

        # ----------------------------------------------------
        # Prevent duplicate ULPIN matches
        # ----------------------------------------------------

        if ulpin_id in used_ulpins:
            continue

        used_ulpins.add(
            ulpin_id
        )

        # ====================================================
        # SAVE MATCH
        # ====================================================

        matches.append({

            "roof": roof,

            "match": match

        })

    return matches