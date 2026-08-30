from fastapi import APIRouter, HTTPException, UploadFile, File
from src.data.dataset_loader import get_dataset
from src.inference.predictor import detect_roofs
from src.inference.ulpin_matcher import match_roofs
from PIL import Image
import pandas as pd
import math
import io
from pathlib import Path


router = APIRouter()


# ============================================================
# JSON CLEANER
# ============================================================

def clean_for_json(data):

    if isinstance(data, dict):

        return {
            key: clean_for_json(value)
            for key, value in data.items()
        }

    if isinstance(data, list):

        return [
            clean_for_json(value)
            for value in data
        ]

    if isinstance(data, float):

        if not math.isfinite(data):
            return None

    return data


# ============================================================
# GET ALL BUILDINGS
# ============================================================

@router.get("/buildings")
def get_buildings():

    df = get_dataset()

    records = df.to_dict(
        orient="records"
    )

    return clean_for_json(records)


# ============================================================
# GET BUILDING BY ULPIN
# ============================================================

@router.get("/buildings/{ulpin_id}")
def get_building_by_ulpin(
    ulpin_id: str
):

    df = get_dataset()

    result = df[
        df["ULPIN_ID"].astype(str)
        == ulpin_id
    ]

    if result.empty:

        raise HTTPException(
            status_code=404,
            detail=f"ULPIN not found: {ulpin_id}"
        )

    record = result.iloc[0].to_dict()

    return clean_for_json(record)


# ============================================================
# PREDICT + MATCH ULPIN
# ============================================================

@router.post("/predict")
async def predict_roofs(
    file: UploadFile = File(...)
):

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        # ====================================================
        # YOLO DETECTION
        # ====================================================

        roofs = detect_roofs(image)

        if not roofs:

            return {
                "success": True,
                "detection_count": 0,
                "detections": [],
                "matches": []
            }

        # ====================================================
        # LOAD MASTER DATASET
        # ====================================================

        master_df = get_dataset()

        # ====================================================
        # LOAD PIXEL MAPPING
        # ====================================================

        project_root = Path(
            __file__
        ).resolve().parents[2]

        mapping_path = (
            project_root
            / "data"
            / "WEB_READY_ULPIN_MASTER_DATASET.csv"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # If your Scene3 mapping is stored in another CSV,
        # change mapping_path to that file.
        # ----------------------------------------------------

        if not mapping_path.exists():

            raise FileNotFoundError(
                f"Mapping dataset not found: "
                f"{mapping_path}"
            )

        mapping_df = pd.read_csv(
            mapping_path
        )

        # ====================================================
        # MATCH YOLO ROOFS TO ULPINS
        # ====================================================

        matches = match_roofs(
            roofs,
            mapping_df,
            master_df
        )

        # ====================================================
        # BUILD FINAL RESPONSE
        # ====================================================

        results = []

        for item in matches:

            roof = item["roof"]

            match = item["match"]

            building = match["building"].copy()

            # ------------------------------------------------
            # Add actual YOLO information
            # ------------------------------------------------

            building["2D_Roof_ID"] = roof["roof_id"]

            building["YOLO_Confidence"] = (
                roof["confidence"]
            )

            building["Roof_Pixel_X"] = (
                roof["pixel_x"]
            )

            building["Roof_Pixel_Y"] = (
                roof["pixel_y"]
            )

            building["Matched_ULPIN_ID"] = (
                match["ulpin_id"]
            )

            building["Matched_Building_ID"] = (
                match["building_id"]
            )

            building["Pixel_Distance"] = (
                match["pixel_distance"]
            )

            building["Match_Score"] = (
                match["match_score"]
            )

            building["Match_Accepted"] = True

            building["Roof_Detected_2D"] = True

            building["Data_Status"] = (
                "2D_3D_matched"
            )

            results.append(
                clean_for_json(building)
            )

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "success": True,

            "detection_count": len(
                roofs
            ),

            "matched_count": len(
                results
            ),

            "detections": clean_for_json(
                roofs
            ),

            "matches": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )