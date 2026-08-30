from fastapi import APIRouter, HTTPException
from src.data.dataset_loader import get_dataset
import math

router = APIRouter()


def clean_for_json(data):
    """Convert NaN and infinite values into JSON-safe None."""

    if isinstance(data, dict):
        return {
            key: clean_for_json(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [clean_for_json(value) for value in data]

    if isinstance(data, float) and not math.isfinite(data):
        return None

    return data


@router.get("/buildings")
def get_buildings():
    df = get_dataset()

    records = df.to_dict(orient="records")

    return clean_for_json(records)


@router.get("/buildings/{ulpin_id}")
def get_building_by_ulpin(ulpin_id: str):

    df = get_dataset()

    result = df[
        df["ULPIN_ID"].astype(str) == ulpin_id
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"ULPIN not found: {ulpin_id}"
        )

    record = result.iloc[0].to_dict()

    return clean_for_json(record)