from fastapi import APIRouter, HTTPException
from src.data.dataset_loader import get_dataset

router = APIRouter()

@router.get("/buildings")
def get_buildings():
    df = get_dataset()
    return df.to_dict(orient="records")


@router.get("/buildings/{ulpin_id}")
def get_building_by_ulpin(ulpin_id: str):
    df = get_dataset()

    result = df[df["ULPIN_ID"].astype(str) == ulpin_id]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"ULPIN not found: {ulpin_id}"
        )

    return result.iloc[0].to_dict()