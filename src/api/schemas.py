from pydantic import BaseModel
from typing import Optional


class BuildingResponse(BaseModel):
    ULPIN_ID: str
    Scene_ID: int
    Building_ID: int

    Footprint_Area_m2: Optional[float] = None
    Building_Height_m: Optional[float] = None
    Building_Volume_m3: Optional[float] = None

    Roof_Area_m2: Optional[float] = None
    Usable_Roof_Area_m2: Optional[float] = None
    Estimated_Solar_Capacity_kW: Optional[float] = None

    Building_Type: Optional[str] = None
    Estimated_Floors: Optional[int] = None

    Roof_Detected_2D: Optional[bool] = None
    Match_Accepted: Optional[bool] = None