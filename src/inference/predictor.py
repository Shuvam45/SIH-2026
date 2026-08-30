from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print(f"Model loaded: {MODEL_PATH}")
print(f"Classes: {model.names}")


# ============================================================
# ROOF DETECTION
# ============================================================

def detect_roofs(image: Image.Image, confidence=0.10):

    image = image.convert("RGB")

    image_np = np.array(image)

    results = model.predict(
        source=image_np,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    masks = None

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    for i, (box, conf) in enumerate(
        zip(boxes, confidences)
    ):

        x1, y1, x2, y2 = box

        pixel_x = float((x1 + x2) / 2)
        pixel_y = float((y1 + y2) / 2)

        mask_pixels = 0

        if masks is not None and i < len(masks):

            mask = masks[i]

            mask_pixels = int(
                np.sum(mask > 0.5)
            )

        detections.append({
            "roof_id": i,
            "confidence": float(conf),
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "mask_pixels": mask_pixels,
            "box": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }
        })

    return detections