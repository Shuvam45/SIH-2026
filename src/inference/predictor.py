from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

print(f"Looking for model at: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print(f"Model loaded successfully: {MODEL_PATH}")
print(f"Classes: {model.names}")
print("=" * 60)


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

    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if result.boxes is None or len(result.boxes) == 0:

        print("=" * 60)
        print("YOLO PREDICTIONS")
        print("=" * 60)
        print("No roofs detected.")
        print("=" * 60)

        return detections

    # ========================================================
    # GET YOLO DATA
    # ========================================================

    boxes = result.boxes.xyxy.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()

    masks = None

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    # ========================================================
    # PRINT PREDICTIONS
    # ========================================================

    print("=" * 60)
    print("YOLO PREDICTIONS")
    print("=" * 60)

    for i, (box, conf) in enumerate(
        zip(boxes, confidences)
    ):

        x1, y1, x2, y2 = box

        confidence_score = float(conf)

        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        pixel_x = float((x1 + x2) / 2)
        pixel_y = float((y1 + y2) / 2)

        # ----------------------------------------------------
        # MASK PIXELS
        # ----------------------------------------------------

        mask_pixels = 0

        if masks is not None and i < len(masks):

            mask = masks[i]

            mask_pixels = int(
                np.sum(mask > 0.5)
            )

        # ====================================================
        # SERVER OUTPUT
        # ====================================================

        print(
            f"Roof {i}: "
            f"confidence = {confidence_score:.4f} "
            f"({confidence_score * 100:.2f}%)"
        )

        print(
            f"    Center: "
            f"({pixel_x:.2f}, {pixel_y:.2f})"
        )

        print(
            f"    Mask pixels: {mask_pixels}"
        )

        print(
            f"    Box: "
            f"({x1:.2f}, {y1:.2f}, "
            f"{x2:.2f}, {y2:.2f})"
        )

        # ====================================================
        # STORE DETECTION
        # ====================================================

        detections.append({
            "roof_id": i,

            # ACTUAL YOLO CONFIDENCE
            "confidence": confidence_score,

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

    # ========================================================
    # SUMMARY
    # ========================================================

    scores = [
        detection["confidence"]
        for detection in detections
    ]

    if scores:

        average_confidence = sum(scores) / len(scores)

        highest_confidence = max(scores)

        lowest_confidence = min(scores)

        print("=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)

        print(
            f"Total roofs detected: "
            f"{len(detections)}"
        )

        print(
            f"Average confidence: "
            f"{average_confidence:.4f} "
            f"({average_confidence * 100:.2f}%)"
        )

        print(
            f"Highest confidence: "
            f"{highest_confidence:.4f} "
            f"({highest_confidence * 100:.2f}%)"
        )

        print(
            f"Lowest confidence: "
            f"{lowest_confidence:.4f} "
            f"({lowest_confidence * 100:.2f}%)"
        )

        print("=" * 60)

    return detections