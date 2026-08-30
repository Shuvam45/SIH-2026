import json
import os
import shutil

# =========================
# PATHS
# =========================

BASE = "/app/data/InstanceBuilding"

TRAIN_JSON = os.path.join(
    BASE,
    "via_region_data-train.json"
)

VAL_JSON = os.path.join(
    BASE,
    "via_region_data-val.json"
)

TRAIN_IMAGES = os.path.join(BASE, "train")
VAL_IMAGES = os.path.join(BASE, "val")

OUTPUT = "/app/data/yolo"

# =========================
# CREATE DIRECTORIES
# =========================

for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(
        os.path.join(OUTPUT, folder),
        exist_ok=True
    )


# =========================
# CONVERSION FUNCTION
# =========================

def convert_dataset(json_path, image_dir, split):

    print("\n" + "=" * 60)
    print(f"Processing {split.upper()}")
    print("=" * 60)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_images = 0
    total_polygons = 0
    skipped_images = 0

    for key, item in data.items():

        filename = item["filename"]

        image_path = os.path.join(
            image_dir,
            filename
        )

        # Check image exists
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {filename}")
            skipped_images += 1
            continue

        # Image dimensions from dataset
        image_width = 1000
        image_height = 1000

        # Output paths
        output_image = os.path.join(
            OUTPUT,
            "images",
            split,
            filename
        )

        label_filename = os.path.splitext(filename)[0] + ".txt"

        output_label = os.path.join(
            OUTPUT,
            "labels",
            split,
            label_filename
        )

        # Copy image
        shutil.copy2(
            image_path,
            output_image
        )

        polygons = []

        regions = item.get("regions", {})

        for region_id, region in regions.items():

            shape = region.get("shape_attributes", {})

            if shape.get("name") != "polygon":
                continue

            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])

            if len(xs) < 3 or len(ys) < 3:
                continue

            if len(xs) != len(ys):
                print(
                    f"WARNING: coordinate mismatch "
                    f"in {filename}, region {region_id}"
                )
                continue

            # YOLO segmentation format
            points = []

            for x, y in zip(xs, ys):

                # Normalize to 0-1
                x_norm = x / image_width
                y_norm = y / image_height

                # Keep values inside valid range
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                points.extend([
                    x_norm,
                    y_norm
                ])

            # Class 0 = building/roof
            line = "0 " + " ".join(
                f"{p:.6f}" for p in points
            )

            polygons.append(line)

            total_polygons += 1

        # Write label file
        with open(
            output_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(polygons))

        total_images += 1

    print(f"Images processed: {total_images}")
    print(f"Polygons found:   {total_polygons}")
    print(f"Images skipped:   {skipped_images}")


# =========================
# RUN
# =========================

convert_dataset(
    TRAIN_JSON,
    TRAIN_IMAGES,
    "train"
)

convert_dataset(
    VAL_JSON,
    VAL_IMAGES,
    "val"
)

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)