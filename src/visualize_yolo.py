from PIL import Image, ImageDraw
import os
import random

IMAGE_DIR = "/app/data/yolo/images/train"
LABEL_DIR = "/app/data/yolo/labels/train"
OUTPUT_DIR = "/app/verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
]

# Select 10 random images
random.seed(42)
selected = random.sample(files, min(10, len(files)))

for filename in selected:

    image_path = os.path.join(IMAGE_DIR, filename)

    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_filename)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) < 7:
                continue

            # First value = class
            coords = list(map(float, values[1:]))

            points = []

            for i in range(0, len(coords), 2):

                x = int(coords[i] * width)
                y = int(coords[i + 1] * height)

                points.append((x, y))

            if len(points) >= 3:

                draw.polygon(
                    points,
                    outline="red",
                    width=3
                )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    image.save(output_path)

    print("Saved:", output_path)

print("\nVerification images created.")