from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = "/app/data/InstanceBuilding/train"
output_path = "/app/previews"

os.makedirs(output_path, exist_ok=True)

files = [
    f for f in os.listdir(image_path)
    if f.lower().endswith(".png")
]

if not files:
    print("ERROR: No PNG files found!")
    exit()

file = files[0]
full_path = os.path.join(image_path, file)

img = Image.open(full_path)
arr = np.array(img)

rgb = arr[:, :, :3]
alpha = arr[:, :, 3]

print("=" * 50)
print("INSTANCEBUILDING IMAGE INSPECTION")
print("=" * 50)

print("Image:", file)
print("Size:", img.size)
print("Mode:", img.mode)
print("Shape:", arr.shape)

print("\n4TH CHANNEL INFORMATION")
print("Min:", alpha.min())
print("Max:", alpha.max())
print("Mean:", alpha.mean())
print("Unique values:", len(np.unique(alpha)))

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")
plt.savefig(
    os.path.join(output_path, "rgb_preview.png"),
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(8, 8))
plt.imshow(alpha, cmap="gray")
plt.colorbar(label="4th channel value")
plt.title("Fourth Channel / Height Candidate")
plt.axis("off")
plt.savefig(
    os.path.join(output_path, "height_preview.png"),
    bbox_inches="tight"
)
plt.close()

print("\nSaved previews:")
print("/app/previews/rgb_preview.png")
print("/app/previews/height_preview.png")

print("=" * 50)