import shutil
from pathlib import Path
import cv2

# Input and output paths
input_dir = Path(r"V:\model\dataset\val")
output_dir = Path(r"V:\model\dataset\val")

# Create required folders
(output_dir / "images").mkdir(parents=True, exist_ok=True)
(output_dir / "depth").mkdir(parents=True, exist_ok=True)
(output_dir / "labels").mkdir(parents=True, exist_ok=True)

# Get RGB files (support jpg, jpeg, png)
rgb_files = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    rgb_files.extend((input_dir / "images").glob(ext))
rgb_files = sorted(rgb_files)

# Depth and labels (assume png for depth)
depth_files = sorted((input_dir / "depth").glob("*.png"))
label_files = sorted((input_dir / "labels").glob("*.txt"))

print(f"Found {len(rgb_files)} RGB, {len(depth_files)} Depth, {len(label_files)} Labels")

# Process files (index-based matching)
for i, (rgb_file, depth_file, label_file) in enumerate(zip(rgb_files, depth_files, label_files)):
    stem = f"{i:05d}"  # new uniform name like 00000.png

    # Convert RGB to PNG and save
    rgb_image = cv2.imread(str(rgb_file))
    cv2.imwrite(str(output_dir / "images" / f"{stem}.png"), rgb_image)

    # Copy depth (already png)
    shutil.copy(depth_file, output_dir / "depth" / f"{stem}.png")

    # Copy label
    shutil.copy(label_file, output_dir / "labels" / f"{stem}.txt")

print("✅ Dataset converted successfully: all RGBs and Depth images saved as PNGs")
