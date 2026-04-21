import os
import random
import shutil

# Source Snow Dataset
SOURCE = r"D:\Snow.v1i.yolov8\train"

# Destination Snow Model Dataset
DEST = r"D:\FOREST_FIRE\Forest-fire-detection\FireDetectionApp\snow_model\dataset"

# Create folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DEST, "labels", split), exist_ok=True)

# Get image list
images = os.listdir(os.path.join(SOURCE, "images"))
random.shuffle(images)

split_index = int(len(images) * 0.8)

train_imgs = images[:split_index]
val_imgs = images[split_index:]

# Copy Train
for img in train_imgs:
    shutil.copy(
        os.path.join(SOURCE, "images", img),
        os.path.join(DEST, "images", "train")
    )

    label = img.replace(".jpg", ".txt")
    shutil.copy(
        os.path.join(SOURCE, "labels", label),
        os.path.join(DEST, "labels", "train")
    )

# Copy Val
for img in val_imgs:
    shutil.copy(
        os.path.join(SOURCE, "images", img),
        os.path.join(DEST, "images", "val")
    )

    label = img.replace(".jpg", ".txt")
    shutil.copy(
        os.path.join(SOURCE, "labels", label),
        os.path.join(DEST, "labels", "val")
    )

print("Snow dataset split completed!")