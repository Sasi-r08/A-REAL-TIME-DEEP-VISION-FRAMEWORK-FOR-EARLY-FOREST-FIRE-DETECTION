import os
import random
import shutil

# ==========================
# SOURCE & DEST PATHS
# ==========================

SOURCE = r"D:\fog_dataset\fog images"
DEST = r"D:\FOREST_FIRE\Forest-fire-detection\FireDetectionApp\fog_model\dataset"

TRAIN_RATIO = 0.8

# ==========================
# CREATE FOLDER STRUCTURE
# ==========================

for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DEST, "labels", split), exist_ok=True)

# ==========================
# GET IMAGE LIST
# ==========================

images = [f for f in os.listdir(SOURCE) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

split_index = int(len(images) * TRAIN_RATIO)

train_images = images[:split_index]
val_images = images[split_index:]

# ==========================
# COPY + CREATE LABELS
# ==========================

def process_images(image_list, split_name):
    for img in image_list:
        src_path = os.path.join(SOURCE, img)
        dest_img_path = os.path.join(DEST, "images", split_name, img)

        shutil.copy(src_path, dest_img_path)

        # Create YOLO label file (class_id center_x center_y width height)
        label_name = os.path.splitext(img)[0] + ".txt"
        dest_label_path = os.path.join(DEST, "labels", split_name, label_name)

        with open(dest_label_path, "w") as f:
            f.write("0 0.5 0.5 1.0 1.0")  # Full image bounding box for fog

process_images(train_images, "train")
process_images(val_images, "val")

print("✅ Fog dataset prepared successfully!")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")