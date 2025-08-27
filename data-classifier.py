import os
import shutil
import random

# Paths
SRC_DIR = "garbage-dataset"  # original dataset
DEST_DIR = "binary-dataset"  # new dataset folder

# Define which categories count as recycling
RECYCLING_CLASSES = {"cardboard", "glass", "plastic", "metal", "paper"}
# Everything else goes to trash

# Train/test split ratio
TRAIN_RATIO = 0.8

# Create new folders
for split in ["train", "test"]:
    for category in ["recycling", "trash"]:
        os.makedirs(os.path.join(DEST_DIR, split, category), exist_ok=True)

# Walk through dataset
for category in os.listdir(SRC_DIR):
    category_path = os.path.join(SRC_DIR, category)
    if not os.path.isdir(category_path):
        continue

    # Decide if category is recycling or trash
    label = "recycling" if category.lower() in RECYCLING_CLASSES else "trash"

    # Shuffle and split
    files = os.listdir(category_path)
    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Move files
    for fname in train_files:
        src = os.path.join(category_path, fname)
        dst = os.path.join(DEST_DIR, "train", label, fname)
        shutil.copy2(src, dst)

    for fname in test_files:
        src = os.path.join(category_path, fname)
        dst = os.path.join(DEST_DIR, "test", label, fname)
        shutil.copy2(src, dst)

print("✅ Dataset reorganized into binary classification (recycling vs trash).")
