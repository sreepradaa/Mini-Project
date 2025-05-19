import os
import shutil
import random

# Paths
image_dir = "C:/MiniP/all-images"
train_dir = "C:/MiniP/dataset/images/train"
val_dir = "C:/MiniP/dataset/images/val"
train_labels_dir = "C:/MiniP/dataset/labels/train"
val_labels_dir = "C:/MiniP/dataset/labels/val"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of images
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)  # Shuffle to ensure random split

# Split into 80% train, 20% val
train_split = int(0.8 * len(images))
train_images = images[:train_split]
val_images = images[train_split:]

# Copy training images and labels
for img in train_images:
    shutil.copy(os.path.join(image_dir, img), train_dir)
    label = img.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(image_dir, label)
    if os.path.exists(label_path):
        shutil.copy(label_path, train_labels_dir)
    else:
        print(f"Warning: Label file {label} not found for image {img}")

# Copy validation images and labels
for img in val_images:
    shutil.copy(os.path.join(image_dir, img), val_dir)
    label = img.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(image_dir, label)
    if os.path.exists(label_path):
        shutil.copy(label_path, val_labels_dir)
    else:
        print(f"Warning: Label file {label} not found for image {img}")

print("Dataset split complete!")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
