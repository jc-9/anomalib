import numpy as np
import torch
import torchvision.transforms as T_vision
from anomalib.data.dataclasses.torch import ImageBatch, ImageItem  # Need ImageItem for intermediate type hints
from pathlib import Path
import cv2  # Used for loading images from disk

# --- Configuration (UPDATE THESE) ---
# Replace this with the path to your directory containing test images
IMAGE_DIR = Path("./dataset/train/good")
BATCH_SIZE = 8
IMAGE_SIZE = 224


def denormalize_tensor(tensor):
    """Denormalizes a single image tensor (C, H, W) and converts it to HWC, uint8 NumPy array."""
    # Standard ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 1. Denormalize: Reverse the normalization
    denorm_tensor = tensor * std + mean

    # 2. Convert to NumPy, scale to 0-255, and convert to 8-bit integer
    image_np = (denorm_tensor.clip(0, 1) * 255).to(torch.uint8).cpu().numpy()

    # 3. Permute dimensions: CHW -> HWC
    image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, C)

    return image_np

# --- 1. Define Transforms (from previous step) ---
transform = T_vision.Compose([
    T_vision.ToTensor(),  # Converts HWC uint8 NumPy array -> CHW float32 Tensor [0, 1]
    T_vision.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    T_vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. Load and Transform Individual Images ---
image_tensors = []
image_paths = []
label_list = []

# Safely get the first BATCH_SIZE images (e.g., .png, .jpg)
# We assume they are all 'normal' (label 0) for this loading example
for i, path in enumerate(list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))):
    if i >= BATCH_SIZE:
        break

    # Load image using OpenCV (returns HWC NumPy array)
    image_np = cv2.imread(str(path))

    if image_np is None:
        print(f"Warning: Could not read image at {path}. Skipping.")
        continue

    # OpenCV loads images in BGR; convert to RGB (required for standard normalization)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Apply the transforms
    image_tensor = transform(image_np)

    image_tensors.append(image_tensor)
    image_paths.append(path)
    label_list.append(0)  # All are assigned a 'normal' label (0) for training

if not image_tensors:
    raise FileNotFoundError(f"No images found in the directory: {IMAGE_DIR}. Check your path.")

# --- 3. Stack Tensors into a Batch ---
# This creates the final [B, C, H, W] tensor
image_batch_tensor = torch.stack(image_tensors)
label_tensor = torch.tensor(label_list, dtype=torch.long)

# --- 4. Instantiate the ImageBatch ---
batch = ImageBatch(
    image=image_batch_tensor,
    gt_label=label_tensor,
    image_path=image_paths,  # Path is a list of Path objects
)

print(f"\n--- ImageBatch Loaded from Disk (Size {batch.batch_size}) ---")
print(f"Batch Image Shape: {tuple(batch.image.shape)}")

# --- 5. Verify by Iterating (Same as previous tutorial) ---
print("\n--- Verifying Item Shapes After Loading ---")
for i, item in enumerate(batch):
    print(f"Item {i}: Image Shape {tuple(item.image.shape)}, Path: {item.image_path}")