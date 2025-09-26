# 1. Import required modules
from pathlib import Path
import cv2  # OpenCV for image manipulation
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# Define paths (REPLACE THESE WITH YOUR ACTUAL PATHS)
TEST_IMAGES_PATH = Path("/Users/justinclay/PycharmProjects/Pytorch Tutorials Sep 25 2025/anomalib/examples/notebooks/01_getting_started/datasets/MVTecAD/bottle/test/broken_large")
MODEL_CKPT_PATH = "/Users/justinclay/PycharmProjects/Pytorch Tutorials Sep 25 2025/anomalib/examples/notebooks/01_getting_started/results/EfficientAd/MVTecAD/bottle/v10/weights/lightning/model.ckpt"
MODEL_PTH_PATH = "/Users/justinclay/PycharmProjects/Pytorch Tutorials Sep 25 2025/anomalib/examples/notebooks/01_getting_started/pre_trained/efficientad_pretrained_weights/pretrained_teacher_medium.pth" # <<< Change to .pth path
OUTPUT_DIR = Path("./Inference_output_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Initialize the model and engine
model = EfficientAd()

# # --- Load the .pth file and update the model's state ---
# try:
#     # Load the state dictionary from the .pth file
#     state_dict = torch.load(MODEL_PTH_PATH, map_location=torch.device('cpu'))
#
#     # Load the state dictionary into the model.
#     # We use strict=False because .pth files from pretraining might have extra keys.
#     model.load_state_dict(state_dict, strict=False)
#     print(f"Successfully loaded weights from {MODEL_PTH_PATH}")
# except FileNotFoundError:
#     print(f"[ERROR] Model file not found at: {MODEL_PTH_PATH}. Check your path.")
#     exit() # Exit the script if the model isn't found
# # --------------------------------------------------------


engine = Engine()

# 3. Prepare test data
dataset = PredictDataset(
    path=TEST_IMAGES_PATH,
    image_size=(256, 256),
)

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    # ckpt_path=MODEL_CKPT_PATH,
)

# 5. Access and Visualize the results
if predictions is not None:

    for prediction in predictions:

        image_path = prediction.image_path
        print(f'Image Path: {image_path} \n Image Path Type:{type(image_path)}')
        # Load the original image (or the resized version) for overlay
        original_image = cv2.imread(image_path[0])
        # --- FIX: Check if the image was loaded successfully ---
        if original_image is None:
            print(f"[ERROR] Could not read image: {image_path}. Skipping visualization for this file.")
            continue  # Skip to the next prediction

        anomaly_map = prediction.anomaly_map  # NumPy array of pixel scores (0 to 1)

        # --- HEATMAP VISUALIZATION STEPS ---
        # 5a. Convert anomaly map to NumPy array and move to CPU
        anomaly_map_np = anomaly_map.cpu().numpy()

        # 5b. Squeeze to ensure 2D shape (256, 256) for OpenCV
        # This removes the channel dimension (e.g., from [1, 256, 256] or [256, 256, 1])
        anomaly_map_sq = anomaly_map_np.squeeze()  # <<< FIX: Added .squeeze()

        # 5c. Convert anomaly map to 8-bit image (0-255)
        # Apply scaling and cast to unsigned 8-bit integer
        heatmap = (anomaly_map_sq * 255).astype(np.uint8)

        # 5d. Apply a color map (This now receives a proper 2D CV_8UC1 image)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


        # Ensure the heatmap is the same size as the original image
        # Note: You might need to resize the original image or heatmap_colored here
        # to ensure they match (e.g., using cv2.resize)
        # Assuming original_image is 256x256 here due to PredictDataset param:
        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, original_image.shape[:2][::-1])

        # Create the blended image (e.g., 50% image, 50% heatmap)
        blended_image = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

        # 5d. Save the heatmap and the blended image
        base_name = image_path[0]
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_heatmap.png"), heatmap_colored)
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_blended.png"), blended_image)

        print(f"Processed {base_name}. Heatmap and blended image saved to {OUTPUT_DIR}")