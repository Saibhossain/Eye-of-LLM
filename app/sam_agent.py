import torch
import numpy as np
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMAgent:
    def __init__(self, checkpoint_path="./sam2/checkpoints/sam2_hiera_tiny.pt", config="sam2_hiera_t.yaml"):
        # On M1, CPU is safer for SAM 2 custom kernels than MPS currently
        self.device = "cpu"
        print(f"Loading SAM 2 on {self.device} (M1 Optimized)...")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Did you run download_ckpts.sh?")

        self.sam2_model = build_sam2(config, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print("SAM 2 Loaded successfully.")

    def segment_object(self, frame, x, y):
        """
        1. Takes a frame and a point (x, y).
        2. Predicts the mask for the object at that point.
        3. Returns the object image (background blacked out).
        """
        # Convert BGR (OpenCV) to RGB (SAM 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)

        # Prompt with the specific point
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 means 'foreground'

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Use the best mask
        best_mask = masks[0]

        # Apply mask to create the "cutout"
        # Convert mask to uint8 (0 or 255)
        mask_uint8 = (best_mask * 255).astype(np.uint8)

        # Bitwise AND to blackout background
        segmented_image = cv2.bitwise_and(frame, frame, mask=mask_uint8)

        return segmented_image, best_mask