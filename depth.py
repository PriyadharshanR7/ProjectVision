"""
MiDaS-small monocular depth estimation.

Produces a per-frame **normalised inverse-depth** map in [0, 1] where
1 = closest and 0 = farthest.  Normalisation is min-max per frame,
so depths are relative and cannot be compared across frames.
"""

import cv2
import torch
import numpy as np

from config import MIDAS_MODEL_TYPE


class DepthEstimator:
    """Load MiDaS-small once and expose `estimate(frame) → depth_map`."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = torch.hub.load(
            "intel-isl/MiDaS", MIDAS_MODEL_TYPE, trust_repo="check"
        )
        self.model.to(self.device).eval()

        # Transform (small variant)
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo="check"
        )
        self.transform = transforms.small_transform

    # ── public API ──────────────────────────────────────────────
    def estimate(self, frame):
        """Return a float32 depth map (H×W) normalised to [0, 1]."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Per-frame min-max normalisation
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth  # 0 = far, 1 = near
