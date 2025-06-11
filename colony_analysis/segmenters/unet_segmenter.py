import os
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
except Exception:  # pragma: no cover - optional dependency
    torch = None
    F = None
    smp = None


class UnetSegmenter:
    """Simple U-Net based segmenter used as a fallback when SAM masks fail."""

    def __init__(self, model_path: str, device: Optional[str] = None, threshold: float = 0.5) -> None:
        if torch is None or smp is None:
            raise ImportError("PyTorch and segmentation_models_pytorch are required for UnetSegmenter")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Unet model weights not found: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask for the given RGB image."""
        if image.ndim != 3:
            raise ValueError("Expected an RGB image for U-Net segmentation")
        img = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
            mask = torch.sigmoid(pred)[0, 0]
        mask_np = mask.cpu().numpy()
        return (mask_np > self.threshold).astype(np.uint8)
