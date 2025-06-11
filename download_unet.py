import torch
import os
from segmentation_models_pytorch import Unet
import urllib.request

# Create model without weights first
model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

# Download pretrained weights manually
weights_url = "https://download.pytorch.org/models/resnet34-b627a593.pth"
weights_path = "resnet34_pretrained.pth"

if not os.path.exists(weights_path):
    print("Downloading pretrained weights...")
    urllib.request.urlretrieve(weights_url, weights_path)

# Load pretrained weights to encoder
pretrained_dict = torch.load(weights_path, map_location='cpu')
model_dict = model.encoder.state_dict()

# Filter and load matching weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.encoder.load_state_dict(model_dict)

# Save the complete model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/unet_fallback.pth")
print("✅ 已保存 U-Net 模型到 model/unet_fallback.pth")

# Clean up
os.remove(weights_path)