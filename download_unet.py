import torch
import os
from segmentation_models_pytorch import Unet

# 创建模型，使用ResNet34编码器，加载ImageNet预训练权重
model = Unet(
    encoder_name="resnet34",        # 编码器选择
    encoder_weights="imagenet",     # ImageNet预训练
    in_channels=3,                  # 输入RGB三通道
    classes=1,                      # 输出单通道二值分割
)

# 创建 model 目录
os.makedirs("model", exist_ok=True)

# 保存 state_dict
torch.save(model.state_dict(), "model/unet_fallback.pth")
print("✅ 已保存 U-Net 模型到 model/unet_fallback.pth")