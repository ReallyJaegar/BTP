import torch
from PIL import Image
import torchvision.transforms as transforms
from ram.models import ram
from ram import inference_ram as inference

# Load RAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ram(pretrained="checkpoints/ram_swin_large_14m.pth", image_size=384, vit="swin_l")
model.eval()
model = model.to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# Load your test image
image_path = "img4.png"  # <-- put your image path here
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Inference
tags, tags_chinese = inference(image, model)

print("Predicted Tags:")
print(tags)
