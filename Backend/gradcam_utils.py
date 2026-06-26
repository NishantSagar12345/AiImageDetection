import cv2
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(image_path, model, processor, device, save_path):
    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    target_layers = [
        model.backbone.vision_model.embeddings.patch_embedding
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=pixel_values
    )[0]

    rgb_image = image.resize((224, 224))
    rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    )

    return save_path