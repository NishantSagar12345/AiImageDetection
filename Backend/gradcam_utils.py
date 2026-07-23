import cv2
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    image_path,
    model,
    processor,
    device,
    save_path,
    mask_save_path,
    mask_threshold=0.70,
):
    """
    Generate:

    1. A coloured Grad-CAM overlay for the frontend.
    2. A binary attention mask for the LLM.

    White mask regions represent the strongest activations.
    Black mask regions should be ignored.
    """

    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    target_layers = [
        model.backbone.vision_model.embeddings.patch_embedding
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    grayscale_cam = cam(
        input_tensor=pixel_values,
    )[0]

    # Ensure Grad-CAM values stay between 0 and 1.
    grayscale_cam = np.clip(
        grayscale_cam,
        0.0,
        1.0,
    )

    # -----------------------------------------------------
    # Coloured Grad-CAM overlay for the frontend
    # -----------------------------------------------------

    rgb_image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    rgb_image = (
        np.array(rgb_image).astype(np.float32) / 255.0
    )

    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )

    gradcam_saved = cv2.imwrite(
        save_path,
        cv2.cvtColor(
            cam_image,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not gradcam_saved:
        raise RuntimeError(
            f"Failed to save Grad-CAM image: {save_path}"
        )

    # -----------------------------------------------------
    # Binary attention mask for the LLM
    # -----------------------------------------------------

    binary_mask = np.where(
        grayscale_cam >= mask_threshold,
        255,
        0,
    ).astype(np.uint8)

    # Remove tiny isolated white pixels.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Fill small gaps inside strong highlighted regions.
    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask_saved = cv2.imwrite(
        mask_save_path,
        binary_mask,
    )

    if not mask_saved:
        raise RuntimeError(
            f"Failed to save attention mask: {mask_save_path}"
        )

    return {
        "gradcam_path": save_path,
        "attention_mask_path": mask_save_path,
    }