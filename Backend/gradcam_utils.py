import cv2
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor):
    """
    Convert SigLIP transformer activations from:

        [batch_size, number_of_patches, hidden_size]

    to the spatial format required by Grad-CAM:

        [batch_size, hidden_size, grid_height, grid_width]

    For google/siglip-base-patch16-224:
        224 / 16 = 14
        14 × 14 = 196 patch tokens
    """

    if isinstance(tensor, tuple):
        tensor = tensor[0]

    batch_size, number_of_tokens, hidden_size = tensor.shape

    grid_size = int(number_of_tokens**0.5)

    if grid_size * grid_size != number_of_tokens:
        raise ValueError(
            f"Cannot reshape {number_of_tokens} tokens into a square grid."
        )

    tensor = tensor.reshape(
        batch_size,
        grid_size,
        grid_size,
        hidden_size,
    )

    tensor = tensor.permute(0, 3, 1, 2)

    return tensor


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

    # Use LayerNorm from the final SigLIP transformer block.
    target_layers = [
        model.backbone.vision_model.encoder.layers[-1].layer_norm1
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=pixel_values,
    )[0]

    # Keep Grad-CAM values within the expected range.
    grayscale_cam = np.clip(
        grayscale_cam,
        0.0,
        1.0,
    )

    # Ensure the CAM matches the displayed image dimensions.
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    # -----------------------------------------------------
    # Coloured Grad-CAM overlay for the frontend
    # -----------------------------------------------------

    rgb_image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    rgb_image = (
        np.asarray(rgb_image, dtype=np.float32) / 255.0
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

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    # Remove small isolated activations.
    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Fill small gaps within strongly activated areas.
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