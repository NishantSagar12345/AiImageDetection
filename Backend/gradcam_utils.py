


import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor):
    """
    Convert ViT activations from

        [B, N, C]

    to

        [B, C, H, W]
    """

    if isinstance(tensor, tuple):
        tensor = tensor[0]

    batch_size, num_tokens, channels = tensor.shape

    grid_size = int(np.sqrt(num_tokens))

    tensor = tensor.reshape(
        batch_size,
        grid_size,
        grid_size,
        channels,
    )

    tensor = tensor.permute(
        0,
        3,
        1,
        2,
    )

    return tensor


def generate_gradcam(
    image_path,
    model,
    processor,
    device,
    save_path,
):
    """
    Generate a Grad-CAM overlay using the final SigLIP transformer
    block and a grayscale background.
    """

    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    # Final transformer block
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

    grayscale_cam = np.clip(
        grayscale_cam,
        0.0,
        1.0,
    )

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    # -----------------------------
    # Grayscale background
    # -----------------------------

    gray_image = image.convert("L").resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    gray = (
        np.asarray(gray_image, dtype=np.float32)
        / 255.0
    )

    gray_rgb = np.stack(
        [gray, gray, gray],
        axis=-1,
    )

    cam_image = show_cam_on_image(
        gray_rgb,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.45,
    )

    success = cv2.imwrite(
        save_path,
        cv2.cvtColor(
            cam_image,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not success:
        raise RuntimeError(
            f"Failed to save Grad-CAM image: {save_path}"
        )

    return save_path