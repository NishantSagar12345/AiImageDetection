import cv2
import torch
import numpy as np

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform(tensor):
    """
    Convert SigLIP transformer activations from:

        [batch_size, number_of_patches, hidden_size]

    to the spatial format required by Grad-CAM:

        [batch_size, hidden_size, grid_height, grid_width]

    For google/siglip-base-patch16-224:
        Input size:   224 x 224
        Patch size:   16 x 16
        Patch grid:   14 x 14
        Patch tokens: 196

    SigLIP uses 196 spatial patch tokens here, so no CLS token
    is removed before reconstructing the 14 x 14 grid.
    """

    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(
            "Expected transformer activation shape "
            "[batch, tokens, hidden_size], "
            f"but received {tuple(tensor.shape)}."
        )

    batch_size, number_of_tokens, hidden_size = tensor.shape

    grid_size = int(number_of_tokens ** 0.5)

    if grid_size * grid_size != number_of_tokens:
        raise ValueError(
            f"Cannot reshape {number_of_tokens} tokens "
            "into a square spatial grid."
        )

    # [B, tokens, hidden]
    #        ↓
    # [B, grid, grid, hidden]
    tensor = tensor.reshape(
        batch_size,
        grid_size,
        grid_size,
        hidden_size,
    )

    # Grad-CAM expects:
    # [B, channels, height, width]
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
    Generate a class-specific Grad-CAM visualisation for the
    SigLIP-based Real / AI-generated image classifier.

    The predicted class is explicitly used as the Grad-CAM target.

    Output:
        Coloured Grad-CAM heatmap over a grayscale version
        of the original image.
    """

    model.eval()

    # -----------------------------------------------------
    # 1. Load image
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    # -----------------------------------------------------
    # 2. Get classifier prediction
    # -----------------------------------------------------

    with torch.no_grad():

        logits = model(
            pixel_values
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    real_probability = float(
        probabilities[0].item()
    )

    ai_probability = float(
        probabilities[1].item()
    )

    prediction = (
        "AI-generated"
        if predicted_class == 1
        else "Real"
    )

    # -----------------------------------------------------
    # 3. Select final SigLIP transformer block
    # -----------------------------------------------------

    target_layers = [
        model
        .backbone
        .vision_model
        .encoder
        .layers[-1]
        .layer_norm1
    ]

    # Explicitly explain the predicted class.
    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    # -----------------------------------------------------
    # 4. Generate Grad-CAM
    # -----------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=pixel_values,
        targets=targets,
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

    # -----------------------------------------------------
    # 5. Create grayscale background
    # -----------------------------------------------------

    resized_image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    image_np = np.asarray(
        resized_image,
        dtype=np.uint8,
    )

    grayscale_background = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY,
    )

    # show_cam_on_image expects 3 channels.
    grayscale_background = cv2.cvtColor(
        grayscale_background,
        cv2.COLOR_GRAY2RGB,
    )

    grayscale_background = (
        grayscale_background.astype(
            np.float32
        )
        / 255.0
    )

    # -----------------------------------------------------
    # 6. Overlay coloured Grad-CAM on grayscale image
    # -----------------------------------------------------

    cam_image = show_cam_on_image(
        grayscale_background,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )

    # -----------------------------------------------------
    # 7. Save Grad-CAM image
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 8. Return results
    # -----------------------------------------------------

    return {
        "prediction": prediction,
        "predicted_class": predicted_class,
        "real_probability": real_probability,
        "ai_probability": ai_probability,
        "gradcam_path": save_path,
    }