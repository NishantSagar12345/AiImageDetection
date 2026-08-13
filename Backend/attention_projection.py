import os

import cv2
import numpy as np
import torch
from PIL import Image


def get_target_encoder(model):
    """
    Locates the Vision Transformer encoder inside common model structures.

    Supported structures:
    - model.vision_model
    - model.backbone.vision_model
    - model.backbone
    - model itself
    """
    if hasattr(model, "vision_model"):
        return model.vision_model

    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "vision_model"):
            return model.backbone.vision_model

        return model.backbone

    return model


def infer_patch_grid(pixel_values, target_encoder, token_count):
    """
    Determines the spatial patch grid used by the Vision Transformer.

    Example:
        Input resolution: 224 x 224
        Patch size:       16 x 16
        Patch grid:       14 x 14
        Spatial tokens:   196

    Returns:
        grid_height
        grid_width
        number_of_special_tokens
    """
    image_height = pixel_values.shape[-2]
    image_width = pixel_values.shape[-1]

    config = getattr(target_encoder, "config", None)
    patch_size = getattr(config, "patch_size", None)

    if isinstance(patch_size, int):
        grid_height = image_height // patch_size
        grid_width = image_width // patch_size

    elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        grid_height = image_height // patch_size[0]
        grid_width = image_width // patch_size[1]

    else:
        # Fallback for models whose configuration does not expose patch_size.
        # Try common counts of leading special tokens.
        for special_tokens in (0, 1, 2, 4, 8):
            spatial_tokens = token_count - special_tokens
            grid_size = int(np.sqrt(spatial_tokens))

            if grid_size * grid_size == spatial_tokens:
                return grid_size, grid_size, special_tokens

        raise ValueError(
            f"Unable to infer the spatial patch grid from {token_count} tokens."
        )

    spatial_token_count = grid_height * grid_width
    number_of_special_tokens = token_count - spatial_token_count

    if number_of_special_tokens < 0:
        raise ValueError(
            f"The attention tensor contains {token_count} tokens, but the "
            f"inferred patch grid requires {spatial_token_count} tokens."
        )

    return grid_height, grid_width, number_of_special_tokens


def compute_projection_attention(
    attentions,
    grid_height,
    grid_width,
    number_of_special_tokens=0,
):
    """
    Computes Max-Fusion Terminal-Layer Attention Projection.

    Method:
    1. Select the final transformer attention layer.
    2. Apply maximum aggregation across attention heads.
    3. Apply maximum projection across query tokens.
    4. Remove non-spatial tokens such as CLS or register tokens.
    5. Reconstruct the remaining patch scores into the spatial grid.

    This produces a final-layer attention visualisation. It does not perform
    standard multi-layer Attention Rollout because attention matrices are not
    recursively multiplied across transformer layers.

    Args:
        attentions:
            Tuple of attention tensors. Each tensor normally has shape:
            [batch, heads, query_tokens, key_tokens].

        grid_height:
            Number of patch rows.

        grid_width:
            Number of patch columns.

        number_of_special_tokens:
            Number of leading non-spatial tokens to remove.

    Returns:
        Normalised NumPy attention map with shape:
        [grid_height, grid_width].
    """
    if attentions is None or len(attentions) == 0:
        raise ValueError(
            "The encoder did not return attention tensors. Ensure that "
            "output_attentions=True is enabled."
        )

    final_layer = attentions[-1]

    if final_layer.ndim != 4:
        raise ValueError(
            "Expected attention shape "
            "[batch, heads, query_tokens, key_tokens], "
            f"but received {tuple(final_layer.shape)}."
        )

    # Use the first image in the batch.
    # Shape: [heads, query_tokens, key_tokens]
    final_layer = final_layer[0]

    # Maximum fusion across attention heads.
    # Shape: [query_tokens, key_tokens]
    max_head_attention = torch.max(
        final_layer,
        dim=0,
    ).values

    # Maximum projection across query tokens.
    # Each value represents the strongest attention received by a key token
    # from any query token in the final transformer layer.
    # Shape: [key_tokens]
    token_scores = torch.max(
        max_head_attention,
        dim=0,
    ).values

    spatial_token_count = grid_height * grid_width

    if number_of_special_tokens > 0:
        token_scores = token_scores[number_of_special_tokens:]

    if token_scores.numel() != spatial_token_count:
        raise ValueError(
            f"Expected {spatial_token_count} spatial token scores, "
            f"but received {token_scores.numel()} after removing "
            f"{number_of_special_tokens} special tokens."
        )

    attention_map = token_scores.reshape(
        grid_height,
        grid_width,
    )

    # Local min-max normalisation.
    minimum = attention_map.min()
    maximum = attention_map.max()

    attention_map = (
        attention_map - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return attention_map.detach().float().cpu().numpy()


def create_attention_overlay(
    image,
    attention_map,
    output_size=(224, 224),
    background_weight=0.40,
    heatmap_weight=0.60,
):
    """
    Creates a JET heatmap over a grayscale version of the input image.

    Args:
        image:
            PIL RGB image.

        attention_map:
            Normalised 2D attention map in the range [0, 1].

        output_size:
            Output dimensions as (width, height).

        background_weight:
            Weight applied to the grayscale image.

        heatmap_weight:
            Weight applied to the coloured attention heatmap.

    Returns:
        BGR NumPy image suitable for cv2.imwrite().
    """
    output_width, output_height = output_size

    attention_uint8 = np.uint8(
        np.clip(attention_map, 0.0, 1.0) * 255
    )

    resized_attention = cv2.resize(
        attention_uint8,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = cv2.applyColorMap(
        resized_attention,
        cv2.COLORMAP_JET,
    )

    image_rgb = np.array(
        image.resize(
            (output_width, output_height),
            Image.Resampling.LANCZOS,
        )
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    grayscale_background = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    grayscale_background = cv2.cvtColor(
        grayscale_background,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = cv2.addWeighted(
        grayscale_background,
        background_weight,
        heatmap,
        heatmap_weight,
        0,
    )

    return overlay


def create_attention_masked_image(
    image,
    attention_map,
    top_percentage=0.20,
):
    """
    Masks the most highly attended regions identified by the explanation.

    The selected regions are blurred rather than blacked out. This reduces
    the risk that the perturbation introduces artificial black-border evidence.

    Args:
        image:
            Original PIL RGB image.

        attention_map:
            Normalised patch-level attention map.

        top_percentage:
            Fraction of highest-attention pixels to perturb.
            For example, 0.20 masks the strongest 20%.

    Returns:
        PIL image with highlighted regions blurred.
    """
    if not 0.0 < top_percentage < 1.0:
        raise ValueError(
            "top_percentage must be between 0 and 1."
        )

    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]

    resized_attention = cv2.resize(
        attention_map.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    threshold = np.quantile(
        resized_attention,
        1.0 - top_percentage,
    )

    important_region_mask = resized_attention >= threshold

    blurred_image = cv2.GaussianBlur(
        image_np,
        (31, 31),
        0,
    )

    masked_image = image_np.copy()

    masked_image[important_region_mask] = blurred_image[
        important_region_mask
    ]

    return Image.fromarray(masked_image)


def extract_logits(model_output):
    """
    Extracts classifier logits from common PyTorch and Hugging Face outputs.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    logits = getattr(model_output, "logits", None)

    if logits is not None:
        return logits

    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]

    if isinstance(model_output, (tuple, list)):
        if len(model_output) > 0 and isinstance(
            model_output[0],
            torch.Tensor,
        ):
            return model_output[0]

    raise ValueError(
        "Could not extract classification logits from the model output."
    )


def get_class_probabilities(logits):
    """
    Converts binary or multiclass logits into probabilities.

    Supported classifier outputs:
    - One logit: sigmoid binary classifier
    - Two logits: softmax binary classifier
    - More than two logits: softmax multiclass classifier
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.shape[-1] == 1:
        positive_probability = torch.sigmoid(logits[0, 0])

        probabilities = torch.stack(
            [
                1.0 - positive_probability,
                positive_probability,
            ]
        )

        return probabilities

    return torch.softmax(
        logits[0],
        dim=-1,
    )


def run_classifier(model, pixel_values):
    """
    Executes the complete classifier.

    Some custom models accept pixel_values as a keyword argument, while
    others accept the image tensor positionally.
    """
    try:
        output = model(pixel_values=pixel_values)
    except TypeError:
        output = model(pixel_values)

    logits = extract_logits(output)
    probabilities = get_class_probabilities(logits)

    return logits, probabilities


def verify_attention_impact(
    image,
    attention_map,
    model,
    processor,
    device,
    top_percentage=0.20,
):
    """
    Tests whether the regions highlighted by the attention map influence
    the complete classifier prediction.

    Validation procedure:
    1. Run the original image through the complete classifier.
    2. Identify the top-attention regions.
    3. Blur those regions.
    4. Run the perturbed image through the complete classifier.
    5. Measure the change in confidence for the original predicted class.

    Returns:
        Dictionary containing prediction and confidence information.
    """
    original_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    original_pixel_values = original_inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        _, original_probabilities = run_classifier(
            model,
            original_pixel_values,
        )

    predicted_class = int(
        torch.argmax(original_probabilities).item()
    )

    original_confidence = float(
        original_probabilities[predicted_class].item()
    )

    masked_image = create_attention_masked_image(
        image=image,
        attention_map=attention_map,
        top_percentage=top_percentage,
    )

    masked_inputs = processor(
        images=masked_image,
        return_tensors="pt",
    )

    masked_pixel_values = masked_inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        _, masked_probabilities = run_classifier(
            model,
            masked_pixel_values,
        )

    masked_confidence = float(
        masked_probabilities[predicted_class].item()
    )

    confidence_drop = (
        original_confidence - masked_confidence
    )

    result = {
        "predicted_class": predicted_class,
        "original_confidence": original_confidence,
        "masked_confidence": masked_confidence,
        "confidence_drop": confidence_drop,
        "masked_image": masked_image,
    }

    print("\n" + "=" * 55)
    print("MAX-FUSION ATTENTION FAITHFULNESS REPORT")
    print("=" * 55)
    print(f"Predicted class:       {predicted_class}")
    print(f"Original confidence:   {original_confidence:.4f}")
    print(f"Masked confidence:     {masked_confidence:.4f}")
    print(f"Confidence drop:       {confidence_drop:.4f}")
    print(
        f"Perturbed top regions: {top_percentage * 100:.1f}%"
    )

    if confidence_drop >= 0.10:
        print(
            "\nRESULT: The highlighted regions had a substantial "
            "effect on the classifier prediction."
        )
    elif confidence_drop >= 0.03:
        print(
            "\nRESULT: The highlighted regions had a moderate "
            "effect on the classifier prediction."
        )
    elif confidence_drop > 0:
        print(
            "\nRESULT: The highlighted regions had a small "
            "measurable effect on the classifier prediction."
        )
    else:
        print(
            "\nRESULT: Masking the highlighted regions did not "
            "reduce confidence. The attention visualisation may "
            "not faithfully represent the classifier decision."
        )

    print("=" * 55 + "\n")

    return result


def generate_attention_projection(
    image_path,
    model,
    processor,
    device,
    save_path,
    run_faithfulness_test=False,
    top_percentage=0.20,
    masked_image_save_path=None,
):
    """
    Generates a Max-Fusion Terminal-Layer Attention Projection heatmap.

    
    Method implemented from the dissertation literature survey:
    1. Extract self-attention from the final Vision Transformer layer.
    2. Apply maximum fusion across attention heads.
    3. Apply maximum projection across query tokens.
    4. Remove non-spatial special tokens.
    5. Reconstruct patch scores into the ViT spatial grid.
    6. Generate a localised heatmap overlay.
    7. Optionally validate highlighted regions through classifier-level
       perturbation testing.

    Args:
        image_path:
            Path to the source image.

        model:
            Complete AI-generated image classification model.

        processor:
            Hugging Face image processor.

        device:
            PyTorch device.

        save_path:
            Destination for the heatmap overlay.

        run_faithfulness_test:
            Whether to perturb highlighted regions and measure confidence change.

        top_percentage:
            Percentage of highest-attention regions to perturb.

        masked_image_save_path:
            Optional destination for the perturbed validation image.

    Returns:
        save_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image was not found: {image_path}"
        )

    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    target_encoder = get_target_encoder(model)
    target_encoder.eval()

    # Extract attention matrices from the Vision Transformer.
    with torch.no_grad():
        try:
            outputs = target_encoder(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )
        except TypeError:
            outputs = target_encoder(
                pixel_values=pixel_values,
                output_attentions=True,
            )

    attentions = getattr(
        outputs,
        "attentions",
        None,
    )

    if attentions is None:
        raise RuntimeError(
            "The vision encoder did not return attention weights. "
            "Load the transformer with eager attention where required, "
            "for example attn_implementation='eager'."
        )

    final_attention = attentions[-1]

    if final_attention.ndim != 4:
        raise ValueError(
            "Unexpected attention tensor shape: "
            f"{tuple(final_attention.shape)}"
        )

    token_count = final_attention.shape[-1]

    (
        grid_height,
        grid_width,
        number_of_special_tokens,
    ) = infer_patch_grid(
        pixel_values=pixel_values,
        target_encoder=target_encoder,
        token_count=token_count,
    )

    attention_map = compute_projection_attention(
        attentions=attentions,
        grid_height=grid_height,
        grid_width=grid_width,
        number_of_special_tokens=number_of_special_tokens,
    )

    output_height = int(pixel_values.shape[-2])
    output_width = int(pixel_values.shape[-1])

    cam_image = create_attention_overlay(
        image=image,
        attention_map=attention_map,
        output_size=(output_width, output_height),
        background_weight=0.40,
        heatmap_weight=0.60,
    )

    output_directory = os.path.dirname(save_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    success = cv2.imwrite(
        save_path,
        cam_image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save the attention image to: {save_path}"
        )

    if run_faithfulness_test:
        try:
            verification_result = verify_attention_impact(
                image=image,
                attention_map=attention_map,
                model=model,
                processor=processor,
                device=device,
                top_percentage=top_percentage,
            )

            if masked_image_save_path is not None:
                masked_directory = os.path.dirname(
                    masked_image_save_path
                )

                if masked_directory:
                    os.makedirs(
                        masked_directory,
                        exist_ok=True,
                    )

                verification_result[
                    "masked_image"
                ].save(masked_image_save_path)

        except (ValueError, TypeError, RuntimeError) as error:
            # The heatmap remains usable even when the custom classifier
            # output format prevents automatic faithfulness evaluation.
            print(
                "\nAttention heatmap generated successfully, but "
                f"faithfulness validation could not run: {error}\n"
            )

    return save_path