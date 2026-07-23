import os

import torch
from PIL import Image

from gradcam_utils import generate_gradcam


def predict_image(
    image_path,
    model,
    processor,
    device,
    gradcam_path=None,
    attention_mask_path=None,
    threshold=0.5,
):
    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    if fake_prob >= threshold:
        prediction = "AI-generated"
        confidence = fake_prob
    else:
        prediction = "Real"
        confidence = real_prob

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
    }

    if gradcam_path is not None:
        if attention_mask_path is None:
            raise ValueError(
                "attention_mask_path is required when "
                "gradcam_path is provided."
            )

        generate_gradcam(
            image_path=image_path,
            model=model,
            processor=processor,
            device=device,
            save_path=gradcam_path,
            mask_save_path=attention_mask_path,
        )

        result["gradcam_url"] = (
            f"/gradcam/{os.path.basename(gradcam_path)}"
        )

    return result

