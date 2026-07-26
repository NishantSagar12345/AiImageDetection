import base64
import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


def image_to_png_data_url(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> str:
    """
    Resize an image to a fixed size and convert it into
    a base64-encoded PNG data URL.
    """

    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file could not be found: {image_path}"
        )

    temp_path = None

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

            image = image.resize(
                target_size,
                Image.Resampling.LANCZOS,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

            image.save(
                temp_path,
                format="PNG",
            )

        with open(temp_path, "rb") as file:
            encoded_image = base64.b64encode(
                file.read()
            ).decode("utf-8")

        return f"data:image/png;base64,{encoded_image}"

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def explain_gradcam_with_llm(
    original_image_path: str,
    attention_mask_path: str,
    prediction: str,
    real_prob: float,
    fake_prob: float,
) -> str:
    """
    Generate a concise explanation using:

    Image 1: Original image
    Image 2: Binary high-attention mask

    The mask is the only source used to locate important regions.
    """

    if client is None:
        return (
            "LLM explanation unavailable because "
            "OPENAI_API_KEY is not configured."
        )

    required_files = {
        "original image": original_image_path,
        "attention mask": attention_mask_path,
    }

    for description, file_path in required_files.items():
        if not file_path or not os.path.isfile(file_path):
            return (
                "LLM explanation unavailable because the "
                f"{description} could not be found."
            )
     

    try:
        original_data_url = image_to_png_data_url(
            image_path=original_image_path,
            target_size=(224, 224),
        )

        attention_mask_data_url = image_to_png_data_url(
            image_path=attention_mask_path,
            target_size=(224, 224),
        )

        real_percentage = real_prob * 100
        fake_percentage = fake_prob * 100

        prompt = f"""
You are explaining the output of an AI-generated image detector for a non-technical audience.

You will receive two spatially aligned images in order:
1. Original image
2. Internal Grad-CAM guide (white areas indicate classifier focus; black areas should be ignored)

Both images have identical dimensions, so every location in the second image corresponds exactly to the same location in the first image.

Prediction: {prediction}
Real probability: {real_percentage:.1f}%
AI-generated probability: {fake_percentage:.1f}%

Task:
- Step 1: Use the second image only to locate the main white areas. Ignore small isolated noise spots.
- Step 2: Describe only the object part or background located at the same positions in the original image.

Rules:
- Never use the first image alone to determine where focus is located.
- Describe only what is directly covered by the white areas in the second image.
- If only part of an object is covered, describe only that specific part.
- Do not describe or mention unrelated objects outside the white areas in the second image.
- Do not speculate beyond what is directly visible in those areas.
- Use cautious language such as "may", "might", or "could".
- Always include both the real probability and AI-generated probability in your explanation.

Output Rules:
The user only sees a colored transparent overlay on the image.

Forbidden vocabulary (DO NOT MENTION):
- "Image 1" or "Image 2"
- "attention guide" or "attention map"
- "mask"
- "white areas" or "black areas"
- "internal processing"

Allowed vocabulary:
Refer to focal areas ONLY as "highlighted regions", "highlighted areas", or "Grad-CAM highlighted regions".

Format:
Write exactly one paragraph (5 to 7 sentences, 150 words).
""".strip()

        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            reasoning_effort="none",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "text",
                            "text": (
                                "Image 1: Original image. "
                                "Use this only to identify what lies "
                                "underneath the white mask regions."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": original_data_url,
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Image 2: Black-and-white attention "
                                "mask. Use only the white regions to "
                                "locate strong classifier attention."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": attention_mask_data_url,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
           
            max_completion_tokens=200,
        )
        
        explanation = response.choices[0].message.content
        print("The response is ",response)
        if not explanation:
            return (
                "LLM explanation unavailable because "
                "the model returned an empty response."
            )

        return explanation.strip()

    except FileNotFoundError as error:
        return f"LLM explanation unavailable: {error}"

    except Exception as error:
        return f"LLM explanation unavailable: {error}"