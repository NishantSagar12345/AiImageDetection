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
You are analysing the output of an AI-generated image detector.

You will receive two spatially aligned images.

Image 1: Original image.
Image 2: Internal Grad-CAM attention guide.

Both images have identical dimensions, so each location in Image 2 corresponds to the same location in Image 1.

Prediction:
{prediction}

Real probability: {real_percentage:.1f}%
AI-generated probability: {fake_percentage:.1f}%

Task:

Use Image 2 only to identify the two strongest highlighted regions.

Use Image 1 only to identify the object part or background directly underneath those highlighted regions.

Rules:

- Never use Image 1 to determine attention.
- Describe only what is directly underneath the highlighted region.
- Ignore tiny isolated highlighted regions that appear to be noise.
- If only one meaningful highlighted region exists, describe only that region.
- If the highlighted region is on the background, identify only that background element.


Limitations:

- Grad-CAM indicates where the classifier focused, not why.
- Highlighted regions do not prove an image is real or AI-generated.
- Do not speculate beyond the highlighted region.
- Use cautious words such as "may", "might" or "could".
- Include both probabilities.

Output rules:

The user only sees the coloured transparent Grad-CAM overlay.

Never mention:
- Image 1 or Image 2
- attention guide
- mask
- white or black regions
- internal processing

Instead, refer to them only as:
- highlighted region
- highlighted area
- Grad-CAM highlighted region

Write one paragraph of 80–120 words for a non-technical audience.
""".strip()

        response = client.chat.completions.create(
            model="gpt-4.1",
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
            temperature=0.1,
            max_completion_tokens=220,
        )

        explanation = response.choices[0].message.content

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