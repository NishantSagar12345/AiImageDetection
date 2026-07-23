import base64
import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)


def image_to_png_data_url(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> str:
    """
    Load an image, resize it to a fixed size, and convert it
    to a base64-encoded PNG data URL.

    Using the same target size for both images helps preserve
    coordinate alignment between the original and Grad-CAM image.
    """

    if not os.path.isfile(image_path):
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
    gradcam_path: str,
    prediction: str,
    real_prob: float,
    fake_prob: float,
) -> str:
    """
    Generate a concise, non-technical explanation of the
    strongest Grad-CAM regions.
    """

    if not OPENAI_API_KEY:
        return (
            "LLM explanation unavailable because "
            "OPENAI_API_KEY is not configured."
        )

    required_files = {
        "original image": original_image_path,
        "Grad-CAM image": gradcam_path,
    }

    for description, file_path in required_files.items():
        if not file_path or not os.path.isfile(file_path):
            return (
                "LLM explanation unavailable because the "
                f"{description} could not be found."
            )

    try:
        # Both images are converted to exactly 224 × 224.
        original_data_url = image_to_png_data_url(
            image_path=original_image_path,
            target_size=(224, 224),
        )

        gradcam_data_url = image_to_png_data_url(
            image_path=gradcam_path,
            target_size=(224, 224),
        )

        # Input probabilities are between 0 and 1.
        real_percentage = real_prob * 100
        fake_percentage = fake_prob * 100

        prompt = f"""
You are analysing the output of an AI-generated image detector.

You will receive two spatially aligned images in this exact order:

Image 1: The original image.
Image 2: The Grad-CAM heatmap overlay of the same image.

Both images have the same dimensions. A location in Image 2 therefore
corresponds to the same location in Image 1.

Model results:

Prediction: {prediction}
Real probability: {real_percentage:.1f}%
AI-generated probability: {fake_percentage:.1f}%

Your task is to explain which regions may have received the strongest
attention from the classifier.

PRIMARY EVIDENCE RULE

Use Image 2 as the only source for determining where the classifier
focused.

Use Image 1 only to identify the object, object part, or background area
located underneath a highlighted region.

Do not use the colours, appearance, or composition of Image 1 to decide
which areas are important.

Interpret genuine Grad-CAM activation as a continuous heatmap gradient:

- Red: strongest activation
- Bright yellow next to red: strong activation
- Green: moderate activation
- Blue: weak activation

Instructions:

1. Identify at most the two strongest red regions in Image 2.
2. If no red region exists, identify the strongest bright-yellow region
   that forms part of a visible Grad-CAM gradient.
3. Use Image 1 only to name what lies underneath those exact regions.
4. If only part of an object is highlighted, describe only that part.
5. Mention a background region when it contains a strong activation.
6. Ignore green and blue regions when red or valid bright-yellow regions
   are present.
7. Do not describe any part of Image 1 that does not correspond to a
   strong activation in Image 2.

COLOUR-SEPARATION RULE

A naturally coloured object in Image 1 must not be treated as a Grad-CAM
activation merely because the same colour remains visible in Image 2.

Ignore an isolated yellow area when:

- it appears yellow in both Image 1 and Image 2; and
- it is not part of a surrounding Grad-CAM gradient containing nearby
  red, green, or blue activation colours.

Treat such isolated yellow as part of the original image rather than as
heatmap evidence.

Important limitations:

- Grad-CAM shows where the classifier focused, not why it made its
  decision.
- Say that the identified regions may, might, or could have influenced
  the prediction.
- Do not claim that highlighted regions prove the image is real or
  AI-generated.
- Do not say that an object is characteristic of AI-generated images.
- Do not invent reasons that cannot be directly inferred from the
  images.
- Do not speculate about lighting, textures, shadows, reflections, or
  visual artefacts unless they are clearly visible inside the strongest
  highlighted region.
- Include both prediction probabilities as percentages.

Write exactly one concise paragraph between 80 and 120 words for a
non-technical audience.
""".strip()

        response = client.chat.completions.create(
            model="gpt-4o",
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
                                "Use it only to identify what is located "
                                "under the highlighted Grad-CAM regions."
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
                                "Image 2: Grad-CAM heatmap overlay. "
                                "Use this image to locate the strongest "
                                "classifier attention."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": gradcam_data_url,
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