import base64
import io
import os
import tempfile
import cv2
import json
import numpy as np
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
    resampling=Image.Resampling.LANCZOS,
) -> str:
    """
    Resize an image and convert it into a base64 PNG data URL.
    """

    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file could not be found: {image_path}"
        )

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

        image = image.resize(
            target_size,
            resampling,
        )

        image_buffer = io.BytesIO()

        image.save(
            image_buffer,
            format="PNG",
        )

    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"
def create_region_crops(
    original_path: str,
    mask_path: str,
    output_directory: str,
    max_regions: int = 2,
    min_area: int = 30,
    padding: int = 4,
) -> list[str]:
    """
    Find the largest connected white regions in the binary mask
    and crop those exact locations from the original image.
    """

    original = Image.open(original_path).convert("RGB")
    original = original.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(
            f"Could not read attention mask: {mask_path}"
        )

    mask = cv2.resize(
        mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST,
    )

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    regions = []

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])

        regions.append(
            {
                "area": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    regions.sort(
        key=lambda region: region["area"],
        reverse=True,
    )

    regions = regions[:max_regions]

    crop_paths = []

    for region_number, region in enumerate(regions, start=1):
        left = max(0, region["x"] - padding)
        top = max(0, region["y"] - padding)

        right = min(
            224,
            region["x"] + region["width"] + padding,
        )

        bottom = min(
            224,
            region["y"] + region["height"] + padding,
        )

        crop = original.crop(
            (left, top, right, bottom)
        )

        crop_path = os.path.join(
            output_directory,
            f"gradcam_region_{region_number}.png",
        )

        crop.save(crop_path)

        crop_paths.append(crop_path)

    return crop_paths

def create_masked_original(
    original_path: str,
    mask_path: str,
    output_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> str:
    """
    Create an image where only regions selected by the binary
    Grad-CAM mask remain visible.

    White regions in the mask reveal the original image.
    Black regions in the mask remain hidden.
    """

    if not original_path or not os.path.isfile(original_path):
        raise FileNotFoundError(
            f"Original image could not be found: {original_path}"
        )

    if not mask_path or not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"Attention mask could not be found: {mask_path}"
        )

    with Image.open(original_path) as opened_original:
        original = opened_original.convert("RGB").resize(
            target_size,
            Image.Resampling.LANCZOS,
        )

    with Image.open(mask_path) as opened_mask:
        mask = opened_mask.convert("L").resize(
            target_size,
            Image.Resampling.NEAREST,
        )

    original_array = np.asarray(
        original,
        dtype=np.uint8,
    )

    mask_array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    # Only reveal pixels corresponding to white mask regions.
    selected_regions = mask_array >= 128

    masked_array = np.zeros_like(
        original_array,
        dtype=np.uint8,
    )

    masked_array[selected_regions] = original_array[selected_regions]

    masked_image = Image.fromarray(
        masked_array,
        mode="RGB",
    )

    masked_image.save(
        output_path,
        format="PNG",
    )

    return output_path


def get_top_mask_regions(
    mask_path: str,
    max_regions: int = 2,
    min_white_pixels: int = 20,
) -> list[dict]:
    """
    Divide the binary attention mask into a 3×3 grid and return
    the grid cells containing the most white attention pixels.

    This is deterministic and avoids asking an LLM to estimate
    spatial locations.
    """

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not read attention mask: {mask_path}"
        )

    # Ensure the mask is binary.
    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    height, width = binary_mask.shape

    row_edges = [
        0,
        height // 3,
        (2 * height) // 3,
        height,
    ]

    column_edges = [
        0,
        width // 3,
        (2 * width) // 3,
        width,
    ]

    location_names = [
        ["upper-left", "upper-centre", "upper-right"],
        ["centre-left", "centre", "centre-right"],
        ["lower-left", "lower-centre", "lower-right"],
    ]

    regions = []

    for row_index in range(3):
        for column_index in range(3):
            y1 = row_edges[row_index]
            y2 = row_edges[row_index + 1]
            x1 = column_edges[column_index]
            x2 = column_edges[column_index + 1]

            cell = binary_mask[y1:y2, x1:x2]

            white_pixels = int(
                np.count_nonzero(cell == 255)
            )

            cell_pixels = int(cell.size)

            coverage_percentage = (
                (white_pixels / cell_pixels) * 100
                if cell_pixels
                else 0.0
            )

            if white_pixels < min_white_pixels:
                continue

            regions.append(
                {
                    "location": location_names[
                        row_index
                    ][column_index],
                    "white_pixels": white_pixels,
                    "coverage_percentage": round(
                        coverage_percentage,
                        2,
                    ),
                }
            )

    regions.sort(
        key=lambda region: region["white_pixels"],
        reverse=True,
    )

    return regions[:max_regions]
def explain_gradcam_with_llm(
    original_image_path: str,
    attention_mask_path: str,
    prediction: str,
    real_prob: float,
    fake_prob: float,
) -> str:
    """
    Generate a natural-language explanation from exact image crops
    corresponding to the strongest connected regions in the binary
    Grad-CAM attention mask.

    The LLM receives only the cropped highlighted regions, not the
    complete original image.
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

    crop_paths: list[str] = []

    try:
        # ---------------------------------------------------------
        # Create exact crops from the largest attention-mask regions
        # ---------------------------------------------------------

        crop_paths = create_region_crops(
            original_path=original_image_path,
            mask_path=attention_mask_path,
            output_directory=os.path.dirname(attention_mask_path),
            max_regions=2,
            min_area=30,
            padding=4,
        )

        if not crop_paths:
            return (
                "LLM explanation unavailable because no meaningful "
                "Grad-CAM regions were found in the attention mask."
            )

        # ---------------------------------------------------------
        # Convert each crop to a PNG data URL
        # ---------------------------------------------------------

        crop_data_urls = [
            image_to_png_data_url(
                image_path=crop_path,
                target_size=(224, 224),
                resampling=Image.Resampling.LANCZOS,
            )
            for crop_path in crop_paths
        ]

        real_percentage = real_prob * 100
        fake_percentage = fake_prob * 100

        # ---------------------------------------------------------
        # Prompt: describe only what appears inside the exact crops
        # ---------------------------------------------------------

        explanation_prompt = f"""
You are explaining the output of an AI-generated image detector for a non-technical audience.

The supplied images contain exact crops of the strongest Grad-CAM highlighted regions.

Prediction: {prediction}
Real probability: {real_percentage:.1f}%
AI-generated probability: {fake_percentage:.1f}%

Describe only what is clearly visible inside the supplied highlighted regions.

Rules:
- Describe only what is directly visible inside the supplied highlighted regions. Do not identify an object unless enough of that object is visible to support a confident identification.
- Mention an object, person, body part, or background only when it is clearly visible inside a supplied region.
- Do not infer the complete subject from a partial fragment.
- Do not infer nearby objects, hidden object parts, or surrounding areas.
- If a region cannot be identified confidently, describe it neutrally as part of the background or scene.
- Do not infer texture, edges, contours, contrast, lighting, reflections, shadows, realism, photographic quality, or why the classifier focused there.
- State only that the highlighted regions may have contributed to the prediction.
- Include both probabilities.
- Use cautious language such as "may", "might", or "could".
- Never say clearly identifiable object or person.

Do not mention crops, masks, black or white regions, internal processing, separate images, confidently identifiable or image manipulation.

Refer only to:
- "Grad-CAM highlighted regions"
- "highlighted regions"
- "highlighted areas"

Write one paragraph of 70–100 words.
""".strip()

        # ---------------------------------------------------------
        # Build multimodal content dynamically
        # ---------------------------------------------------------

        message_content = [
            {
                "type": "text",
                "text": explanation_prompt,
            }
        ]

        for index, crop_data_url in enumerate(
            crop_data_urls,
            start=1,
        ):
            message_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Highlighted region {index}. "
                            "Describe only what is clearly visible here."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": crop_data_url,
                            "detail": "high",
                        },
                    },
                ]
            )

        # ---------------------------------------------------------
        # LLM request
        # ---------------------------------------------------------

        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            reasoning_effort="none",
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            max_completion_tokens=220,
        )

        print("Generated crop paths:", crop_paths)
        print("Explanation response:", response)

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

    finally:
        # Delete all temporary crop files.
        for crop_path in crop_paths:
            if crop_path and os.path.exists(crop_path):
                try:
                    os.remove(crop_path)
                except OSError as cleanup_error:
                    print(
                        f"Could not remove temporary crop "
                        f"{crop_path}: {cleanup_error}"
                    )