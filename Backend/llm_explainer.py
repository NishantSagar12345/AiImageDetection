import os
import base64
import tempfile
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def image_to_png_data_url(image_path, max_size=768):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:
        temp_path = tmp.name

    image.save(temp_path, format="PNG")

    with open(temp_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    os.remove(temp_path)

    return f"data:image/png;base64,{encoded}"


def explain_gradcam_with_llm(
    original_image_path,
    gradcam_path,
    prediction,
    real_prob,
    fake_prob
):

    if os.getenv("OPENAI_API_KEY") is None:
        return "LLM explanation unavailable because OPENAI_API_KEY is not configured."

    original_data_url = image_to_png_data_url(
        original_image_path
    )

    gradcam_data_url = image_to_png_data_url(
        gradcam_path
    )

    prompt = f"""
You are analysing the output of an AI-generated image detector.

Image 1 is the original image.
Image 2 is the Grad-CAM heatmap overlay.

Prediction: {prediction}
Real probability: {real_prob:.3f}
AI-generated probability: {fake_prob:.3f}

Compare Image 1 and Image 2. Identify the actual image regions highlighted by the heatmap.
Do not speculate about specific visual artefacts or claim that certain features are characteristic of AI-generated images unless they are directly observable. Base the explanation only on the highlighted regions in the Grad-CAM heatmap and describe them as areas that may have influenced the classifier's decision.
Explain why these specific regions may have contributed to the detector's prediction.
Use cautious language such as "may", "could", or "might".
Do not simply say "the highlighted regions contributed".
Do not claim absolute proof.
Do not state that the detector was confused, misled, or fooled.
Interpret the Grad-CAM heatmap only as an indication of which image regions influenced the classifier's decision.
Always include the probability percentages in the paragraph.
Write one concise paragraph under 150 words for a non-technical user.
"""

    try:

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": original_data_url
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": gradcam_data_url
                            }
                        }
                    ]
                }
            ],

            temperature=0.2,
            max_completion_tokens=200

        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM explanation unavailable: {str(e)}"