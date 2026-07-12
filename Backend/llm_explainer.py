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

Focus primarily on Image 2 (the Grad-CAM heatmap). Treat the heatmap as the primary source of information.

Use Image 1 only as a reference to identify the objects or regions that correspond to the highlighted areas in the heatmap. Do not base your explanation on the colours, appearance or composition of the original image unless they directly overlap with the highlighted regions in the heatmap.

Interpret the heatmap using the following priority:
• Red = highest importance
• Bright yellow = high importance
• Green = moderate importance
• Blue = low importance

Base the explanation only on the strongest red and bright yellow regions. Mention at most the two most strongly highlighted regions. Ignore green and blue regions unless no stronger activations exist. If only part of an object is highlighted, describe only that highlighted part.

Important rules:
• Base the explanation only on information visible in the Grad-CAM heatmap.
• The Grad-CAM heatmap indicates where the classifier focused its attention. It does not explain the model's reasoning.
• Explain that the highlighted regions may have influenced the prediction.
• Do not identify important regions from the colours of the original image.
• Do not invent reasons that cannot be directly inferred from the images.
• Do not state that any object is characteristic of AI-generated images.
• Do not speculate about lighting, textures, shadows or visual artefacts unless they are clearly visible within the strongest highlighted regions.
• If the highlighted region in the heatmap is in the background talk about it also.
• Use cautious language such as "may", "might" or "could".
• Include both prediction probabilities as percentages.

Write one concise paragraph (80–120 words) suitable for a non-technical audience."""

    try:

        response = client.chat.completions.create(

            model="gpt-4o",

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