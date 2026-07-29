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

• Red / Orange zones = Main focus area (Primary Activation Hotspots)

• Blue zones Bright Yellow and Green zones = Least focus area (Peripheral and Contextual Triggers)

Base the explanation only on the strongest red and orange regions. Mention at most the two most strongly highlighted regions. Ignore green and blue regions unless no stronger activations exist. If only part of an object is highlighted, describe only that highlighted part.

Important rules:
• Base the explanation only on information visible in the Grad-CAM heatmap.
• The Grad-CAM heatmap indicates where the classifier focused its attention. It does not explain the model's reasoning.
• Explain that the highlighted regions may have influenced the prediction.
• Do not invent reasons that cannot be directly inferred from the images.
• Do not state that any object is characteristic of AI-generated images.
• If the higlighted areas are in the background near the face of the person then do not focus on the person facial features.
• Do not speculate about lighting, textures, shadows or visual artefacts unless they are clearly visible within the strongest highlighted regions.
• Always check for the background regions including the edges of the image 2 for primary Activation Hotspots 
• Use cautious language such as "may", "might" or "could".
• Include both prediction probabilities as percentages.
Return the explanation as plain text with no Markdown formatting. 
Write one concise paragraph (80–120 words) suitable for a non-technical audience."""

    try:

        response = client.chat.completions.create(

            model="gpt-5.6-luna",
            reasoning_effort="none",
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
            max_completion_tokens=200

        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM explanation unavailable: {str(e)}"