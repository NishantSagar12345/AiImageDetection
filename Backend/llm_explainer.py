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
    gradcam_path,
    prediction,
    real_prob,
    fake_prob
):

    if os.getenv("OPENAI_API_KEY") is None:
        return "LLM explanation unavailable because OPENAI_API_KEY is not configured."

    
    gradcam_data_url = image_to_png_data_url(
        gradcam_path
    )

    prompt = f"""
    You are analysing the output of an AI-generated image detector.

    The image provided is the GradCAM heatmap overlay.

    Prediction: {prediction}
    Real probability: {real_prob:.3f}
    AI-generated probability: {fake_prob:.3f}

    Focus only on the GradCAM heatmap overlay. Treat the heatmap as the primary and only source of information.

    • Red / Orange = Main focus area (Primary Activation Hotspots)

    • Blue zones, bright yellow and green zones = Least focus area
    • Grey Zones = Ignore this zones     
    Base the explanation only on the strongest red and orange regions. Mention at most the four most strongly highlighted regions. Ignore green and blue regions unless no stronger activations exist. If only part of an object or region is highlighted, describe only that highlighted part.

    Important rules:
    • Base the explanation only on information visible in the GradCAM heatmap.
    • Always ignore the Grey Zone Areas
    • The attention heatmap indicates where the classifier focused its attention. It does not explain the model's reasoning.
    • Explain that the highlighted regions may have influenced the prediction.
    • Do not invent reasons that cannot be directly inferred from the heatmap.
    • Do not state that any object is characteristic of AI-generated images.
    • If the highlighted areas are in the background near a person's face, do not focus on the person's facial features.
    • Do not speculate about lighting, textures, shadows or visual artefacts unless they are clearly visible within the strongest highlighted regions.
    • Always check background regions, including the edges of the heatmap, for primary activation hotspots.
    • Use cautious language such as "may", "might" or "could".
    • Include both prediction probabilities as percentages.

    Return the explanation as plain text with no Markdown formatting.
    Write one concise paragraph of 150 words suitable for a non-technical audience."""

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