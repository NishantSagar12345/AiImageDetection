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
    You are analysing the visual explanation produced by an AI-generated image detector.

The provided image is a class-specific Grad-CAM attribution heatmap overlay generated for the detector's predicted class.

Prediction: {prediction}
Real probability: {real_prob:.3f}
AI-generated probability: {fake_prob:.3f}

Interpret the heatmap using the following rules:

• Red / Orange = strongest class-specific attribution regions (Primary Activation Hotspots). Larger and more intense red/orange regions indicate stronger Grad-CAM attribution.
• Yellow / Green = moderate or weaker attribution.
• Blue = low attribution.
• Grey = underlying image background only and should not be interpreted as attribution.

Focus primarily on the strongest red and orange regions. Mention at most four dominant highlighted regions. If only part of an object or area is strongly highlighted, describe only that highlighted part rather than the whole object.

Important rules:
• Base the spatial explanation only on information visible in the Grad-CAM overlay.
• Ignore grey background regions unless they contain a coloured Grad-CAM activation.
• Describe highlighted regions as areas strongly associated with the predicted class.
• Do not claim that a highlighted region definitively caused the prediction or reveals the model's complete reasoning.
• Do not invent explanations that cannot be directly supported by the visualisation.
• Do not state that any object, texture, colour, or scene characteristic is inherently typical of AI-generated images.
• If a hotspot appears in the background or near the edge of the image, describe that location accurately rather than assigning it to a nearby object.
• If activation occurs near a person but primarily covers the surrounding background, describe the background region rather than the person.
• Do not speculate about lighting, textures, shadows, manipulation artefacts, or generation artefacts unless they are clearly visible within the strongest highlighted region.
• Use cautious language such as "appears", "may", "might", "could", "is associated with", or "shows strong attribution".
• Include both prediction probabilities as percentages.
• Keep the prediction result and the Grad-CAM interpretation conceptually separate.

Return the explanation as plain text with no Markdown formatting.
Write one concise paragraph of approximately 150 words for a non-technical audience."""

    try:

        response = client.chat.completions.create(

            model="gpt-5.6-sol",
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