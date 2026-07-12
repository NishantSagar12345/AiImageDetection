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

Compare Image 1 and Image 2. Focus primarily on the regions with the strongest activation (red and yellow areas) in the Grad-CAM heatmap, as these represent the image regions that contributed most strongly to the classifier's prediction. Mention green or blue regions only if they provide important additional context.

Important rules:

• Base the explanation only on information visible in the original image and the Grad-CAM heatmap.
• Describe only the regions that are visibly highlighted.
• Explain that the highlighted regions may have influenced the classifier's prediction.
• Treat Grad-CAM as evidence of model attention, not proof of the model's reasoning.
• Do not invent reasons that cannot be directly inferred from the images.
• Do not state that a highlighted object is "characteristic of AI-generated images."
• Do not mention visual artefacts, lighting inconsistencies or texture abnormalities unless they are clearly visible in the original image.
• Use cautious language such as "may", "might" or "could".
• Include both prediction probabilities as percentages.
• Do not explain how Grad-CAM or deep learning works.

Write one concise paragraph (80–120 words) suitable for a non-technical audience.
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