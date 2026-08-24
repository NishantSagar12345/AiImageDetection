import os
import base64
import tempfile
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file.
# This is used to retrieve the OpenAI API key without hard-coding it.
load_dotenv()


# Initialise the OpenAI client using the API key
# stored in the OPENAI_API_KEY environment variable.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def image_to_png_data_url(image_path, max_size=768):
    # Open the input image and ensure that it uses RGB colour format.
    image = Image.open(image_path).convert("RGB")

    # Resize the image while preserving its aspect ratio.
    # Neither dimension will exceed max_size pixels.
    # This reduces the amount of image data sent to the API.
    image.thumbnail((max_size, max_size))

    # Create a temporary PNG file for storing the resized image.
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:
        temp_path = tmp.name

    # Save the resized RGB image as a PNG file.
    image.save(temp_path, format="PNG")

    # Read the PNG image as binary data and convert it
    # into a Base64-encoded UTF-8 string.
    with open(temp_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    # Remove the temporary file after encoding has completed.
    os.remove(temp_path)

    # Return the encoded image as a data URL that can
    # be supplied directly to the OpenAI multimodal model.
    return f"data:image/png;base64,{encoded}"


def explain_gradcam_with_llm(
    gradcam_path,
    original_image_path,
    prediction,
    real_prob,
    fake_prob
):

    # Check that an OpenAI API key is available before
    # attempting to generate the natural-language explanation.
    if os.getenv("OPENAI_API_KEY") is None:
        return "LLM explanation unavailable because OPENAI_API_KEY is not configured."

    # Convert the Grad-CAM visualisation into a Base64 PNG data URL
    # so that it can be provided to the multimodal model.
    gradcam_data_url = image_to_png_data_url(
        gradcam_path
    )

    # Convert the corresponding original input image into
    # a Base64 PNG data URL.
    original_image_data_url = image_to_png_data_url(
        original_image_path
    )

    # Construct the instruction prompt for the multimodal LLM.
    #
    # The prompt provides:
    # 1. The detector's predicted class.
    # 2. The probabilities for the Real and AI-generated classes.
    # 3. Instructions for interpreting Grad-CAM colours.
    # 4. Constraints intended to prevent unsupported interpretation.
    # 5. Instructions to use the original image only for identifying
    #    what exists at locations highlighted by Grad-CAM.
    prompt = f"""
You are analysing the visual explanation produced by an AI-generated image detector.

Two images are provided:

Image 1 is the original input image.
Image 2 is the class-specific Grad-CAM attribution heatmap overlay generated for the detector's predicted class.

Use the original image only to identify the objects, structures, people, or scene regions corresponding to highlighted areas in the Grad-CAM overlay.

The Grad-CAM overlay must be the primary and only source for determining which image regions received attribution.

Prediction: {prediction}
Real probability: {real_prob:.3f}
AI-generated probability: {fake_prob:.3f}

Interpret the Grad-CAM heatmap using the following rules:

• Red / Orange = strongest class-specific attribution regions (Primary Activation Hotspots). Larger and more intense red/orange regions indicate stronger Grad-CAM attribution.
• Yellow / Green = moderate or weaker attribution.
• Blue = low attribution.
• Grey = underlying image background only and should not be interpreted as attribution.

Focus primarily on the strongest red and orange regions. Mention at most four dominant highlighted regions. If only part of an object or area is strongly highlighted, describe only that highlighted part rather than the whole object.

Important rules:
• Determine attribution locations only from the Grad-CAM overlay.
• Use the original image only to understand what is present at those highlighted locations.
• Do not identify a region as important simply because it is visually prominent in the original image.
• Ignore grey background regions in the Grad-CAM overlay unless they contain a coloured Grad-CAM activation.
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
Write one concise paragraph of approximately 150 words for a non-technical audience.
"""

    try:

        # Send the prompt, original image, and Grad-CAM visualisation
        # to the multimodal OpenAI model.
        response = client.chat.completions.create(

            # Model used to generate the natural-language explanation.
            model="gpt-5.6-sol",

            # Disable additional reasoning effort for this
            # straightforward visual interpretation task.
            reasoning_effort="none",

            # Construct a multimodal user message containing:
            # - textual interpretation instructions,
            # - the original input image,
            # - the Grad-CAM attribution visualisation.
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            # Send the interpretation instructions as text.
                            "type": "text",
                            "text": prompt
                        },
                        {
                            # Image 1: original input image.
                            # This provides semantic context for identifying
                            # objects or regions highlighted by Grad-CAM.
                            "type": "image_url",
                            "image_url": {
                                "url": original_image_data_url
                            }
                        },
                        {
                            # Image 2: Grad-CAM attribution overlay.
                            # This is used as the primary source for determining
                            # which regions received strong model attribution.
                            "type": "image_url",
                            "image_url": {
                                "url": gradcam_data_url
                            }
                        }
                    ]
                }
            ],

            # Restrict the maximum generated response length
            # to keep the explanation concise.
            max_completion_tokens=200

        )

        # Extract and return the natural-language explanation
        # generated by the model.
        return response.choices[0].message.content

    except Exception as e:
        # Handle API, network, authentication, or other runtime errors
        # without causing the wider application to fail.
        return f"LLM explanation unavailable: {str(e)}"