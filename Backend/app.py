import os
import uuid
import torch

# FastAPI components used to create the backend API,
# receive uploaded image files, configure CORS,
# and serve generated Grad-CAM images as static files.
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Hugging Face image processor used to preprocess images
# according to the requirements of the pretrained SigLIP model.
from transformers import AutoImageProcessor

# Import the custom SigLIP binary classifier.
from model import SigLIPDetector

# Import the inference pipeline responsible for generating
# predictions, probabilities and Grad-CAM visualisations.
from inference import predict_image

# Import the multimodal LLM explanation component used to
# generate a natural-language interpretation of the Grad-CAM output.
from llm_explainer import explain_gradcam_with_llm


# Directory used to temporarily store uploaded input images.
UPLOAD_DIR = "uploads"

# Directory used to store generated Grad-CAM visualisations.
GRADCAM_DIR = "gradcam"

# Path to the trained SigLIP detector model weights.
MODEL_PATH = "models/best_openfake_siglip50k.pth"

# Hugging Face identifier for the pretrained SigLIP backbone.
MODEL_NAME = "google/siglip-base-patch16-224"


# Create the upload and Grad-CAM directories if they
# do not already exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


# Initialise the FastAPI web application and provide
# basic metadata describing the API.
app = FastAPI(
    title="AI Image Detector API",
    description=(
        "AI-generated image detection using SigLIP, "
        "Grad-CAM and Vision LLM explanation"
    ),
    version="1.0",
)


# Configure Cross-Origin Resource Sharing (CORS).
# This allows the frontend application to communicate
# with the FastAPI backend from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Expose the Grad-CAM output directory through the /gradcam URL.
# This allows generated attribution images to be accessed
# by the frontend after inference.
app.mount(
    "/gradcam",
    StaticFiles(directory=GRADCAM_DIR),
    name="gradcam",
)


# Select GPU acceleration when CUDA is available.
# Otherwise, inference is performed on the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"


# Display which computational device is being used.
if device == "cuda":
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using the CPU Now")


# Load the image processor associated with the pretrained
# SigLIP model. It prepares uploaded images in the format
# required by the vision encoder.
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)


# Initialise the custom SigLIP detector and move it
# to the selected computational device.
model = SigLIPDetector(
    MODEL_NAME
).to(device)


# Load the previously trained model parameters.
# map_location ensures that the weights are loaded
# onto the currently selected CPU or GPU device.
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
)


# Restore the trained parameters into the model architecture.
model.load_state_dict(
    state_dict
)


# Set the model to evaluation mode.
# This disables training-specific behaviour such as dropout.
model.eval()


# Root endpoint used to confirm that the FastAPI backend
# is running and to report the active computational device.
@app.get("/")
def home():
    return {
        "message": (
            "AI Image Detector backend is running successfully"
        ),
        "device": device,
    }


# Prediction endpoint.
# The frontend sends an uploaded image to this endpoint
# using an HTTP POST request.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Extract the uploaded file extension and convert
    # it to lowercase for format validation.
    file_ext = file.filename.split(".")[-1].lower()


    # Define the image formats accepted by the API.
    allowed_exts = [
        "jpg",
        "jpeg",
        "png",
        "webp",
        "avif",
        "bmp",
        "tiff",
    ]


    # Reject the uploaded file if its extension
    # is not included in the supported image formats.
    if file_ext not in allowed_exts:
        return {
            "error": "Unsupported image format."
        }


    # Generate a unique identifier for the current request.
    # This prevents uploaded images from overwriting each other.
    unique_id = str(uuid.uuid4())


    # Construct a unique path for storing the uploaded image.
    upload_path = os.path.join(
        UPLOAD_DIR,
        f"{unique_id}.{file_ext}",
    )


    # Construct a corresponding path for the Grad-CAM
    # visualisation generated for this image.
    gradcam_path = os.path.join(
        GRADCAM_DIR,
        f"gradcam_{unique_id}.png",
    )


    # Read the uploaded image asynchronously and save
    # its binary contents to the uploads directory.
    with open(upload_path, "wb") as file_object:
        file_object.write(
            await file.read()
        )


    # Run the trained SigLIP detector on the uploaded image.
    #
    # The inference pipeline performs image preprocessing,
    # binary classification and Grad-CAM generation.
    #
    # A threshold of 0.5 is used when assigning the
    # final Real or AI-generated class.
    result = predict_image(
    image_path=upload_path,
    model=model,
    processor=processor,
    device=device,
    gradcam_path=gradcam_path,
    threshold=0.5,
    )


    # Generate a natural-language explanation using the
    # original uploaded image, its Grad-CAM visualisation,
    # the predicted class and both class probabilities.
    explanation = explain_gradcam_with_llm(
        gradcam_path=gradcam_path,
        original_image_path=upload_path, 
        prediction=result["prediction"],
        real_prob=result["real_probability"],
        fake_prob=result["fake_probability"],
    )


    # Add the generated LLM explanation to the inference result.
    result["llm_explanation"] = explanation


    # Return the complete prediction result to the frontend.
    return result