import os
import uuid
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transformers import AutoImageProcessor

from model import SigLIPDetector
from inference import predict_image
from llm_explainer import explain_gradcam_with_llm


UPLOAD_DIR = "uploads"
GRADCAM_DIR = "gradcam"
MODEL_PATH = "models/best_openfake_siglip50k.pth"
MODEL_NAME = "google/siglip-base-patch16-224"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

app = FastAPI(
    title="AI Image Detector API",
    description="AI-generated image detection using SigLIP, Grad-CAM and Vision LLM explanation",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/gradcam",
    StaticFiles(directory=GRADCAM_DIR),
    name="gradcam"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using the CPU Now")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SigLIPDetector(
    MODEL_NAME
).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()


@app.get("/")
def home():
    return {
        "message": "AI Image Detector backend is running successfully",
        "device": device
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    file_ext = file.filename.split(".")[-1].lower()

    allowed_exts = [
        "jpg",
        "jpeg",
        "png",
        "webp",
        "avif",
        "bmp",
        "tiff"
    ]

    if file_ext not in allowed_exts:
        return {
            "error": "Unsupported image format."
        }

    unique_id = str(
        uuid.uuid4()
    )

    upload_path = os.path.join(
        UPLOAD_DIR,
        f"{unique_id}.{file_ext}"
    )

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        f"gradcam_{unique_id}.png"
    )

    with open(upload_path, "wb") as f:
        f.write(
            await file.read()
        )

    result = predict_image(
        image_path=upload_path,
        model=model,
        processor=processor,
        device=device,
        gradcam_path=gradcam_path,
        threshold=0.5
    )

    explanation = explain_gradcam_with_llm(
        original_image_path=upload_path,
        gradcam_path=gradcam_path,
        prediction=result["prediction"],
        real_prob=result["real_probability"],
        fake_prob=result["fake_probability"]
    )

    result["llm_explanation"] = explanation

    return result