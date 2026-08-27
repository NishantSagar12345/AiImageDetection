# DeepCheck AI

An explainable AI-powered web application for detecting AI-generated images using a fine-tuned SigLIP Vision Transformer, Grad-CAM visual attribution, and natural-language explanations.


---

## Overview

DeepCheck AI is an end-to-end Explainable AI (XAI) web application for detecting AI-generated images.

The system combines a fine-tuned **SigLIP Vision Transformer** with **Grad-CAM** and **OpenAI GPT-5.6 Sol** to provide image classification together with visual and natural-language explanations.

Grad-CAM generates class-specific attribution heatmaps, while GPT-5.6 Sol converts the detector-derived visual attribution into a concise explanation suitable for non-technical users.

The complete system is implemented using a **React frontend** and **FastAPI backend** and was deployed on **AWS EC2** using Docker, Nginx and GitHub Actions.

---

## Features

- AI-generated image detection
- Fine-tuned SigLIP Vision Transformer
- Grad-CAM visual attribution
- Natural-language explanation using GPT-5.6 Sol
- Image confidence and class probabilities
- React web interface
- FastAPI backend
- Sample images for testing
- User help section
- Docker containerisation
- AWS EC2 deployment
- Nginx reverse proxy
- GitHub Actions CI/CD

---

## Project Structure

```text
AiImageDetection/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── Backend/
│   ├── model.py
│   ├── inference.py
│   ├── llm_explainer.py
│   ├── main.py
│   ├── models/
│   └── ...
│
├── Frontend/
│   └── ai-image-detector/
│       ├── public/
│       ├── src/
│       ├── package.json
│       └── ...
│
├── Kaggle/
│   └── Training and evaluation notebooks/scripts
│
├── .gitignore
│
└── deploymentarch.txt
```

### Directory Description

**`.github/workflows/`**  
Contains the GitHub Actions workflow used for the CI/CD deployment process.

**`Backend/`**  
Contains the FastAPI API, SigLIP detector, inference pipeline, Grad-CAM generation and GPT-5.6 Sol explanation component.

**`Frontend/ai-image-detector/`**  
Contains the React frontend used for image upload, sample image selection and presentation of prediction and explanation outputs.

**`Kaggle/`**  
Contains notebooks and scripts used for dataset preparation, model training and evaluation.

**`deploymentarch.txt`**  
Contains additional information relating to the deployment architecture.

---

## Dataset

The primary dataset used for training the DeepCheck detector is **OpenFake**.

**OpenFake Dataset:**  
https://huggingface.co/datasets/ComplexDataLab/OpenFake

AI Detector Arena v0.1 was additionally used as an independent evaluation dataset.

The datasets themselves are not included directly in this repository.

---

## AI Prompt Documentation

The prompt used for the GPT-5.6 Sol natural-language explanation component is documented separately.

The **AI Prompt Document is available inside the `Research Proposal` OneDrive folder** associated with the dissertation project.

The document contains the prompt and instructions used to guide the natural-language explanation component.

---

## Requirements

Before running the application, make sure the following are installed:

- Python 3.11+
- Node.js
- npm
- Git

Docker is optional when running the backend locally but is used for the deployed version.

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone <repository-url>
cd AiImageDetection
```

---

## 2. Set Up the Backend

Move into the backend directory:

```bash
cd Backend
```

Create a Python virtual environment:

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 3. Add the Trained Model

Place the trained model checkpoint inside:

```text
Backend/models/
```

The backend expects the trained model to be available before starting the prediction service.

---

## 4. Configure Environment Variables

Create a `.env` file inside the `Backend` directory:

```text
Backend/.env
```

Add the OpenAI API key:

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

Do not commit the `.env` file or API keys to GitHub.

---

## 5. Start the FastAPI Backend

From the `Backend` directory, run:

```bash
uvicorn main:app --reload
```

The backend will normally start at:

```text
http://127.0.0.1:8000
```

FastAPI documentation can be accessed at:

```text
http://127.0.0.1:8000/docs
```

---

## 6. Set Up the Frontend

Open another terminal and navigate to:

```bash
cd Frontend/ai-image-detector
```

Install the frontend dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

The terminal will display the local URL for the frontend.

Open this URL in a browser to use DeepCheck.

---

## Running the Backend with Docker

The backend can alternatively be run using Docker.

Navigate to:

```bash
cd Backend
```

Build and start the container:

```bash
docker compose up -d --build
```

Check the running container:

```bash
docker ps
```

View backend logs:

```bash
docker logs deepcheck-backend
```

Stop the backend:

```bash
docker compose down
```

---

## Using DeepCheck

1. Open the DeepCheck web application.
2. Upload an image or select one of the provided sample images.
3. Start the image analysis.
4. DeepCheck returns:
   - predicted class,
   - confidence,
   - Real probability,
   - AI-generated probability,
   - Grad-CAM attribution heatmap,
   - natural-language explanation.

---

## Deployment

The deployed version of DeepCheck uses:

- AWS EC2
- Docker
- Nginx
- GitHub Actions

The React frontend is served through Nginx, while the FastAPI backend runs inside a Docker container.

GitHub Actions is used to automate deployment following updates to the configured repository branch.

---

## Security

Sensitive files and credentials should never be committed to the repository.

The `.gitignore` should include:

```gitignore
.env
*.pem
__pycache__/
node_modules/
```

AWS credentials, private SSH keys and OpenAI API keys should be stored securely outside the repository.

---

## Dissertation

DeepCheck was developed as part of an MSc Artificial Intelligence dissertation at the **University of Stirling**.

**Project:**  
*DeepCheck: AI-Generated Image Detection via a Full Stack Web Application*

---

## Author

**Nishant Sagar Pandey**  
MSc Artificial Intelligence  
University of Stirling