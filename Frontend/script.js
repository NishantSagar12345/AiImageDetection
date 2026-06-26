const API_URL = "http://127.0.0.1:8000/predict";
const BACKEND_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const gradcamImage = document.getElementById("gradcamImage");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (file) {
        previewImage.src = URL.createObjectURL(file);
        gradcamImage.src = "";
        document.getElementById("resultBox").classList.add("hidden");
    }
});

async function predictImage() {
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("predictionText").innerText = "Analyzing...";
    document.getElementById("resultBox").classList.remove("hidden");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        document.getElementById("predictionText").innerText =
            `Prediction: ${result.prediction}`;

        document.getElementById("confidenceText").innerText =
            `Confidence: ${(result.confidence * 100).toFixed(2)}%`;

        document.getElementById("realBar").style.width =
            `${result.real_probability * 100}%`;

        document.getElementById("fakeBar").style.width =
            `${result.fake_probability * 100}%`;

        if (result.gradcam_url) {
            gradcamImage.src = BACKEND_URL + result.gradcam_url;
        }

    } catch (error) {
        console.error(error);
        alert("Error connecting to backend.");
    }
}