import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";
const BACKEND_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [gradcamUrl, setGradcamUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setGradcamUrl("");
      setResult(null);
    }
  };

  const predictImage = async () => {
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      setResult(data);

      if (data.gradcam_url) {
        setGradcamUrl(BACKEND_URL + data.gradcam_url);
      }
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>AI-Generated Image Detector</h1>
      <p className="subtitle">
        Upload an image to check whether it is real or AI-generated.
      </p>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button onClick={predictImage} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      </div>

      <div className="preview-section">
        <div>
          <h3>Uploaded Image</h3>
          {previewUrl ? (
            <img src={previewUrl} className="image-box" alt="Uploaded" />
          ) : (
            <div className="image-box placeholder">No image selected</div>
          )}
        </div>

        <div>
          <h3>Grad-CAM Explanation</h3>
          {gradcamUrl ? (
            <img src={gradcamUrl} className="image-box" alt="Grad-CAM" />
          ) : (
            <div className="image-box placeholder">No Grad-CAM yet</div>
          )}
        </div>
      </div>

      {(result || loading) && (
        <div className="result-box">
          <h2>
            {loading ? "Analyzing..." : `Prediction: ${result.prediction}`}
          </h2>

          {!loading && (
            <>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>

              <div className="bar-container">
                <label>Real Probability</label>
                <div className="bar">
                  <div
                    className="real-bar"
                    style={{
                      width: `${result.real_probability * 100}%`,
                    }}
                  ></div>
                </div>

                <label>AI Probability</label>
                <div className="bar">
                  <div
                    className="fake-bar"
                    style={{
                      width: `${result.fake_probability * 100}%`,
                    }}
                  ></div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;