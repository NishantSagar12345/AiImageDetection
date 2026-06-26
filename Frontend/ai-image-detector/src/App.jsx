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
    setGradcamUrl("");

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
      <h1>DeepCheck</h1>

      <p className="subtitle">
        Explainable AI-generated image detection using Deep Learning and
        Vision-Language reasoning.
      </p>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleImageChange} />

        <button onClick={predictImage} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      </div>

      <div className="main-layout">
        <div className="preview-section">
          <div className="preview-card">
            <h3>Uploaded Image</h3>

            {previewUrl ? (
              <img src={previewUrl} className="image-box" alt="Uploaded" />
            ) : (
              <div className="image-box placeholder">No image selected</div>
            )}
          </div>

          <div className="preview-card">
            <h3>Grad-CAM Explanation</h3>

            {gradcamUrl ? (
              <img src={gradcamUrl} className="image-box" alt="Grad-CAM" />
            ) : (
              <div className="image-box placeholder">No Grad-CAM yet</div>
            )}
          </div>
        </div>

        <div className="result-box">
          {loading ? (
            <div className="loading-box">
              <h2>Analyzing...</h2>
              <p>
                Generating prediction, Grad-CAM heatmap, and Vision-Language
                explanation.
              </p>
            </div>
          ) : result ? (
            <>
              <h2>Prediction: {result.prediction}</h2>

              <p className="confidence-text">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>

              <div className="bar-container">
                <label>
                  Real Probability:{" "}
                  {(result.real_probability * 100).toFixed(2)}%
                </label>

                <div className="bar">
                  <div
                    className="real-bar"
                    style={{
                      width: `${result.real_probability * 100}%`,
                    }}
                  ></div>
                </div>

                <label>
                  AI Probability: {(result.fake_probability * 100).toFixed(2)}%
                </label>

                <div className="bar">
                  <div
                    className="fake-bar"
                    style={{
                      width: `${result.fake_probability * 100}%`,
                    }}
                  ></div>
                </div>
              </div>

              {result.llm_explanation && (
                <div className="llm-box">
                  <h3>Vision-Language Explanation</h3>
                  <p>{result.llm_explanation}</p>
                </div>
              )}
            </>
          ) : (
            <div className="empty-result">
              Upload an image and click Analyze Image to view prediction
              results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;