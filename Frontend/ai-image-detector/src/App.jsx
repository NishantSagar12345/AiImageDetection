import { useState } from "react";
import "./App.css";

const API_URL = "/predict";
const BACKEND_URL = "";

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

  const isFake = result?.prediction?.toLowerCase().includes("ai");

  return (
    <div className="page">
      <div className="container">
        <header className="hero">
          <h1>DeepCheck AI</h1>
          <p>
            AI image authenticity analysis with Deep Learning and
            GPT-4o's reasoning.
          </p>
        </header>

        <div className="upload-panel">
          
            <input type="file" accept="image/*" onChange={handleImageChange} />
            

          <button onClick={predictImage} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>

        <main className="main-layout">
          <section className="preview-section">
            <div className="preview-card">
              <div className="card-title">Uploaded Image</div>
              {previewUrl ? (
                <img src={previewUrl} className="image-box" alt="Uploaded" />
              ) : (
                <div className="image-box placeholder">No image selected</div>
              )}
            </div>

            <div className="preview-card">
              <div className="card-title">Grad-CAM Heatmap</div>
              {gradcamUrl ? (
                <img src={gradcamUrl} className="image-box" alt="Grad-CAM" />
              ) : (
                <div className="image-box placeholder">No Grad-CAM yet</div>
              )}
            </div>
          </section>

          <section className="result-box">
            {loading ? (
              <div className="loading-box">
                <div className="spinner"></div>
                <h2>DeepCheck is analysing...</h2>
                <p>Running Deep Learning prediction, Grad-CAM, and GPT4o explanation.</p>
              </div>
            ) : result ? (
              <>
                <div className={`prediction-card ${isFake ? "fake" : "real"}`}>
                  <span className="prediction-label">Deep learning model Prediction</span>
                  <h2>{result.prediction}</h2>
                  <p>{(result.confidence * 100).toFixed(2)}% confidence</p>
                </div>

                <div className="metrics-grid">
                  <div className="metric-card">
                    <span>Real</span>
                    <strong>
                      {(result.real_probability * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>AI</span>
                    <strong>
                      {(result.fake_probability * 100).toFixed(2)}%
                    </strong>
                  </div>
                </div>

                <div className="bar-container">
                  <label>
                    Real Probability{" "}
                    <span>
                      {(result.real_probability * 100).toFixed(2)}%
                    </span>
                  </label>
                  <div className="bar">
                    <div
                      className="real-bar"
                      style={{ width: `${result.real_probability * 100}%` }}
                    ></div>
                  </div>

                  <label>
                    AI Probability{" "}
                    <span>
                      {(result.fake_probability * 100).toFixed(2)}%
                    </span>
                  </label>
                  <div className="bar">
                    <div
                      className="fake-bar"
                      style={{ width: `${result.fake_probability * 100}%` }}
                    ></div>
                  </div>
                </div>

                {result.llm_explanation && (
                  <div className="llm-box">
                    <h3>GPT-4o Explanation</h3>
                    <p>{result.llm_explanation}</p>
                  </div>
                )}
              </>
            ) : (
              <div className="empty-result">
                <h2>Ready to analyse</h2>
                <p>Upload an image to view prediction, heatmap, and explanation.</p>
              </div>
            )}
          </section>
        </main>
        <footer className="footer">
  <div className="footer-content">
    <div className="footer-left">
      <h3>DeepCheck</h3>
      <p>
        Designed and Developed by <strong>Nishant Sagar Pandey</strong>
      </p>
   
    </div>

    <div className="footer-right">
      <a
        href="https://github.com/NishantSagar12345"
        target="_blank"
        rel="noopener noreferrer"
      >
        GitHub
      </a>

      <a
        href="https://www.linkedin.com/in/nishant-pandey-310551206/"
        target="_blank"
        rel="noopener noreferrer"
      >
        LinkedIn
      </a>
    </div>
  </div>
</footer>
      </div>
    </div>
  );
}

export default App;