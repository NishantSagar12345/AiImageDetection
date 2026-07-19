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
  const [showInfo, setShowInfo] = useState(false);
  const [showSamples, setShowSamples] = useState(false);
  const sampleImages = [
  {
    name: "Real Sample 1",
    url: "/sample-images/real-sample-1.jpg",
  },
  {
    name: "Real Sample 2",
    url: "/sample-images/real-sample-2.jpg",
  },
  {
    name: "AI Sample 1",
    url: "/sample-images/ai-sample-1.png",
  },
  {
    name: "AI Sample 2",
    url: "/sample-images/ai-sample-2.jpg",
  },
];
  const selectSampleImage = async (sample) => {
  try {
    const response = await fetch(sample.url);

    if (!response.ok) {
      throw new Error("Could not load sample image.");
    }

    const blob = await response.blob();

    const sampleFile = new File(
      [blob],
      sample.url.split("/").pop(),
      {
        type: blob.type || "image/jpeg",
      }
    );

    setFile(sampleFile);
    setPreviewUrl(sample.url);
    setGradcamUrl("");
    setResult(null);
    setShowSamples(false);
  } catch (error) {
    console.error(error);
    alert("Could not load the selected sample image.");
  }
};
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
    <>
    {/* Fixed top-left button */}
    <button
      type="button"
      className="samples-btn"
      onClick={() => setShowSamples(true)}
      aria-label="Open sample images"
    >
      <span className="samples-icon">🖼️</span>
      <span>Images</span>
    </button>

    {/* Fixed top-right button */}
    <button
      type="button"
      className="info-btn"
      onClick={() => setShowInfo(true)}
      aria-label="About DeepCheck"
    >
      Help
    </button>

    {/* Sample image modal */}
    {showSamples && (
      <div
        className="sample-overlay"
        onClick={() => setShowSamples(false)}
      >
        <div
          className="sample-modal"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="sample-close-btn"
            onClick={() => setShowSamples(false)}
            aria-label="Close sample images"
          >
            ×
          </button>

          <h2>Sample Images</h2>

          <p className="sample-description">
            Select an image to test DeepCheck AI.
          </p>

          <div className="sample-grid">
            {sampleImages.map((sample) => (
              <button
                type="button"
                className="sample-card"
                key={sample.url}
                onClick={() => selectSampleImage(sample)}
              >
                <img src={sample.url} alt={sample.name} />
                <span>{sample.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    )}
    <div className="page">
      <button
      className="samples-btn"
      onClick={() => setShowSamples(true)}
      aria-label="Open sample images"
    >
      <span className="samples-icon">🖼️</span>
      <span>Images</span>
    </button>

    {showSamples && (
      <div
        className="sample-overlay"
        onClick={() => setShowSamples(false)}
      >
        <div
          className="sample-modal"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            className="sample-close-btn"
            onClick={() => setShowSamples(false)}
            aria-label="Close sample images"
          >
            ×
          </button>

          <h2>Sample Images</h2>

          <p className="sample-description">
            Select an image to test DeepCheck AI.
          </p>

          <div className="sample-grid">
            {sampleImages.map((sample) => (
              <button
                type="button"
                className="sample-card"
                key={sample.url}
                onClick={() => selectSampleImage(sample)}
              >
                <img src={sample.url} alt={sample.name} />
                <span>{sample.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    )}
      <button
  className="info-btn"
  onClick={() => setShowInfo(true)}
  aria-label="About DeepCheck"
>
  help
</button>
      <div className="container">
        <header className="hero">
          <h1>DeepCheck</h1>
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
        {showInfo && (
  <div
    className="modal-overlay"
    onClick={() => setShowInfo(false)}
  >
    <div
      className="info-modal"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        className="close-btn"
        onClick={() => setShowInfo(false)}
      >
        ✕
      </button>

      <h2>About DeepCheck AI</h2>

      <p>
        DeepCheck AI is an explainable AI image detection system developed
        by Nishant Sagar Pandey.
      </p>

      <h3>Features</h3>

      <ul>
        <li>Deep Learning image classification</li>
        <li>Grad-CAM visual explanations</li>
        <li>GPT-4o reasoning for prediction interpretation</li>
        <li>Confidence scores and probability visualization</li>
      </ul>

      <h3>How it works</h3>

      <ol>
        <li>Upload an image or select an image form image button.</li>
        <li>The model predicts whether it is AI-generated or real.</li>
        <li>Grad-CAM highlights influential image regions.</li>
        <li>GPT-4o provides a natural language explanation.</li>
      </ol>

      <p className="modal-note">
        This tool is intended for research and educational purposes. AI
        detection is probabilistic and should not be considered absolute proof
        of authenticity.
      </p>
    </div>
  </div>
)}
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
      </>
  );
}

export default App;