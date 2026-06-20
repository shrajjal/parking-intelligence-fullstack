import { useState } from "react";
import { explainHotspotWithAI } from "./api";

export default function AiExplanationCard({ hotspot, onClose }) {
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [error, setError] = useState("");

  const handleExplain = async () => {
    if (!hotspot) return;

    try {
      setLoading(true);
      setError("");
      setExplanation("");

      const data = await explainHotspotWithAI(hotspot);
      setExplanation(data.explanation);
    } catch (err) {
      setError(err.message || "Something went wrong while generating explanation");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ai-card">
      <button className="ai-close-btn" onClick={onClose}>
        ×
      </button>

      <div className="ai-card-header">
        <div>
          <h3>AI Enforcement Assistant</h3>
          <p>Gemini explanation for selected hotspot</p>
        </div>

        <button onClick={handleExplain} disabled={loading || !hotspot}>
          {loading ? "Generating..." : "Explain with AI"}
        </button>
      </div>

      {error && <div className="ai-error">{error}</div>}

      {explanation && (
        <div className="ai-output">
          {explanation}
        </div>
      )}
    </div>
  );
}