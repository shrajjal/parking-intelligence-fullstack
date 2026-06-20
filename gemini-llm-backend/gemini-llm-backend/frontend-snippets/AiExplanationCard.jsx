// Add this component in your React frontend as:
// src/components/AiExplanationCard.jsx

import { useState } from "react";
import { explainHotspotWithAI } from "../services/llmApi";

export default function AiExplanationCard({ hotspot }) {
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [error, setError] = useState("");

  const handleExplain = async () => {
    try {
      setLoading(true);
      setError("");
      setExplanation("");

      const data = await explainHotspotWithAI({
        hotspot_id: hotspot.hotspot_id,
        location: hotspot.location || hotspot.junction_name || "Unknown location",
        police_station: hotspot.police_station || "Unknown police station",
        junction_name: hotspot.junction_name,
        hour: hotspot.hour,
        day_of_week: hotspot.day_of_week,
        month: hotspot.month,
        predicted_violation_count: hotspot.predicted_violation_count,
        risk_score: hotspot.risk_score,
        risk_level: hotspot.risk_level,
        recommended_time: hotspot.recommended_time,
        current_speed: hotspot.current_speed,
        free_flow_speed: hotspot.free_flow_speed,
        congestion_score: hotspot.congestion_score,
        traffic_delay: hotspot.traffic_delay,
        road_closure: hotspot.road_closure,
        common_vehicle_type: hotspot.common_vehicle_type,
        common_violation_type: hotspot.common_violation_type,
      });

      setExplanation(data.explanation);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-900">AI Enforcement Assistant</h3>
          <p className="text-sm text-slate-500">Gemini-generated explanation for this hotspot</p>
        </div>

        <button
          onClick={handleExplain}
          disabled={loading || !hotspot}
          className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Generating..." : "Explain with AI"}
        </button>
      </div>

      {error && (
        <div className="rounded-xl bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {explanation && (
        <div className="whitespace-pre-line rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-800">
          {explanation}
        </div>
      )}
    </div>
  );
}
