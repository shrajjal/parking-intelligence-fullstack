// Add this file in your React frontend as:
// src/services/llmApi.js

const LLM_API_BASE_URL = import.meta.env.VITE_LLM_API_URL || "http://127.0.0.1:8001";

export async function explainHotspotWithAI(hotspot) {
  const response = await fetch(`${LLM_API_BASE_URL}/api/explain-hotspot`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(hotspot),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to generate AI explanation");
  }

  return response.json();
}

export async function generateEnforcementReport(reportPayload) {
  const response = await fetch(`${LLM_API_BASE_URL}/api/generate-report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(reportPayload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to generate AI report");
  }

  return response.json();
}
