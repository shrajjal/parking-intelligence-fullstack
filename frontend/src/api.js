const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const data = await response.json();
      message = data.detail || data.message || message;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(message);
  }
  return response.json();
}

export function fetchSummary() {
  return request('/api/summary');
}

export function fetchTopStations() {
  return request('/api/top-stations?limit=15');
}

export function fetchTopJunctions() {
  return request('/api/top-junctions?limit=15&include_unmapped=false');
}

export function fetchHourlyTrend() {
  return request('/api/hourly-trend');
}

export function fetchVehicleTypes() {
  return request('/api/vehicle-types?limit=12');
}

export function fetchFilterOptions() {
  return request('/api/filter-options');
}

export function fetchHotspots(filters = {}) {
  const params = new URLSearchParams();
  params.set('limit', filters.limit || '200');
  params.set('live_traffic', filters.liveTraffic ? 'true' : 'false');

  if (filters.policeStation) params.set('police_station', filters.policeStation);
  if (filters.junctionName) params.set('junction_name', filters.junctionName);
  if (filters.riskLevel) params.set('risk_level', filters.riskLevel);
  if (filters.hour !== '' && filters.hour !== undefined && filters.hour !== null) {
    params.set('hour', String(filters.hour));
  }

  return request(`/api/hotspots?${params.toString()}`);
}

export function regeneratePredictions() {
  return request('/api/predict/regenerate', { method: 'POST' });
}

export function fetchMetrics() {
  return request('/api/metrics');
}


const LLM_API_BASE_URL =
  import.meta.env.VITE_LLM_API_URL || "http://127.0.0.1:8001";

export async function explainHotspotWithAI(hotspot) {
  const payload = {
    hotspot_id: hotspot.hotspot_id ?? hotspot.id ?? "selected-hotspot",

    location:
      hotspot.location ||
      hotspot.junction_name ||
      `${hotspot.latitude || hotspot.road_latitude || ""}, ${
        hotspot.longitude || hotspot.road_longitude || ""
      }`,

    police_station: hotspot.police_station || "Unknown police station",
    junction_name: hotspot.junction_name || null,

    hour: hotspot.hour ?? null,
    day_of_week: hotspot.day_of_week ?? null,
    month: hotspot.month ?? null,

    predicted_violation_count:
      hotspot.predicted_violation_count ??
      hotspot.predicted_count ??
      hotspot.violation_count ??
      null,

    risk_score: hotspot.risk_score ?? null,
    risk_level: hotspot.risk_level ?? hotspot.risk ?? null,
    recommended_time: hotspot.recommended_time ?? null,

    current_speed: hotspot.current_speed ?? null,
    free_flow_speed: hotspot.free_flow_speed ?? null,
    congestion_score: hotspot.congestion_score ?? null,
    traffic_delay: hotspot.traffic_delay ?? null,
    road_closure: hotspot.road_closure ?? null,

    common_vehicle_type:
      hotspot.common_vehicle_type || hotspot.vehicle_type || null,

    common_violation_type:
      hotspot.common_violation_type || hotspot.violation_type || null,
  };

  const response = await fetch(`${LLM_API_BASE_URL}/api/explain-hotspot`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to generate AI explanation");
  }

  return response.json();
}