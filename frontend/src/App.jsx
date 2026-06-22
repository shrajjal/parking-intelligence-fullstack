import { useEffect, useMemo, useState } from 'react';
// import { useEffect, useState } from "react";
import AiExplanationCard from './AiExplanationCard';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CircleMarker, MapContainer, Popup, TileLayer } from 'react-leaflet';
import {
  fetchFilterOptions,
  fetchHourlyTrend,
  fetchHotspots,
  fetchMetrics,
  fetchSummary,
  fetchTopJunctions,
  fetchTopStations,
  fetchVehicleTypes,
  regeneratePredictions,
} from './api.js';

const pieColors = [
  '#2563eb',
  '#16a34a',
  '#f97316',
  '#dc2626',
  '#7c3aed',
  '#0891b2',
  '#64748b',
];

const safeNumber = (value, fallback = 0) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const isValidCoordinate = (value) => {
  const num = Number(value);
  return Number.isFinite(num) && num !== 0;
};

const formatTrafficSource = (source) => {
  if (source === 'tomtom_live') return 'Live';
  if (source === 'proxy') return 'Proxy';
  return source || 'Proxy';
};

const getMapLatitude = (row) => {
  if (
    row.traffic_data_source === 'tomtom_live' &&
    isValidCoordinate(row.road_latitude)
  ) {
    return safeNumber(row.road_latitude);
  }

  return safeNumber(row.latitude_center);
};

const getMapLongitude = (row) => {
  if (
    row.traffic_data_source === 'tomtom_live' &&
    isValidCoordinate(row.road_longitude)
  ) {
    return safeNumber(row.road_longitude);
  }

  return safeNumber(row.longitude_center);
};

const getMapPointType = (row) => {
  if (
    row.traffic_data_source === 'tomtom_live' &&
    isValidCoordinate(row.road_latitude) &&
    isValidCoordinate(row.road_longitude)
  ) {
    return 'Nearest road segment';
  }

  return 'Hotspot center';
};

const getTrafficModeLabel = (liveTrafficEnabled, hotspots) => {
  const hasTomTom = hotspots?.some(
    (row) => row.traffic_data_source === 'tomtom_live'
  );

  if (liveTrafficEnabled && hasTomTom) {
    return 'Live Traffic Mode';
  }

  if (liveTrafficEnabled && !hasTomTom) {
    return 'Live Traffic Requested: Proxy Fallback';
  }

  return 'Proxy Congestion Mode';
};

const calculateTrafficStats = (hotspots = []) => {
  if (!hotspots.length) {
    return {
      avgDelayRatio: 1,
      avgTrafficScore: 0,
      avgCurrentSpeed: 0,
      avgFreeFlowSpeed: 0,
      tomTomCount: 0,
      proxyCount: 0,
    };
  }

  const tomTomRows = hotspots.filter(
    (row) => row.traffic_data_source === 'tomtom_live'
  );

  const rowsForStats = tomTomRows.length ? tomTomRows : hotspots;

  const avg = (key) =>
    rowsForStats.reduce((sum, row) => sum + safeNumber(row[key]), 0) /
    rowsForStats.length;

  return {
    avgDelayRatio: avg('traffic_delay_ratio'),
    avgTrafficScore: avg('live_traffic_score'),
    avgCurrentSpeed: avg('current_speed_kmph'),
    avgFreeFlowSpeed: avg('free_flow_speed_kmph'),
    tomTomCount: tomTomRows.length,
    proxyCount: hotspots.length - tomTomRows.length,
  };
};

const getMarkerColor = (row, liveTrafficEnabled) => {
  const source = row.traffic_data_source;
  const finalScore = safeNumber(row.final_enforcement_score);
  const liveScore = safeNumber(row.live_traffic_score);

  if (liveTrafficEnabled && source === 'tomtom_live') {
    if (liveScore >= 70) return '#dc2626';
    if (liveScore >= 40) return '#f97316';
    return '#16a34a';
  }

  if (finalScore >= 80) return '#dc2626';
  if (finalScore >= 60) return '#f97316';
  return '#16a34a';
};

function formatNumber(value) {
  if (value === undefined || value === null || value === '') return '-';
  return Number(value).toLocaleString('en-IN');
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <p>{label}</p>
      <h2>{formatNumber(value)}</h2>
    </div>
  );
}

function BarPanel({ title, data, dataKey = 'violations' }) {
  const chartHeight = Math.max(330, data.length * 32);

  const shortName = (name) => {
    if (!name) return '';
    return name.length > 28 ? name.slice(0, 28) + '...' : name;
  };

  const chartData = data.map((item) => ({
    ...item,
    shortName: shortName(item.name),
  }));

  return (
    <div className="panel chart-panel">
      <h3>{title}</h3>

      <div style={{ width: '100%', height: chartHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, bottom: 10, left: 90 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis
              type="category"
              dataKey="shortName"
              width={150}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(value) => [value, 'Violations']}
              labelFormatter={(label, payload) => {
                if (payload && payload.length > 0) {
                  return payload[0].payload.name;
                }
                return label;
              }}
            />
            <Bar dataKey={dataKey} fill="#2563eb" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function HotspotMap({ hotspots, liveTraffic }) {
  const validHotspots = useMemo(
    () =>
      hotspots.filter(
        (h) =>
          Number.isFinite(getMapLatitude(h)) &&
          Number.isFinite(getMapLongitude(h))
      ),
    [hotspots]
  );

  const center = useMemo(() => {
    if (!validHotspots.length) return [12.9716, 77.5946];

    const lat =
      validHotspots.reduce((sum, h) => sum + getMapLatitude(h), 0) /
      validHotspots.length;

    const lon =
      validHotspots.reduce((sum, h) => sum + getMapLongitude(h), 0) /
      validHotspots.length;

    return [lat, lon];
  }, [validHotspots]);

  return (
    <div className="panel">
      <h3>Hotspot Map</h3>

      <MapContainer
        center={center}
        zoom={12}
        scrollWheelZoom
        className="map-container"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {validHotspots.map((h, index) => {
          const color = getMarkerColor(h, liveTraffic);
          const radius = Math.max(
            5,
            Math.min(
              20,
              safeNumber(h.final_enforcement_score, h.priority_score || 20) / 5
            )
          );

          return (
            <CircleMarker
              key={`${h.hotspot_id}-${h.recommended_enforcement_time}-${index}`}
              center={[getMapLatitude(h), getMapLongitude(h)]}
              radius={radius}
              pathOptions={{
                color,
                fillColor: color,
                fillOpacity: 0.75,
                weight: h.traffic_data_source === 'tomtom_live' ? 3 : 2,
              }}
            >
              <Popup>
                <div className="map-popup">
                  <b>{h.hotspot_id}</b>

                  <p>{h.junction_name}</p>

                  <p>
                    <strong>Police Station:</strong> {h.police_station}
                  </p>

                  <p>
                    <strong>Map Point:</strong> {getMapPointType(h)}
                  </p>

                  <p>
                    <strong>Predicted Count:</strong>{' '}
                    {h.predicted_violation_count}
                  </p>

                  <p>
                    <strong>Risk:</strong> {h.risk_level}
                  </p>

                  <p>
                    <strong>Priority:</strong> {h.priority_score}
                  </p>

                  <p>
                    <strong>Congestion Impact:</strong>{' '}
                    {h.congestion_impact_score}
                  </p>

                  <p>
                    <strong>Final Score:</strong> {h.final_enforcement_score}
                  </p>

                  <p>
                    <strong>Time:</strong> {h.recommended_enforcement_time}
                  </p>

                  <p>
                    <strong>Traffic Source:</strong>{' '}
                    {formatTrafficSource(h.traffic_data_source)}
                  </p>

                  <p>
                    <strong>Current Speed:</strong>{' '}
                    {safeNumber(h.current_speed_kmph).toFixed(1)} km/h
                  </p>

                  <p>
                    <strong>Free Flow Speed:</strong>{' '}
                    {safeNumber(h.free_flow_speed_kmph).toFixed(1)} km/h
                  </p>

                  <p>
                    <strong>Delay Ratio:</strong>{' '}
                    {safeNumber(h.traffic_delay_ratio, 1).toFixed(2)}x
                  </p>

                  <p>
                    <strong>Live Traffic Score:</strong>{' '}
                    {safeNumber(h.live_traffic_score).toFixed(2)}
                  </p>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}

function HotspotTable({ hotspots }) {
  const [selectedAiHotspot, setSelectedAiHotspot] = useState(null);

  return (
    <div className="panel table-panel">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Hotspot</th>
              <th>Junction</th>
              <th>Police Station</th>
              <th>Predicted Count</th>
              <th>Risk</th>
              <th>Priority</th>
              <th>Congestion Impact</th>
              <th>Final Score</th>
              <th>Current Speed</th>
              <th>Free Flow Speed</th>
              <th>Delay Ratio</th>
              <th>Live Traffic Score</th>
              <th>Time</th>
              <th>Traffic Source</th>
              <th>AI</th>
            </tr>
          </thead>

          <tbody>
            {hotspots.map((h, index) => (
              <tr
                key={`${h.hotspot_id}-${h.recommended_enforcement_time}-${index}`}
              >
                <td>{h.hotspot_id}</td>
                <td>{h.junction_name}</td>
                <td>{h.police_station}</td>
                <td>{h.predicted_violation_count}</td>
                <td>
                  <span
                    className={`risk-badge ${String(
                      h.risk_level
                    ).toLowerCase()}`}
                  >
                    {h.risk_level}
                  </span>
                </td>
                <td>{h.priority_score}</td>
                <td>{h.congestion_impact_score}</td>
                <td>
                  <strong>{h.final_enforcement_score}</strong>
                </td>
                <td>{safeNumber(h.current_speed_kmph).toFixed(1)} km/h</td>
                <td>{safeNumber(h.free_flow_speed_kmph).toFixed(1)} km/h</td>
                <td>{safeNumber(h.traffic_delay_ratio, 1).toFixed(2)}x</td>
                <td>{safeNumber(h.live_traffic_score).toFixed(2)}</td>
                <td>{h.recommended_enforcement_time}</td>
                <td>
                  <span
                    className={`traffic-source-badge ${
                      h.traffic_data_source === 'tomtom_live'
                        ? 'tomtom'
                        : 'proxy'
                    }`}
                  >
                    {formatTrafficSource(h.traffic_data_source)}
                  </span>
                </td>
                <td>
                  <button
                    className="ai-table-btn"
                    onClick={() => setSelectedAiHotspot(h)}
                  >
                    Explain
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedAiHotspot && (
  <AiExplanationCard
    hotspot={selectedAiHotspot}
    onClose={() => setSelectedAiHotspot(null)}
  />
)}
    </div>
  );
}

export default function App() {


  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };
  const [summary, setSummary] = useState(null);
  const [stations, setStations] = useState([]);
  const [junctions, setJunctions] = useState([]);
  const [hourly, setHourly] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [filterOptions, setFilterOptions] = useState({
    police_stations: [],
    junctions: [],
    risk_levels: [],
    hours: [],
  });
  const [hotspots, setHotspots] = useState([]);
  const [metrics, setMetrics] = useState('');
  const [loading, setLoading] = useState(true);
  const [hotspotLoading, setHotspotLoading] = useState(false);
  const [error, setError] = useState('');

  const [filters, setFilters] = useState({
    policeStation: '',
    junctionName: '',
    riskLevel: '',
    hour: '',
    liveTraffic: false,
    limit: 100,
  });

  const trafficStats = calculateTrafficStats(hotspots);
  const trafficModeLabel = getTrafficModeLabel(filters.liveTraffic, hotspots);

  async function loadDashboard() {
    try {
      setError('');
      setLoading(true);

      const [
        summaryRes,
        stationRes,
        junctionRes,
        hourlyRes,
        vehicleRes,
        filterRes,
        metricRes,
      ] = await Promise.all([
        fetchSummary(),
        fetchTopStations(),
        fetchTopJunctions(),
        fetchHourlyTrend(),
        fetchVehicleTypes(),
        fetchFilterOptions().catch(() => ({
          police_stations: [],
          junctions: [],
          risk_levels: [],
          hours: [],
        })),
        fetchMetrics().catch(() => ({ metrics: '' })),
      ]);

      setSummary(summaryRes);
      setStations(stationRes.data || []);
      setJunctions(junctionRes.data || []);
      setHourly(hourlyRes.data || []);
      setVehicles(vehicleRes.data || []);
      setFilterOptions(filterRes);
      setMetrics(metricRes.metrics || metricRes.message || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadHotspots(nextFilters = filters) {
    try {
      setError('');
      setHotspotLoading(true);

      const res = await fetchHotspots(nextFilters);
      setHotspots(res.data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setHotspotLoading(false);
    }
  }

  async function handleRegenerate() {
    try {
      setError('');
      setHotspotLoading(true);

      await regeneratePredictions();
      await loadHotspots(filters);
    } catch (err) {
      setError(err.message);
    } finally {
      setHotspotLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard().then(() => loadHotspots(filters));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function updateFilter(key, value) {
    const next = { ...filters, [key]: value };
    setFilters(next);
    loadHotspots(next);
  }

  if (loading) {
    return (
      <div className="loading">
        Loading parking intelligence dashboard...
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h2>Prediction Filters</h2>

        <label>Police Station</label>
        <select
          value={filters.policeStation}
          onChange={(e) => updateFilter('policeStation', e.target.value)}
        >
          <option value="">All</option>
          {filterOptions.police_stations?.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>

        <label>Junction</label>
        <select
          value={filters.junctionName}
          onChange={(e) => updateFilter('junctionName', e.target.value)}
        >
          <option value="">All</option>
          {filterOptions.junctions?.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>

        <label>Risk Level</label>
        <select
          value={filters.riskLevel}
          onChange={(e) => updateFilter('riskLevel', e.target.value)}
        >
          <option value="">All</option>
          {filterOptions.risk_levels?.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>

        <label>Hour</label>
        <select
          value={filters.hour}
          onChange={(e) => updateFilter('hour', e.target.value)}
        >
          <option value="">All</option>
          {filterOptions.hours?.map((item) => (
            <option key={item} value={item}>
              {item}:00
            </option>
          ))}
        </select>

        {/* <label className="checkbox-label">
          <input
            type="checkbox"
            checked={filters.liveTraffic}
            onChange={(e) => updateFilter('liveTraffic', e.target.checked)}
          />
          Enable live traffic
        </label> */}

        <button
          className="secondary-btn"
          onClick={() => loadHotspots(filters)}
          disabled={hotspotLoading}
        >
          {hotspotLoading ? 'Loading...' : 'Refresh Hotspots'}
        </button>
      </aside>

      <main className="main-content">
        <header className="hero">
          <div>
            <p className="eyebrow">Gridlock Hackathon 2.0</p>
            <h1>AI-driven Parking Hotspot Intelligence</h1>
            <p>
              Detect illegal parking hotspots, predict future violation risk,
              and prioritize enforcement with traffic impact scoring.
            </p>
          </div>
          <div className="hero-actions">
  <button className="theme-toggle-btn" onClick={toggleTheme}>
    {theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode"}
  </button>

  <button
    className="primary-btn"
    onClick={handleRegenerate}
    disabled={hotspotLoading}
  >
    {hotspotLoading ? "Processing..." : "Regenerate Predictions"}
  </button>
</div>
        </header>

        {error && <div className="error-box">{error}</div>}

        {summary && (
          <section className="stats-grid">
            <StatCard
              label="Total Violations"
              value={summary.total_violations}
            />
            <StatCard
              label="Police Stations"
              value={summary.police_stations}
            />
            <StatCard label="Junctions" value={summary.junctions} />
            <StatCard label="Vehicle Types" value={summary.vehicle_types} />
          </section>
        )}

        <section className="two-col-grid">
          <BarPanel
            title="Top Police Stations by Violations"
            data={stations}
          />
          <BarPanel title="Top Junctions by Violations" data={junctions} />
        </section>

        <section className="two-col-grid">
          <div className="panel chart-panel">
            <h3>Vehicle Type Distribution</h3>
            <ResponsiveContainer width="100%" height={320}>
              <PieChart>
                <Pie
                  data={vehicles}
                  dataKey="violations"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={105}
                  label
                >
                  {vehicles.map((_, index) => (
                    <Cell
                      key={index}
                      fill={pieColors[index % pieColors.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="panel chart-panel">
            <h3>Hour-wise Violation Trend (IST)</h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart
                data={hourly}
                margin={{ top: 10, right: 20, bottom: 10, left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="violations"
                  stroke="#2563eb"
                  strokeWidth={3}
                  dot
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="section-header">
          <div>
            <h2>Predicted Hotspot Enforcement Priority</h2>
            <p>
              Ranked by final enforcement score = parking priority + congestion
              impact.
            </p>
          </div>

          <span className="pill">{hotspots.length} hotspots</span>
        </section>

        <section className="traffic-mode-panel">
          <div
            className={`traffic-mode-badge ${
              filters.liveTraffic ? 'live' : 'proxy'
            }`}
          >
            {trafficModeLabel}
          </div>

          <div className="traffic-stat-grid">
            <div className="traffic-stat-card">
              <span>Traffic Source</span>
              <strong>
                {trafficStats.tomTomCount > 0 ? 'Live' : 'Proxy'}
              </strong>
              <small>
                {trafficStats.tomTomCount > 0
                  ? `${trafficStats.tomTomCount} hotspots enriched`
                  : 'Using fallback congestion score'}
              </small>
            </div>

            <div className="traffic-stat-card">
              <span>Average Delay Ratio</span>
              <strong>{trafficStats.avgDelayRatio.toFixed(2)}x</strong>
              <small>Current travel time vs normal</small>
            </div>

            <div className="traffic-stat-card">
              <span>Average Live Traffic Score</span>
              <strong>{trafficStats.avgTrafficScore.toFixed(2)}</strong>
              <small>Higher means more congestion</small>
            </div>

            <div className="traffic-stat-card">
              <span>Average Speed</span>
              <strong>
                {trafficStats.avgCurrentSpeed.toFixed(1)} /{' '}
                {trafficStats.avgFreeFlowSpeed.toFixed(1)} km/h
              </strong>
              <small>Current speed / free-flow speed</small>
            </div>
          </div>
        </section>

        <HotspotTable hotspots={hotspots} />

        <HotspotMap
          hotspots={hotspots.slice(0, 200)}
          liveTraffic={filters.liveTraffic}
        />

        {metrics && (
          <div className="panel metrics-panel">
            <h3>Model Metrics</h3>
            <pre>{metrics}</pre>
          </div>
        )}
      </main>
    </div>
  );
}