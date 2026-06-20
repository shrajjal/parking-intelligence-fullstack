# Gemini LLM Backend for Parking Intelligence

This is a separate FastAPI backend for adding Gemini-powered explanations to your parking intelligence project.

Your main ML backend will still do:

```text
DBSCAN hotspot detection
CatBoost violation count prediction
TomTom live traffic enrichment
```

This Gemini backend only does:

```text
Explain hotspot risk
Generate enforcement recommendations
Generate daily enforcement summary
```

---

## Folder Structure

```text
gemini-llm-backend/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_service.py
│   ├── main.py
│   └── schemas.py
│
├── frontend-snippets/
│   ├── api.js
│   ├── AiExplanationCard.jsx
│   └── .env.frontend.example
│
├── .env.example
├── requirements.txt
├── run.bat
└── README.md
```

---

## 1. Setup Backend

Open terminal inside this folder:

```bash
cd gemini-llm-backend
```

Create virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Add Gemini API Key

Create a `.env` file by copying `.env.example`:

```bash
copy .env.example .env
```

For Mac/Linux:

```bash
cp .env.example .env
```

Open `.env` and add your key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000
MOCK_LLM=false
```

Do not put the Gemini API key in React frontend.

---

## 3. Run Backend

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open this in browser:

```text
http://127.0.0.1:8001/docs
```

---

## 4. Test Endpoint

Use `/api/explain-hotspot` from FastAPI docs with this sample JSON:

```json
{
  "hotspot_id": 4,
  "location": "Sakchi Junction",
  "police_station": "Sakchi",
  "junction_name": "Sakchi Roundabout",
  "hour": 18,
  "day_of_week": "Monday",
  "month": 6,
  "predicted_violation_count": 22,
  "risk_score": 0.86,
  "risk_level": "High",
  "recommended_time": "6 PM - 8 PM",
  "current_speed": 14,
  "free_flow_speed": 48,
  "congestion_score": 0.72,
  "traffic_delay": "High",
  "road_closure": false,
  "common_vehicle_type": "Two-wheeler",
  "common_violation_type": "No Parking"
}
```

Expected response:

```json
{
  "hotspot_id": 4,
  "explanation": "Risk Explanation:\n..."
}
```

---

## 5. Mock Mode

If you do not want to call Gemini while testing, set this in `.env`:

```env
MOCK_LLM=true
```

Then the backend will return demo explanations without using your API key.

---

## 6. React Frontend Integration

Copy this file:

```text
frontend-snippets/api.js
```

into your React project as:

```text
src/services/llmApi.js
```

Copy this file:

```text
frontend-snippets/AiExplanationCard.jsx
```

into your React project as:

```text
src/components/AiExplanationCard.jsx
```

Add this to your React frontend `.env` file:

```env
VITE_LLM_API_URL=http://127.0.0.1:8001
```

Restart React after changing `.env`.

---

## 7. Where to Add in Frontend

In the component where you show selected hotspot details, import the AI card:

```jsx
import AiExplanationCard from "./components/AiExplanationCard";
```

If your file is inside `src/pages/Dashboard.jsx`, use:

```jsx
import AiExplanationCard from "../components/AiExplanationCard";
```

Then add it below your hotspot details card:

```jsx
{selectedHotspot && (
  <AiExplanationCard hotspot={selectedHotspot} />
)}
```

Example:

```jsx
function Dashboard() {
  const [selectedHotspot, setSelectedHotspot] = useState(null);

  return (
    <div className="space-y-6">
      <HotspotMap onHotspotClick={setSelectedHotspot} />

      {selectedHotspot && (
        <>
          <HotspotDetails hotspot={selectedHotspot} />
          <AiExplanationCard hotspot={selectedHotspot} />
        </>
      )}
    </div>
  );
}
```

---

## 8. Important Privacy Rule

Send only aggregated hotspot data to Gemini.

Safe fields:

```text
hotspot_id
location
police_station
junction_name
hour
day_of_week
month
predicted_violation_count
risk_score
risk_level
current_speed
free_flow_speed
congestion_score
vehicle_type
violation_type
```

Do not send:

```text
vehicle_number
device_id
created_by_id
raw challan IDs
personal information
```

---

## 9. How This Fits Your Project

Final architecture:

```text
React Frontend
   │
   ├── Calls ML Backend
   │       ├── hotspot prediction
   │       ├── risk score
   │       └── TomTom traffic data
   │
   └── Calls Gemini LLM Backend
           ├── explain selected hotspot
           └── generate enforcement report
```

The LLM backend should not train the model and should not predict violation count. It should only explain the output of your existing ML system.
