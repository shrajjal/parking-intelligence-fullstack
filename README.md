# AI-driven Parking Hotspot Intelligence — Full Stack Project

This ZIP contains a complete runnable full-stack project for the Gridlock Hackathon theme:

> AI-driven parking intelligence to detect illegal parking hotspots and prioritize enforcement.

It includes:

- **ML pipeline:** DBSCAN + CatBoost Regressor
- **FastAPI backend:** serves predictions and dashboard data to React
- **React frontend:** dashboard with filters, charts, hotspot table, and map
- **Optional Mappls/MapmyIndia live traffic module:** backend-only integration with safe proxy fallback

---

## Project Structure

```text
parking-intelligence-fullstack/
├── backend/
│   ├── data/
│   │   └── jan to may police violation_anonymized791b166.csv
│   ├── models/
│   ├── outputs/
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   ├── traffic_service.py
│   │   └── utils.py
│   ├── api.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── .env.example
│
└── README.md
```

---

## Backend Setup

Open terminal inside `backend/`:

```bash
cd backend
python -m venv venv
```

Activate virtual environment on Windows:

```bash
venv\Scripts\activate
```

Activate virtual environment on Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train ML Model

From inside `backend/`, run:

```bash
python src/train_model.py
```

This will:

- clean the dataset
- create DBSCAN hotspots from latitude-longitude
- aggregate records by hotspot + date + hour
- add lag features
- train CatBoost Regressor
- save model files inside `models/`
- save metrics inside `outputs/model_metrics.txt`

---

## Generate Predictions

```bash
python src/predict.py
```

This creates:

```text
outputs/hotspot_predictions.csv
```

Prediction output includes:

```text
hotspot_id
junction_name
police_station
predicted_violation_count
risk_level
priority_score
congestion_impact_score
final_enforcement_score
recommended_enforcement_time
traffic_data_source
```

---

## Run FastAPI Backend

From inside `backend/`, run:

```bash
uvicorn api:app --reload --port 8000
```

Open API docs:

```text
http://localhost:8000/docs
```

Important endpoints:

```text
GET  /api/summary
GET  /api/hotspots
GET  /api/filter-options
GET  /api/hourly-trend
GET  /api/top-stations
GET  /api/top-junctions
GET  /api/vehicle-types
GET  /api/metrics
POST /api/predict/regenerate
```

---

## Optional Mappls / MapmyIndia Traffic API

Create `.env` inside `backend/` using `.env.example`:

```env
MAPPLS_ACCESS_TOKEN=your_mappls_access_token_here
MAPPLS_ROUTE_BASE_URL=https://route.mappls.com/route/direction
```

If token is empty, the app still works using proxy congestion score.

React never stores the Mappls key. It only calls FastAPI.

---

## Frontend Setup

Open a second terminal inside `frontend/`:

```bash
cd frontend
npm install
```

Create `.env` using `.env.example`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Run React:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## Recommended Run Order

Terminal 1:

```bash
cd backend
venv\Scripts\activate
python src/train_model.py
python src/predict.py
uvicorn api:app --reload --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

---

## Theme Alignment

This project solves:

1. Detect illegal parking hotspots using DBSCAN.
2. Predict future parking violation intensity using CatBoost.
3. Estimate congestion impact using proxy score and optional Mappls live traffic API.
4. Rank hotspots for targeted traffic police enforcement.
5. Display everything in a React dashboard.

---

## Presentation Line

> Our system predicts illegal parking hotspot intensity using historical police violation data, enriches it with congestion impact scoring, and ranks enforcement zones with recommended patrol time windows for traffic police.
