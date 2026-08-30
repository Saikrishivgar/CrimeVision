# CrimeVision Suite

This is the unified workspace for **CrimeVision**, containing the custom YOLO model training setup, the FastAPI backend video analysis server, and the Next.js surveillance dashboard.

---

## Directory Overview

- **[CrimeVision-YOLO/](file:///Users/saikrishivgars/Desktop/YOLOO/CrimeVision-YOLO/)**: Dedicated workspace for YOLO custom training (`train.py`, `detect.py`, `inference.py` engine, and local dataset folders).
- **[CrimeVision-backend/](file:///Users/saikrishivgars/Desktop/YOLOO/CrimeVision-backend/)**: FastAPI backend application (`main.py`) which acts as the API endpoint for uploading videos, running frame-by-frame analysis via the YOLO engine, and serving processed streams.
- **[crimevision-frontend/](file:///Users/saikrishivgars/Desktop/YOLOO/crimevision-frontend/)**: Next.js SPA dashboard styled in dark-theme surveillance console aesthetics, providing dragging upload dropzones, scanning overlays, stats widgets, filterable timelines, and LLM incident report tools.

---

## How to Run the Applications

### 1. Start the FastAPI Backend Server
The backend server runs on Python 3 and integrates directly with the virtual environment created under `CrimeVision-YOLO`.

Open a new terminal tab/window and run:
```bash
# Navigate to backend directory
cd CrimeVision-backend

# Activate the shared virtual environment
source ../CrimeVision-YOLO/venv/bin/activate

# Start the uvicorn development server
python3 main.py
```
The server will start and be available at: **`http://localhost:8000`**

---

### 2. Start the Next.js Frontend Dashboard
The frontend runs on Node.js and is built using Next.js and Tailwind CSS v4.

Open another terminal tab/window and run:
```bash
# Navigate to the frontend directory
cd crimevision-frontend

# Start the Next.js development server
npm run dev
```
The dashboard will start and be available in your browser at: **`http://localhost:3000`**

---

## Verifying the Flow
1. Open `http://localhost:3000` in your web browser.
2. Drag and drop any surveillance video file (e.g. an `.mp4` file) into the upload dropzone.
3. Click **Analyze**.
4. The dashboard will show animated loading and scanning states as it progresses through YOLO detection, ByteTrack tracking, attribute classification, and temporal mode voting.
5. Once completed, the annotated video player will load, summary count cards will fill, and the interactive **Surveillance Event Log** timeline will let you search, filter, and inspect objects, track IDs, and descriptions (e.g. clothing colors).
6. Click **Generate AI Incident Report** to see the compiled security report.
