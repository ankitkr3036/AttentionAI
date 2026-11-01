# AttentionAI
Real-Time Attention and Drowsiness Detection System

AttentionAI is an educational computer-vision prototype. It uses a locally attached webcam, OpenCV and MediaPipe Face Mesh to estimate eye closure, blinks, approximate gaze and approximate head orientation. Flask serves a dashboard and APIs; SQLite stores session summaries.

> **Important limitations:** The attention score and classifications are heuristic demonstrations, not scientifically validated concentration measurements or medical diagnoses. Do not use this prototype for medical, employment, grading or safety-critical decisions. Head pose and gaze estimates are approximate and require calibration for meaningful use.

## Features
- Local webcam capture with annotated face/eye landmarks
- Eye Aspect Ratio (EAR) and blink counting
- Approximate gaze and head-orientation estimates
- Focused, Distracted, Drowsy and No face detected states
- Optional browser audio warning (muted by default)
- Session timer, focus streak, attention timeline and history
- SQLite persistence and CSV export
- Unit tests that do not need a webcam

## Requirements
- Windows 10/11, macOS or Linux
- Python 3.11 recommended (64-bit)
- Webcam for live detection
- Modern browser

## Windows PowerShell setup
```powershell
cd path\to\AttentionAI
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## macOS / Linux setup
```bash
cd /path/to/AttentionAI
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Run tests
```bash
pytest -q
```
Tests use synthetic points and a temporary SQLite database. They do not open a webcam.

## How to use
1. Start the Flask server.
2. Open the local URL in your browser.
3. Click **Start session**. The operating system may ask for camera permission.
4. Keep your face visible and reasonably well lit.
5. Click **Stop** to end and save the session.
6. Use **Export session data (CSV)** to download saved samples.

The webcam is opened by the Python process on the same computer. The web page is a dashboard for that local process. The application does not save camera frames or video.

## API examples
`GET /api/status`
```json
{
  "status": "Focused",
  "attention_score": 90,
  "ear": 0.29,
  "blink_count": 3,
  "gaze": "Center",
  "head_pose": "Forward",
  "face_detected": true,
  "camera_available": true,
  "session_active": true
}
```
Values above are illustrative. Actual fields also include session duration, focus streak, alert, message and server timestamp.

`GET /api/history`
```json
{"sessions": []}
```

Other routes:
- `POST /api/start`
- `POST /api/stop`
- `POST /api/reset`
- `POST /api/history/clear`
- `GET /video_feed`
- `GET /export.csv`
- `GET /health`

## Configuration
- `ATTENTIONAI_DB`: optional SQLite database path. Defaults to `data/attentionai.db`.
- `ATTENTIONAI_SAVE_INTERVAL`: seconds between stored samples. Defaults to `2`.
- `PORT`: local server port. Defaults to `5000`.
- EAR threshold and prolonged-closure threshold can be adjusted in `AttentionDetector` in `app.py` / `detector.py`.

## Troubleshooting
**Camera unavailable**
- Check Windows Settings → Privacy & security → Camera and allow desktop apps.
- Close Teams, Zoom, browser tabs or other apps that may be using the webcam.
- Confirm the webcam works in the Windows Camera app.
- Try changing `camera_index=0` to `camera_index=1` when constructing `CameraWorker` in `app.py`.

**MediaPipe install error**
- Use 64-bit Python 3.11.
- Create a fresh virtual environment and upgrade pip.
- Install the pinned requirements from this project.
- Avoid mixing packages from multiple Python installations. Verify with `python --version` and `python -m pip --version`.

**No face detected**
- Improve lighting, face the camera and move closer.
- Avoid strong backlighting or covering the eyes.

**Audio does not play**
- Audio is muted by default. Uncheck “Mute audio alerts”.
- Browser audio may require a user interaction first.

**Port already in use**
```powershell
python app.py
```
Change the port by setting `PORT=5001` before running.

## Architecture
```text
Web browser
  └── HTML/CSS/JavaScript dashboard
       ├── polls Flask /api/status and /api/history
       ├── displays Flask /video_feed MJPEG stream
       └── plays optional browser-side audio alert
             |
             v
Flask app.py
  ├── CameraWorker
  │    └── OpenCV webcam -> MediaPipe Face Mesh -> AttentionDetector
  └── Database
       └── SQLite session and sample tables
```

## Deployment: local vs cloud
**Local deployment (implemented):** Run `python app.py` on the computer that owns the webcam. The Python process can open that computer's camera.

**Cloud deployment (not implemented by this local-camera build):** A cloud server cannot open the webcam attached to a visitor's laptop. A hosted version must move capture to browser JavaScript using `getUserMedia()` and run compatible face/eye processing in the browser (for example, MediaPipe Tasks Vision), or send only minimal consented derived data to an authenticated backend. Do not expose a public webcam stream or deploy this server-side camera worker as if it can access a visitor's camera.

Render's default filesystem may be ephemeral. For hosted persistent records, use a managed database such as PostgreSQL or a supported persistent disk. Do not store private webcam images or video.

## Privacy
- Camera frames are processed in memory on the local computer.
- No webcam images or video are written to disk by default.
- SQLite stores timestamped numerical/status summaries.
- Delete local history using the dashboard's **Clear history** button, or remove `data/attentionai.db` while the server is stopped.
- Do not use the prototype to make consequential decisions about people.

## Future improvements
- Browser-side MediaPipe Tasks implementation for hosted use
- Per-user calibration for gaze and EAR thresholds
- More robust 3-D head-pose estimation
- Better session analytics and configurable alert thresholds
- Authentication, CSRF protection, rate limiting and managed database for multi-user hosting
