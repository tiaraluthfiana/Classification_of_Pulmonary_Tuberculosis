# Classification of Pulmonary Tuberculosis Using a Hybrid CNN

A deep learning system that classifies chest X-ray images as **Normal** or **Tuberculosis (TB)**, served through a Flask web app and a JSON REST API. The research was presented at **IBITEC 2026** as *"Classification of Pulmonary Tuberculosis Using Hybrid CNN"*.

> **Disclaimer:** This project is a research and portfolio prototype for decision support. It is not a medical device and must not replace diagnosis by a qualified medical professional.

## Screenshots

| Upload | Prediction result |
|---|---|
| ![Upload page](docs/screenshot-upload.png) | ![Prediction result](docs/screenshot-result.png) |

## Highlights

- Hybrid **CNN + self-attention** model for early TB detection from chest X-rays.
- **99.76% accuracy, 98.59% recall, 100% precision** on the test set.
- Web app with drag-and-drop upload, prediction, confidence score, and probability chart.
- **REST API** (`POST /api/predict`) that returns predictions as JSON.
- **Prediction history** stored in a database (SQLite locally, PostgreSQL with Docker).
- Containerized with **Docker Compose** (web app + PostgreSQL).

## Model

| Item | Detail |
|---|---|
| Task | Binary classification: Normal vs Tuberculosis |
| Dataset | TB Chest X-ray Database, 4,200 images in total |
| Architecture | Hybrid CNN (ResNet-50 backbone) with a self-attention mechanism |
| Input | RGB image, resized to 512 x 512 and scaled to [0, 1] |
| Class imbalance | Class weighting 1:3 (limited TB data) |
| Decision threshold | 0.569 (tuned), applied to the TB probability |

| Metric | Result |
|---|---|
| Accuracy | 99.76% |
| Recall | 98.59% |
| Precision | 100% |

The goal is early TB detection, so recall (not missing TB cases) is tracked closely alongside accuracy and precision.

## How it works

```
Upload X-ray (PNG/JPG) -> Resize 512x512 + normalize -> Hybrid CNN -> TB probability
      -> threshold 0.569 -> Normal / Tuberculosis + confidence -> saved to database
```

## Tech stack

Python, TensorFlow/Keras, Flask, SQLAlchemy, PostgreSQL, Pillow, NumPy, Chart.js, Gunicorn, Docker.

## Getting started

Make sure the trained model `final_model.keras` is in the project root.

### Option 1: Docker Compose (web app + PostgreSQL)

```bash
docker compose up --build
```

Open <http://localhost:5000>. Stop with `docker compose down`.

### Option 2: Run locally (SQLite)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Set `FLASK_DEBUG=1` to enable auto-reload during development.

## Usage

### Web app

| Route | Description |
|---|---|
| `/` | Upload a chest X-ray and see the prediction, confidence, and probability chart |
| `/history` | Last 100 predictions (time, result, confidence, source) |

### REST API

`POST /api/predict` with a multipart form field named `file` (PNG, JPG, or JPEG, up to 16 MB).

```bash
curl -X POST -F "file=@chest_xray.jpg" http://localhost:5000/api/predict
```

Example response:

```json
{
  "prediction": "Tuberculosis",
  "confidence": 98.2,
  "probabilities": { "Normal": 1.8, "Tuberculosis": 98.2 },
  "threshold": 0.569
}
```

Error responses are JSON with an `error` message: `400` for a missing file or unsupported format, `500` if the image cannot be processed.

## Configuration

All settings are optional environment variables.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite file `predictions.db` | Database connection string (set by `docker-compose.yml` for PostgreSQL) |
| `MODEL_PATH` | `final_model.keras` in project root | Path to the trained model |
| `THRESHOLD` | `0.569` | Decision threshold for the TB class |
| `IMG_SIZE` | `512` | Model input size |
| `FLASK_DEBUG` | off | Set to `1` for development mode |

## Project structure

```
.
├── app.py                  # Flask app: web routes, REST API, model inference, database
├── final_model.keras       # Trained hybrid CNN model
├── requirements.txt
├── Dockerfile
├── docker-compose.yml      # Web app + PostgreSQL
├── templates/
│   ├── index.html          # Upload and result page
│   └── history.html        # Prediction history
└── static/
    ├── style.css
    ├── script.js
    └── uploads/            # Uploaded images (not committed)
```

## Limitations

- Trained and evaluated on a single public dataset; performance on images from other hospitals, devices, or populations is not validated.
- The model outputs a probability, not a diagnosis. Results should be reviewed by a medical professional.
- Uploaded images and prediction history are stored for demonstration purposes; do not upload real patient data or identifying information.

## Author

**Tiara Luthfiana Puspa Marizka**, Informatics, Telkom University Surabaya
[LinkedIn](https://www.linkedin.com/in/tiaraluthfianaa) | [GitHub](https://github.com/tiaraluthfianaa)
