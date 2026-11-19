import base64
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from backend.ml_model import estimate_macs, hardware_details, image_to_png_bytes, model_explanation, preprocess_image_bytes
from backend.settings import ARTIFACT_DIR, MODEL_PATH, METADATA_PATH, SUMMARY_PATH, HISTORY_PATH, EXAMPLES_PATH, STATUS_PATH


app = FastAPI(title="Neuro-Genetic LeNet-5 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACT_DIR)), name="artifacts")
MODEL_CACHE = {"model": None, "modified": None, "metadata": None}


def read_json(path, fallback=None):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def get_model_and_metadata():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise HTTPException(status_code=409, detail="The project has not been trained yet. Run `python train.py` first.")
    modified = MODEL_PATH.stat().st_mtime
    if MODEL_CACHE["model"] is None or MODEL_CACHE["modified"] != modified:
        MODEL_CACHE["model"] = tf.keras.models.load_model(MODEL_PATH)
        MODEL_CACHE["modified"] = modified
        MODEL_CACHE["metadata"] = read_json(METADATA_PATH, {})
    return MODEL_CACHE["model"], MODEL_CACHE["metadata"]


@app.get("/api/health")
def health():
    return {"ok": True, "service": "Neuro-Genetic LeNet-5 API"}


@app.get("/api/status")
def status():
    current = read_json(STATUS_PATH, {"status": "not_started", "stage": "Not trained yet", "percent": 0, "logs": []})
    current["model_ready"] = MODEL_PATH.exists() and METADATA_PATH.exists()
    return current


@app.get("/api/dashboard")
def dashboard():
    metadata = read_json(METADATA_PATH)
    summary = read_json(SUMMARY_PATH)
    history = read_json(HISTORY_PATH)
    if metadata is None or summary is None or history is None:
        raise HTTPException(status_code=404, detail="No experiment results yet. Train the project first.")
    return {"metadata": metadata, "summary": summary, "history": history}


@app.get("/api/examples")
def examples():
    if not EXAMPLES_PATH.exists():
        raise HTTPException(status_code=404, detail="Training must finish before sample inputs are available.")
    stored = np.load(EXAMPLES_PATH)
    items = []
    for index, (image, label) in enumerate(zip(stored["images"], stored["labels"])):
        png = image_to_png_bytes(Image.fromarray(np.uint8(np.clip(image.squeeze() * 255, 0, 255))))
        items.append({"id": index, "label": int(label), "image": base64.b64encode(png).decode("ascii")})
    return {"examples": items}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Choose an image first.")
    try:
        contents = await file.read()
        image_array, normalized_image = preprocess_image_bytes(contents)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read this image: {error}")
    model, metadata = get_model_and_metadata()
    started = time.perf_counter()
    probabilities = model.predict(image_array[None, ...], verbose=0)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    order = np.argsort(probabilities)[::-1]
    predicted_digit = int(order[0])
    confidence = float(probabilities[predicted_digit])
    runner_up = float(probabilities[int(order[1])])
    margin = confidence - runner_up
    if confidence >= 0.90:
        confidence_label = "Strong score"
    elif confidence >= 0.65:
        confidence_label = "Moderate score"
    else:
        confidence_label = "Uncertain score"
    coverage = float(np.mean(image_array > 0.15))
    if coverage < 0.005:
        input_quality = "Almost blank"
        quality_note = "Very little foreground ink was found. Draw or upload a darker, clearer digit."
    elif coverage > 0.55:
        input_quality = "Dense image"
        quality_note = "A large part of the canvas contains ink. A single centered digit usually works best."
    else:
        input_quality = "Usable stroke"
        quality_note = "A foreground stroke was detected and centered in the 28 × 28 input."
    top_three = [
        {"digit": int(index), "probability": float(probabilities[index]), "percentage": round(float(probabilities[index]) * 100, 2)}
        for index in order[:3]
    ]
    analysis = [
        f"The highest class score belongs to digit {predicted_digit} at {confidence * 100:.2f}%.",
        f"The next closest class is digit {int(order[1])} at {runner_up * 100:.2f}%; the top-two gap is {margin * 100:.2f} percentage points.",
        f"The preprocessed image contains about {coverage * 100:.1f}% foreground pixels. {quality_note}",
        "These are model confidence scores, not a guarantee. Similar handwriting can produce close scores, so use the top-three panel to inspect ambiguity.",
    ]
    normalized_png = image_to_png_bytes(normalized_image, scale=10)
    genome = metadata["best_genome"]
    return {
        "prediction": predicted_digit,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_label": confidence_label,
        "top_three": top_three,
        "analysis": analysis,
        "input_quality": input_quality,
        "ink_coverage": round(coverage * 100, 2),
        "quality_note": quality_note,
        "top_two_gap": round(margin * 100, 2),
        "inference_ms": round(elapsed_ms, 2),
        "compute": metadata.get("hardware", hardware_details()),
        "parameter_count": int(model.count_params()),
        "estimated_macs": int(metadata.get("estimated_macs_per_image", estimate_macs(genome))),
        "normalized_image": base64.b64encode(normalized_png).decode("ascii"),
        "pipeline": model_explanation(genome),
        "genome": genome,
    }
