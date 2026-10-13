from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = MODEL_DIR / "trash_class_cnn_mvp.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"

app = FastAPI(title="Garbage Classification CNN")

model = None
classes = []
image_size = 64
test_accuracy = None

if METADATA_PATH.exists():
    import json
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    classes = metadata.get("classes", [])
    image_size = int(metadata.get("image_size", 64))
    test_accuracy = metadata.get("test_accuracy")

if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(classes)
    }

@app.get("/info")
async def info():
    return {
        "classes": classes,
        "image_size": image_size,
        "test_accuracy": test_accuracy,
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Put trash_class_cnn_mvp.keras in model/."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        data = await file.read()
        image = Image.open(__import__("io").BytesIO(data)).convert("RGB")
        image = image.resize((image_size, image_size))
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        probabilities = model.predict(x, verbose=0)[0]
        order = np.argsort(probabilities)[::-1][:5]

        predictions = [
            {
                "class": classes[int(i)],
                "confidence": float(probabilities[int(i)])
            }
            for i in order
        ]

        best = predictions[0]
        return {
            "predicted_class": best["class"],
            "confidence": best["confidence"],
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
