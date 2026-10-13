# ============================================================
# GARBAGE CLASSIFICATION - TEST / EVALUATION
# Evaluate a previously trained model on the same reproducible
# held-out test split used by train.py.
# ============================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dataset import BASE_DIR, IMAGE_SIZE, split_dataset, load_images

MODEL_PATH = BASE_DIR / "model" / "trash_class_cnn_mvp.keras"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run `python train.py` first."
        )

    print("=" * 70)
    print("GARBAGE CLASSIFICATION - TEST")
    print("=" * 70)

    _, test_paths, _, y_test, classes = split_dataset()
    X_test = load_images(test_paths)
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    print(f"Test images : {len(test_paths)}")
    print(f"Test loss   : {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0
    ))

    report = classification_report(
        y_test, y_pred, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "test_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(max(8, len(classes) * 0.8), max(6, len(classes) * 0.65)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=160)
    plt.close()

    pd.DataFrame({
        "image_path": [str(p) for p in test_paths],
        "actual_class": [classes[int(i)] for i in y_test],
        "predicted_class": [classes[int(i)] for i in y_pred],
        "confidence": np.max(probabilities, axis=1)
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metadata_path = BASE_DIR / "model" / "metadata.json"
    metadata_path.write_text(json.dumps({
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "test_accuracy": float(accuracy)
    }, indent=2), encoding="utf-8")

    print(f"Evaluation files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
