# ============================================================
# GARBAGE CLASSIFICATION - CNN
# Train a Convolutional Neural Network to classify waste images.
# ============================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from dataset import (
    BASE_DIR, DATA_DIR, IMAGE_SIZE, TEST_SIZE, RANDOM_STATE,
    split_dataset, load_images, save_metadata
)

warnings.filterwarnings("ignore")

MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "trash_class_cnn_mvp.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 32

print("=" * 70)
print("GARBAGE CLASSIFICATION - CNN")
print("=" * 70)

# 1. Load dataset and split train/test
print("\n[1/8] Loading dataset and creating train/test split...")
train_paths, test_paths, y_train, y_test, classes = split_dataset()
print(f"Classes: {classes}")
print(f"Training images: {len(train_paths):,}")
print(f"Testing images : {len(test_paths):,}")
print(f"Image size     : {IMAGE_SIZE} x {IMAGE_SIZE}")

# 2. Dataset summary and class distribution
print("\n[2/8] Creating dataset summary...")
summary = pd.DataFrame({
    "class": classes,
    "total_images": [int(np.sum(np.concatenate([y_train, y_test]) == i)) for i in range(len(classes))],
    "train_images": [int(np.sum(y_train == i)) for i in range(len(classes))],
    "test_images": [int(np.sum(y_test == i)) for i in range(len(classes))]
})
print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

plt.figure(figsize=(12, 6))
summary.set_index("class")[["train_images", "test_images"]].plot(kind="bar", figsize=(12, 6))
plt.title("Garbage Classification Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.close()

# 3. Load and preprocess images
print("\n[3/8] Loading and preprocessing images...")
X_train = load_images(train_paths)
X_test = load_images(test_paths)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# 4. Build CNN model
print("\n[4/8] Building CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 5. Train
print("\n[5/8] Training model...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH), monitor="val_accuracy", save_best_only=True
    )
]
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
model.save(MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")

# 6. Evaluate on the held-out test set
print("\n[6/8] Evaluating test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

report = classification_report(
    y_test, y_pred, labels=list(range(len(classes))),
    target_names=classes, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")
print(classification_report(
    y_test, y_pred, labels=list(range(len(classes))),
    target_names=classes, zero_division=0
))

# 7. Save evaluation plots and prediction results
print("\n[7/8] Saving evaluation outputs...")
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
plt.figure(figsize=(max(8, len(classes) * 0.8), max(6, len(classes) * 0.65)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Garbage Classification")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history.get("accuracy", []), label="Training accuracy")
plt.plot(history.history.get("val_accuracy", []), label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history.get("loss", []), label="Training loss")
plt.plot(history.history.get("val_loss", []), label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
plt.close()

predictions_df = pd.DataFrame({
    "image_path": [str(p) for p in test_paths],
    "actual_class": [classes[int(i)] for i in y_test],
    "predicted_class": [classes[int(i)] for i in y_pred],
    "confidence": np.max(probabilities, axis=1)
})
predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# 8. Save metadata for app.py and final summary
save_metadata(classes, IMAGE_SIZE, test_accuracy)
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "train_images": int(len(train_paths)),
    "test_images": int(len(test_paths)),
    "classes": classes,
    "image_size": IMAGE_SIZE
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("\n[8/8] Training completed.")
print(f"Model       : {MODEL_PATH}")
print(f"Metadata    : {MODEL_DIR / 'metadata.json'}")
print(f"Outputs     : {OUTPUT_DIR}")
print("=" * 70)
