"""Dataset utilities for the Kaggle Garbage Classification dataset."""
from pathlib import Path
import json
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "garbage_classification"
KAGGLE_DATASET = "mostafaabla/garbage-classification"
IMAGE_SIZE = 64
TEST_SIZE = 0.20
RANDOM_STATE = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _contains_images(folder):
    return any(p.is_file() and p.suffix.lower() in VALID_EXTENSIONS for p in folder.rglob("*"))


def _find_class_directory(download_path):
    """Find the directory whose immediate subfolders are image classes."""
    root = Path(download_path)
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    best_dir, best_count = None, 0
    for directory in candidates:
        class_dirs = [
            child for child in directory.iterdir()
            if child.is_dir() and not child.name.startswith(".") and _contains_images(child)
        ]
        if len(class_dirs) > best_count:
            best_dir, best_count = directory, len(class_dirs)
    if best_dir is None or best_count < 2:
        raise FileNotFoundError(
            f"Could not find class folders in downloaded Kaggle dataset: {download_path}"
        )
    return best_dir


def get_dataset_dir(data_dir=None):
    """Use a local dataset if supplied; otherwise download it from KaggleHub."""
    if data_dir is not None:
        local_dir = Path(data_dir)
        if local_dir.exists() and _find_class_directory(local_dir):
            return _find_class_directory(local_dir)

    if DATA_DIR.exists() and _contains_images(DATA_DIR):
        return _find_class_directory(DATA_DIR)

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    dataset_dir = _find_class_directory(downloaded_path)
    print(f"Dataset downloaded to: {downloaded_path}")
    print(f"Detected class directory: {dataset_dir}")
    return dataset_dir


def collect_image_paths(data_dir=None):
    data_dir = get_dataset_dir(data_dir)
    classes = sorted(
        folder.name for folder in data_dir.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
        and _contains_images(folder)
    )
    if len(classes) < 2:
        raise ValueError(f"At least two image class folders are required in {data_dir}")

    paths, labels = [], []
    for label, class_name in enumerate(classes):
        class_files = [
            p for p in (data_dir / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        paths.extend(class_files)
        labels.extend([label] * len(class_files))

    if not paths:
        raise ValueError(f"No supported images found under {data_dir}")

    print("Dataset class counts:")
    for name, label in zip(classes, range(len(classes))):
        print(f"  {name}: {int(np.sum(np.asarray(labels) == label))}")
    return np.asarray(paths, dtype=object), np.asarray(labels), classes


def split_dataset(test_size=TEST_SIZE, random_state=RANDOM_STATE, data_dir=None):
    paths, labels, classes = collect_image_paths(data_dir)
    counts = np.bincount(labels, minlength=len(classes))
    if np.any(counts < 2):
        rare = [classes[i] for i, count in enumerate(counts) if count < 2]
        raise ValueError(f"Each class needs at least 2 images. Classes with too few images: {rare}")
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_paths, test_paths, y_train, y_test, classes


def load_images(paths, image_size=IMAGE_SIZE):
    images = []
    for path in paths:
        image = load_img(str(path), target_size=(image_size, image_size), color_mode="rgb")
        images.append(img_to_array(image) / 255.0)
    return np.asarray(images, dtype=np.float32)


def save_metadata(classes, image_size, test_accuracy=None):
    metadata = {
        "classes": list(classes),
        "image_size": int(image_size),
        "test_accuracy": None if test_accuracy is None else float(test_accuracy)
    }
    path = BASE_DIR / "model" / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata saved: {path}")
