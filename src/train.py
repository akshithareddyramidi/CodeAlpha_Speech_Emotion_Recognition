"""Train a simple CNN on the RAVDESS speech-only audio files."""
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from audio_utils import load_audio, extract_features

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# RAVDESS filename emotion codes.
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def emotion_from_filename(path):
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    # RAVDESS speech files have modality 03 (audio-only) and vocal channel 01.
    if parts[0] != "03" or parts[1] != "01":
        return None
    return EMOTIONS.get(parts[2])

def main():
    audio_paths = sorted(DATASET_DIR.rglob("*.wav"))
    samples, labels = [], []
    print(f"Found {len(audio_paths)} WAV files under {DATASET_DIR}")
    for index, path in enumerate(audio_paths, start=1):
        label = emotion_from_filename(path)
        if label is None:
            continue
        try:
            y, sr = load_audio(str(path))
            samples.append(extract_features(y, sr))
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_paths)} files")

    if len(samples) < 20:
        raise SystemExit("Not enough RAVDESS speech WAV files found. Check the dataset folder and README.")

    x = np.asarray(samples, dtype=np.float32)[..., np.newaxis]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=x.shape[1:]),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR / "emotion_model.keras", save_best_only=True)
    ]
    model.fit(
        x_train, y_train, validation_split=0.2, epochs=60, batch_size=32,
        callbacks=callbacks, verbose=1
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    (MODEL_DIR / "labels.json").write_text(json.dumps(encoder.classes_.tolist(), indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved model and labels to {MODEL_DIR}")

if __name__ == "__main__":
    main()
