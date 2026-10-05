import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

from demo_dataset import ensure_demo_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_cnn.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

if len(sys.argv) < 2:
    print("Usage: python predict_image.py path_to_image.jpg")
    raise SystemExit(1)

image_path = sys.argv[1]

if not os.path.isfile(MODEL_PATH):
    print("Model not found. Training a demo model first...")
    ensure_demo_dataset(BASE_DIR)
    import train_model

if not os.path.isfile(CLASS_PATH):
    print("Class metadata missing. Training a demo model first...")
    ensure_demo_dataset(BASE_DIR)
    import train_model

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

faces = detector.detectMultiScale(
    image,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(40, 40),
)

if len(faces) == 0:
    print("No face detected.")
    raise SystemExit(0)

for i, (x, y, w, h) in enumerate(faces, start=1):
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))

    probs = model.predict(face, verbose=0)[0]
    index = int(np.argmax(probs))

    print(
        f"Face {i}: {class_names[index]} "
        f"({probs[index] * 100:.2f}%)"
    )
