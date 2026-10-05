import json
import os
import cv2
import numpy as np
import tensorflow as tf

from demo_dataset import ensure_demo_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_cnn.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")


def ensure_model_exists() -> None:
    if os.path.isfile(MODEL_PATH) and os.path.isfile(CLASS_PATH):
        return

    print("Training artifacts missing. Building a demo model...")
    ensure_demo_dataset(BASE_DIR)

    import train_model

    print("Demo model training completed.")


ensure_model_exists()

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or try another camera index."
    )

print("Webcam started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            probabilities = model.predict(face, verbose=0)[0]
            index = int(np.argmax(probabilities))
            emotion = class_names[index]
            confidence = float(probabilities[index]) * 100

            label = f"{emotion}: {confidence:.1f}%"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                frame,
                (x, y - 35),
                (x + w, y),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Face Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
