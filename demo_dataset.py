import os
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _draw_face(draw, emotion: str, variation: int = 0) -> None:
    face_color = 245
    draw.ellipse((8, 7, 40, 39), fill=face_color, outline=0)

    left_eye = 14 + variation
    right_eye = 34 + variation
    eye_y = 16 + (variation % 2)

    draw.ellipse((left_eye, eye_y, left_eye + 2, eye_y + 2), fill=0)
    draw.ellipse((right_eye, eye_y, right_eye + 2, eye_y + 2), fill=0)

    if emotion == "angry":
        draw.line((10, 14, 17, 12), fill=0, width=1)
        draw.line((31, 12, 38, 14), fill=0, width=1)
        draw.arc((15, 21, 33, 30), 200, 340, fill=0, width=2)
    elif emotion == "disgust":
        draw.line((12, 16, 19, 18), fill=0, width=1)
        draw.line((29, 18, 36, 16), fill=0, width=1)
        draw.arc((15, 24, 33, 30), 20, 160, fill=0, width=2)
    elif emotion == "fear":
        draw.ellipse((12, 15, 18, 19), fill=0)
        draw.ellipse((30, 15, 36, 19), fill=0)
        draw.arc((15, 23, 33, 31), 210, 330, fill=0, width=2)
    elif emotion == "happy":
        draw.ellipse((12, 15, 18, 19), fill=0)
        draw.ellipse((30, 15, 36, 19), fill=0)
        draw.arc((14, 23, 34, 31), 200, 340, fill=0, width=2)
        draw.ellipse((11, 27, 14, 30), fill=0)
        draw.ellipse((34, 27, 37, 30), fill=0)
    elif emotion == "neutral":
        draw.ellipse((12, 15, 18, 19), fill=0)
        draw.ellipse((30, 15, 36, 19), fill=0)
        draw.line((17, 26, 31, 26), fill=0, width=2)
    elif emotion == "sad":
        draw.ellipse((12, 17, 18, 21), fill=0)
        draw.ellipse((30, 17, 36, 21), fill=0)
        draw.arc((15, 21, 33, 29), 20, 160, fill=0, width=2)
    elif emotion == "surprise":
        draw.ellipse((12, 15, 18, 19), fill=0)
        draw.ellipse((30, 15, 36, 19), fill=0)
        draw.ellipse((18, 24, 30, 32), fill=0)
    else:
        draw.line((17, 26, 31, 26), fill=0, width=2)

    draw.ellipse((8, 7, 40, 39), outline=0, width=1)


def _make_face_image(emotion: str, variant: int = 0) -> np.ndarray:
    image = Image.new("L", (48, 48), color=220)
    draw = ImageDraw.Draw(image)
    _draw_face(draw, emotion, variant)
    return np.asarray(image, dtype=np.uint8)


def ensure_demo_dataset(base_dir: str) -> None:
    train_dir = os.path.join(base_dir, "dataset", "train")
    test_dir = os.path.join(base_dir, "dataset", "test")

    for root in (train_dir, test_dir):
        os.makedirs(root, exist_ok=True)

    for emotion in EMOTIONS:
        train_emotion_dir = os.path.join(train_dir, emotion)
        test_emotion_dir = os.path.join(test_dir, emotion)
        os.makedirs(train_emotion_dir, exist_ok=True)
        os.makedirs(test_emotion_dir, exist_ok=True)

        existing_train = len(os.listdir(train_emotion_dir))
        existing_test = len(os.listdir(test_emotion_dir))

        if existing_train >= 12 and existing_test >= 4:
            continue

        for i in range(12):
            if i >= existing_train:
                img = _make_face_image(emotion, i)
                path = os.path.join(train_emotion_dir, f"{emotion}_{i:02d}.png")
                Image.fromarray(img).save(path)

        for i in range(4):
            if i >= existing_test:
                img = _make_face_image(emotion, 100 + i)
                path = os.path.join(test_emotion_dir, f"{emotion}_{i:02d}.png")
                Image.fromarray(img).save(path)


def create_demo_dataset(base_dir: str) -> None:
    ensure_demo_dataset(base_dir)


if __name__ == "__main__":
    ensure_demo_dataset(os.path.dirname(os.path.abspath(__file__)))
    print("Demo dataset created.")
