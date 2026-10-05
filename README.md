# Face Emotion Detection System

A real-time facial emotion detection project using Python, OpenCV, TensorFlow/Keras, and a CNN.

## Emotions
The default class order is:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Project structure

```text
Face_Emotion_Detection_System/
├── dataset/
│   └── README.txt
├── models/
├── train_model.py
├── emotion_detector.py
├── predict_image.py
├── requirements.txt
└── README.md
```

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset

The training script expects this structure:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Put face images inside the matching folders.

A commonly used choice is the FER-2013 dataset. If you use another dataset, keep the same class-folder structure.

## 3. Train the CNN

```powershell
python train_model.py
```

The trained model is saved as:

```text
models/emotion_cnn.keras
```

Training plots are saved in:

```text
models/training_history.png
```

## 4. Run webcam detection

```powershell
python emotion_detector.py
```

Press `q` to quit.

## 5. Test one image

```powershell
python predict_image.py path\to\image.jpg
```

## Notes

- The webcam program detects faces with OpenCV's Haar cascade.
- The CNN classifies each detected face.
- The model expects grayscale 48x48 face images.
- Real-world accuracy depends heavily on dataset quality, lighting, camera angle, and training.
