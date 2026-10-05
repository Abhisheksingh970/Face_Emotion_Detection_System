import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, callbacks

from demo_dataset import ensure_demo_dataset

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Check dataset
# -----------------------------
if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
    print("Missing training/test folders. Generating a demo dataset...")
    ensure_demo_dataset(BASE_DIR)

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}\n"
        "Create dataset/train/<emotion>/ folders and add images."
    )

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}\n"
        "Create dataset/test/<emotion>/ folders and add images."
    )

# -----------------------------
# Load datasets
# -----------------------------

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_filtered_dataset(directory: str, allowed_classes, shuffle: bool):
    class_to_index = {name: index for index, name in enumerate(allowed_classes)}
    file_paths = []
    labels = []

    for class_name in allowed_classes:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            lower_name = filename.lower()
            if not lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue

            full_path = os.path.join(class_dir, filename)
            if not is_valid_image(full_path):
                print(f"Skipping unreadable/corrupt image: {full_path}")
                continue

            file_paths.append(full_path)
            labels.append(class_to_index[class_name])

    if not file_paths:
        raise ValueError(f"No supported image files found in {directory} for classes {allowed_classes}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, tf.cast(labels, tf.int32)))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=1, expand_animations=False)
        if image.shape.rank == 2:
            image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(file_paths), 1024),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_merged_dataset(train_dir: str, test_dir: str, allowed_classes, shuffle: bool):
    class_to_index = {name: index for index, name in enumerate(allowed_classes)}
    file_paths = []
    labels = []

    for class_name in allowed_classes:
        for base_dir in (train_dir, test_dir):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in sorted(os.listdir(class_dir)):
                lower_name = filename.lower()
                if not lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    continue

                full_path = os.path.join(class_dir, filename)
                if not is_valid_image(full_path):
                    print(f"Skipping unreadable/corrupt image: {full_path}")
                    continue

                file_paths.append(full_path)
                labels.append(class_to_index[class_name])

    if not file_paths:
        raise ValueError(f"No supported image files found in either {train_dir} or {test_dir} for classes {allowed_classes}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, tf.cast(labels, tf.int32)))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=1, expand_animations=False)
        if image.shape.rank == 2:
            image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(file_paths), 1024),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_classes = sorted(
    entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()
)
test_classes = sorted(
    entry.name for entry in os.scandir(TEST_DIR) if entry.is_dir()
)
all_classes = sorted(set(train_classes) | set(test_classes))

print("All available classes:", all_classes)

train_ds = build_merged_dataset(TRAIN_DIR, TEST_DIR, all_classes, shuffle=True)
test_ds = build_filtered_dataset(TEST_DIR, all_classes, shuffle=False)

class_names = all_classes
print("Classes:", class_names)

if len(class_names) != 7:
    print(
        "Warning: This project is configured for a 7-class emotion setup, but the full dataset contains "
        f"{len(class_names)} classes because the test set includes extra labels."
    )

with open(os.path.join(MODEL_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# -----------------------------
# CNN model
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="augmentation")

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.50),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model_path = os.path.join(MODEL_DIR, "emotion_cnn.keras")

cb = [
    callbacks.ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=cb,
)

# -----------------------------
# Final evaluation
# -----------------------------
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# -----------------------------
# Save training plot
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
plt.close()

print(f"Model saved to: {model_path}")
