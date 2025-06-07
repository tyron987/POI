import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import defaultdict

# --- Parametry ---
img_size = (256, 256)
folder = "Images"  # folder ze zdjęciami do sprawdzenia
model_path = "model_obrazy_1.keras"

# --- Wczytaj model ---
model = load_model(model_path)

# --- Pobierz class_names z katalogu treningowego ---
ds = tf.keras.utils.image_dataset_from_directory(
    "Images",
    image_size=(256, 256),
    batch_size=1
)
class_names = ds.class_names

# --- Statystyki ---
stats = defaultdict(lambda: {"correct": 0, "total": 0})

# --- Funkcja predykcji ---
def predict_image_class(image_path):
    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions)
    return class_names[predicted_class_idx]

# --- Przetwarzanie testowych obrazów ---
for root, _, files in os.walk(folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(root, file)

            # Odczytaj prawdziwą klasę z folderu nadrzędnego
            true_class = os.path.basename(os.path.dirname(image_path))

            try:
                predicted_class = predict_image_class(image_path)

                stats[true_class]["total"] += 1
                if predicted_class == true_class:
                    stats[true_class]["correct"] += 1

                print(f"{image_path} => przewidziano: {predicted_class}, rzeczywista: {true_class}")
            except Exception as e:
                print(f"Błąd przy {image_path}: {e}")

# --- Wyniki końcowe ---
print("\n--- Podsumowanie skuteczności ---")
for class_name, values in stats.items():
    correct = values["correct"]
    total = values["total"]
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"{class_name}: {correct}/{total} poprawnych ({accuracy:.2f}%)")
