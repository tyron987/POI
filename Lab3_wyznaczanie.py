import os
import random
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# Parametry GLCM
distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
props = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

# Wczytanie i przetworzenie jednej próbki
def extract_features(image_path, distances, angles):
    image = io.imread(image_path)
    image = (image * 255).astype(np.uint8)

    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feature_vector = []
    for prop in props:
        values = graycoprops(glcm, prop)
        feature_vector.extend(values.flatten())
    return feature_vector

# Przejście po wszystkich obrazach w podfolderach
def process_texture_dataset(dataset_folder, max_samples_per_class):
    rows = []

    categories = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    for category in tqdm(categories, desc="Foldery", position=0):  # pasek postępu dla folderów
        category_path = os.path.join(dataset_folder, category)
        if not os.path.isdir(category_path):
            continue

        all_files = [f for f in os.listdir(category_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        selected_files = random.sample(all_files, min(len(all_files), max_samples_per_class))

        with tqdm(selected_files, desc=f"{category}", position=1, leave=False) as bar:
            for filename in bar:
                filepath = os.path.join(category_path, filename)
                bar.set_postfix_str(f"{filename}")
                features = extract_features(filepath, distances, angles)
                features.append(category)
                rows.append(features)
    return rows

# Nagłówki kolumn
def create_headers():
    headers = []
    for prop in props:
        for d in distances:
            for a_deg in [0, 45, 90, 135]:
                headers.append(f"{prop}_d{d}_a{a_deg}")
    headers.append("category")
    return headers

# Uruchomienie
if __name__ == "__main__":
    dataset_folder = input("Podaj ścieżkę do folderu z kategoriami tekstur: ").strip()
    max_samples_per_class = int(input("Podaj liczbę próbek do przetworzenia na kategorię: "))
    data = process_texture_dataset(dataset_folder, max_samples_per_class)
    headers = create_headers()
    df = pd.DataFrame(data, columns=headers)
    df.to_csv("wektory_" + str(max_samples_per_class) + ".csv", index=False)
    print("Zapisano wektory cech do wektory_" + str(max_samples_per_class) + ".csv")
