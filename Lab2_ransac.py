import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_plane(points):
    """Służy do obliczania wektora normalnego płaszczyzny przechodzącej przez trzy wybrane punkty."""
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p1)
    return a, b, c, d


def refine_plane_least_squares(points):
    """Służy do oszacowania dokładniejszego modelu dla inlierów z najlepszej grupy"""
    # Obliczanie środka punktów
    centroid = np.mean(points, axis=0)

    # Przesuwanie punktów względem środka
    shifted_points = points - centroid

    # Oblicz SVD - Rozkład według wartości osobliwych
    _, _, vh = np.linalg.svd(shifted_points)

    # Normalna płaszczyzna to ostatni wektor z V (lub vh[-1])
    normal = vh[-1, :]
    a, b, c = normal
    d = -np.dot(normal, centroid)  # wyraz wolny d

    return a, b, c, d


def ransac_plane_fit(points, iterations=100, threshold=1.0):
    """Implementacja algorytmu ransac na podstawie podanego schematu:\n
    1. Wylosuj 3 punkty z całej chmury\n
    2. Sprawdź jak dobrze wszystkie punkty z chmury pasują do tej płaszczyzny\n
    2a. Określ warunek dobrego dopasowania (maksymalną odległość od płaszczyzny)\n
    2b. Wyznacz grupę inlierów i oblicz ile ich jest\n
    2c. Jeśli liczba inlierów jest największa z dotychczasowych to zapamiętaj bieżący model jako najlepszy\n
    3. Jeśli osiągnięto maksymalną liczbę iteracji to przejdź do punktu 4. W przeciwnym przypadku powrót do punktu 1\n
    4. Dla inlierów z najlepszej grupy oszacuj dokładniejszy model (np.metoda najmniejszych kwadratów, tutaj
    mogę już jej użyć ponieważ wiem, że outliery mi tutaj nie popsują tej ostatecznej postaci mojego modelu)\n"""
    best_inliers = []
    best_model = None

    for _ in range(iterations):
        # Losowanie trzech punktów z chmury
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]

        # Dopasanie płaszczyzny
        a, b, c, d = fit_plane(sample)

        # Oblicze odległości wszystkich punktów od płaszczyzny
        distances = np.abs((points @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c]))

        # Wyznaczenie grupy inlierów
        inliers = points[distances <= threshold]

        # Jeśli liczba inlierów jest większa niż dotychczasowa, zapisz nowy model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a, b, c, d)

    # Dalsze dopasowanie do modelu
    if len(best_inliers) > 0:
        a, b, c, d = refine_plane_least_squares(best_inliers)
        best_model = (a, b, c, d)

    return best_model, best_inliers

def set_axes_equal(ax):
    """Funkcja do automatycznego skalowania osi 3D tak aby chmury wyglądały naturalnie (proporcje)"""
    # Aktualne granice osi w przestrzeni 3D
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Obliczanie zakresu dla każdej osi (odległość między minimalną a maksymalną wartością)
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # Promień wykresu
    plot_radius = 0.5 * max(x_range, y_range, z_range)

    # Ustawienie granic osi tak, aby były symetryczne wokół ich środków
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Wczytywanie danych chmury z pliku
file_path = input("Podaj ścieżkę do pliku z danymi: ")
data = pd.read_csv(file_path, header=None, delimiter=',', names=["x", "y", "z"])

# Konwersja wczytanych danych do tablicy numpy
points = data.to_numpy()

# Dopasowanie płaszczyzny algorytmem RANSAC
# Model - dopasowane współczynniki płaszczyzny
# Inliers - punkty należące do dopasowanej płaszczyzny
model, inliers = ransac_plane_fit(points, iterations=1000, threshold=1.0)

# Współczynniki płaszczyzny
a, b, c, d = model
normal_vector = np.array([a, b, c])
print(f"Wektor normalny płaszczyzny: {normal_vector}")

# Obliczenia średniej odległości wszystkich punktów do płaszczyzny
distances = np.abs((points @ normal_vector + d) / np.linalg.norm(normal_vector))
mean_distance = np.mean(distances)

print(f"Średnia odległość punktów od płaszczyzny: {mean_distance:.4f}")

# Sprawdzenie czy chmura tworzy płaszczyznę
if mean_distance < 0.5:  # Próg uznania chmury za płaszczyznę
    print("Chmura punktów jest płaszczyzną.")

    normal_unit = normal_vector / np.linalg.norm(normal_vector)
    vertical_alignment = abs(normal_unit[2])

    # Sprawdzenie orientacji płaszczyzny
    vertical_threshold = 0.05  # Tolerancja orientacji płaszczyzny
    if vertical_alignment < vertical_threshold:
        print("Płaszczyzna jest pionowa.")
    elif vertical_alignment < (1- vertical_threshold):
        print("Płaszczyzna jest pozioma.")
else:
    print("Chmura punktów nie jest płaszczyzną.")

# Wizualizacja 3D chmury
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='gray', alpha=0.3, label='Wszystkie punkty')
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='green', label='Dopasowane punkty')
lim = [np.min(points, axis=0), np.max(points, axis=0)]

grid_size = 10
# Automatyczne dostosowanie rysowania płaszczyzny
if abs(c) > abs(a) and abs(c) > abs(b):
    # Normalne - rysujemy Z jako funkcję X i Y
    xx, yy = np.meshgrid(np.linspace(lim[0][0], lim[1][0], grid_size),
                         np.linspace(lim[0][1], lim[1][1], grid_size))
    zz = (-a * xx - b * yy - d) / c
elif abs(a) > abs(b) and abs(a) > abs(c):
    # Płaszczyzna pionowa - rysujemy X jako funkcję Y i Z
    yy, zz = np.meshgrid(np.linspace(lim[0][1], lim[1][1], grid_size),
                         np.linspace(lim[0][2], lim[1][2], grid_size))
    xx = (-b * yy - c * zz - d) / a
elif abs(b) > abs(a) and abs(b) > abs(c):
    # Inna płaszczyzna pionowa - rysujemy Y jako funkcję X i Z
    xx, zz = np.meshgrid(np.linspace(lim[0][0], lim[1][0], grid_size),
                         np.linspace(lim[0][2], lim[1][2], grid_size))
    yy = (-a * xx - c * zz - d) / b

ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)
set_axes_equal(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('RANSAC – dopasowanie płaszczyzny')
ax.legend()
plt.tight_layout()
plt.show()
