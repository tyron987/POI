import os
# Rozwiązanie do ostrzeżenia UserWarning: KMeans is known to have a memory leak on Windows with MKL,
# when there are less chunks than available threads.
os.environ["OMP_NUM_THREADS"] = "2"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Funkcja do skalowania osi 3D tak aby chmury wyglądały naturalnie
def set_axes_equal(ax):
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

# Tworzenie model KMeans z 3 klastrami
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(points)

# Predykcja do którego klastra należy każdy punkt
labels = kmeans.predict(points)
centers = kmeans.cluster_centers_

# Wykres 3D chmury po wykonaniu algorytmu
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='viridis', s=40)
set_axes_equal(ax)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='X', s=200, label='Centroidy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Algorytm K-średnich – chmura 3D')
plt.legend()
plt.show()
