from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Funkcja skalująca osie 3D tak aby chmury wyglądały naturalnie
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
# Konwersja tablicy na numpy wymagana do działania odczytanej chmury punktów
points = data.to_numpy()

# Uruchomienie algorytmu DBSCAN eps - maksymalna odległość między dwoma punktami, która zostanie uznana za sąsiedztwo
# min_samples - liczba punktów w sąsiedztwie wymaganych do uznania jako główny punkt (core point)
db = DBSCAN(eps=3, min_samples=2).fit(points)
labels = db.labels_

# Obliczenia liczby klastrów i szumu
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Szacowana liczba klastrów: %d" % n_clusters_)
print("Szacowana liczba punktów szumu: %d" % n_noise_)

# Wykres 3D chmury po wykonaniu algorytmu
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='viridis', s=40)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Algorytm DBSCAN – chmura 3D')
set_axes_equal(ax)
plt.show()
