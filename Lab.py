from scipy.stats import norm
from csv import writer

num_points = 2000

# Definicje trzech powierzchni
surfaces = [
    # Nazwa pliku, rozkład: X, Y, Z
    ("Poziom.xyz",   norm(0, 200), norm(0, 20), norm(0.2, 0.05)),
    ("Pion.xyz",     norm(0, 20),  norm(0, 200), norm(0.2, 0.05)),
    ("Cylinder.xyz", norm(0, 20),  norm(0, 20), norm(100, 200)),
]

# Generowanie chmury i zapisywanie do pliku w pętli
for name, distribution_x, distribution_y, distribution_z in surfaces:
    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    with open(name, 'w', encoding='utf-8', newline='\n') as f:
        csv_writer = writer(f)
        csv_writer.writerows(zip(x, y, z))
