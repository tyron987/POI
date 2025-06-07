from PIL import Image
import os
from tqdm import tqdm

wej_folder = "obrazy"
wyj_folder = "obrazy_256"
nowy_rozmiar = (256, 256)

for kategoria in os.listdir(wejscie_folder):
    folder_zrodlowy = os.path.join(wej_folder, kategoria)
    folder_docelowy = os.path.join(wyj_folder, kategoria)

    if not os.path.isdir(folder_zrodlowy):
        continue

    os.makedirs(folder_docelowy, exist_ok=True)

    print(f"Skalowanie kategorii: {kategoria}")
    for filename in tqdm(os.listdir(folder_zrodlowy), desc=kategoria, unit="obraz"):
        sciezka_zrodlowa = os.path.join(folder_zrodlowy, filename)
        sciezka_docelowa = os.path.join(folder_docelowy, filename)
        try:
            with Image.open(sciezka_zrodlowa) as img:
                img = img.convert("RGB")
                img = img.resize(nowy_rozmiar, Image.LANCZOS)
                img.save(sciezka_docelowa)
        except Exception as e:
            tqdm.write(f"Błąd przetwarzania {filename}: {e}")
