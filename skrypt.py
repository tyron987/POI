from icrawler.builtin import BingImageCrawler
from PIL import Image
import imagehash
import os
from tqdm import tqdm

kategorie = ['lampy', 'krzesła', 'biurka', 'szafy'] # to co chcesz znaleźć
liczba_na_kategorie = 750 # prawdopodobnie tylu nie znajdzie, ale lepiej mieć zapas przy możliwych duplikatach


def usun_duplikaty(folder):
    print(f"\nSprawdzanie duplikatów w '{folder}'...")
    hashy = {}
    pliki = os.listdir(folder)

    for filename in tqdm(pliki, desc=f"Analiza {os.path.basename(folder)}", unit="obraz"):
        filepath = os.path.join(folder, filename)
        try:
            with Image.open(filepath) as img:
                h = str(imagehash.average_hash(img))
            if h in hashy:
                os.remove(filepath)
            else:
                hashy[h] = filename
        except Exception as e:
            tqdm.write(f"Błąd przy pliku {filename}: {e}")
            os.remove(filepath)


for kategoria in kategorie:
    folder = f'obrazy/{kategoria}'
    os.makedirs(folder, exist_ok=True)

    print(f"\nPobieranie zdjęć: {kategoria}")
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=kategoria, max_num=liczba_na_kategorie)

    usun_duplikaty(folder)

print("\nZakończono. Duplikaty zostały usunięte.")
