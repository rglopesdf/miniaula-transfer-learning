"""Monta o dataset 'lavouras-4' a partir do HLS Multi-Temporal Crop Classification (IBM/NASA, CC BY 4.0).

Uso:  python scripts/montar_dataset.py CAMINHO/validation_chips  dados

Cada chip original tem 224x224 pixels (30 m por pixel), 18 bandas (6 bandas x 3 datas de 2022)
e uma máscara com a cultura de cada pixel (USDA Cropland Data Layer).
Recortamos janelas de 32x32 pixels (cerca de 1 km x 1 km) em que uma única classe ocupa
pelo menos 70% da área e salvamos uma imagem PNG em falsa-cor da data do meio da safra:
    R = infravermelho próximo (NIR), G = infravermelho de ondas curtas (SWIR 1), B = vermelho.
Treino e teste usam chips diferentes, para não haver vazamento espacial.
"""
import csv, glob, os, random, sys
import numpy as np
import rasterio
from PIL import Image

ORIGEM, DESTINO = sys.argv[1], sys.argv[2]
JANELA, PUREZA = 32, 0.70
N_TREINO, N_TESTE = 150, 100          # por classe
CLASSES = ["soja", "milho", "outras", "nao_ha"]
# classe original (CDL agrupado) -> nossa classe.  0 = sem dado e 10 = pousio ficam de fora.
GRUPO = {4: 0, 3: 1, 8: 2, 9: 2, 11: 2, 12: 2, 13: 2, 1: 3, 2: 3, 5: 3, 6: 3, 7: 3}
NOME_ORIG = {1: "vegetacao natural", 2: "floresta", 3: "milho", 4: "soja", 5: "area umida", 6: "urbano/solo exposto",
             7: "agua", 8: "trigo de inverno", 9: "alfafa", 11: "algodao", 12: "sorgo", 13: "outras culturas"}
BANDAS, ESCALA = [9, 10, 8], [5000, 3500, 1500]   # NIR, SWIR1, vermelho da 2a data; refletancia x 10000

rng = random.Random(42)
mascaras = sorted(f for f in glob.glob(os.path.join(ORIGEM, "*.mask.tif")) if not os.path.basename(f).startswith("._"))
candidatos = []
for f in mascaras:
    m = rasterio.open(f).read(1)
    g = np.full(m.shape, -1)
    for k, v in GRUPO.items():
        g[m == k] = v
    for i in range(0, 224, JANELA):
        for j in range(0, 224, JANELA):
            w = g[i:i + JANELA, j:j + JANELA]
            cont = np.bincount(w[w >= 0].ravel(), minlength=4)
            k = int(cont.argmax())
            if cont[k] / w.size >= PUREZA:
                orig = int(np.bincount(m[i:i + JANELA, j:j + JANELA].ravel(), minlength=14).argmax())
                candidatos.append(dict(chip=os.path.basename(f)[:-9], lin=i, col=j, classe=k,
                                       pureza=round(cont[k] / w.size, 3), classe_original=NOME_ORIG[orig]))

chips = sorted({c["chip"] for c in candidatos})
rng.shuffle(chips)
chips_teste = set(chips[: len(chips) * 4 // 10])

def sortear(k, teste, n):
    """Sorteia n janelas da classe k, alternando entre as subclasses originais para dar variedade."""
    por_sub = {}
    for c in candidatos:
        if c["classe"] == k and (c["chip"] in chips_teste) == teste:
            por_sub.setdefault(c["classe_original"], []).append(c)
    for v in por_sub.values():
        rng.shuffle(v)
    escolhidos, subs = [], sorted(por_sub)
    while len(escolhidos) < n and any(por_sub.values()):
        for s in subs:
            if por_sub[s] and len(escolhidos) < n:
                escolhidos.append(por_sub[s].pop())
    return escolhidos

linhas = []
for parte, teste, n in [("treino", False, N_TREINO), ("teste", True, N_TESTE)]:
    for k, nome in enumerate(CLASSES):
        os.makedirs(os.path.join(DESTINO, parte, nome), exist_ok=True)
        for c in sortear(k, teste, n):
            a = rasterio.open(os.path.join(ORIGEM, c["chip"] + "_merged.tif")).read().astype("float32")
            w = a[BANDAS, c["lin"]:c["lin"] + JANELA, c["col"]:c["col"] + JANELA]
            rgb = np.clip(w / np.array(ESCALA, dtype="float32")[:, None, None], 0, 1)
            arq = f'{nome}_{c["chip"]}_{c["lin"]:03d}_{c["col"]:03d}.png'
            Image.fromarray((rgb.transpose(1, 2, 0) * 255).round().astype("uint8")).save(os.path.join(DESTINO, parte, nome, arq))
            linhas.append(dict(parte=parte, classe=nome, arquivo=arq, **{x: c[x] for x in ("chip", "lin", "col", "pureza", "classe_original")}))

with open(os.path.join(DESTINO, "metadados.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(linhas[0]))
    wr.writeheader(); wr.writerows(linhas)
for parte in ("treino", "teste"):
    print(parte, {n: sum(1 for l in linhas if l["parte"] == parte and l["classe"] == n) for n in CLASSES})
