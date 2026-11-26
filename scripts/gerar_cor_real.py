"""Gera a versão em cor real (vermelho, verde e azul visíveis) dos mesmos recortes de `dados/`.

Uso:  python scripts/gerar_cor_real.py CAMINHO/validation_chips dados dados_cor_real

Lê `dados/metadados.csv` para saber de que chip e posição veio cada recorte e salva, com o mesmo
nome de arquivo e a mesma estrutura de pastas, a composição em cor real da mesma data (2a data da série).
Serve para o aluno comparar o que o olho veria com a falsa-cor usada no treino.
"""
import csv
import os
import sys

import numpy as np
import rasterio
from PIL import Image

ORIGEM, DADOS, DESTINO = sys.argv[1], sys.argv[2], sys.argv[3]
JANELA = 32
BANDAS = [8, 7, 6]        # vermelho, verde e azul da 2a data (refletância x 10000)
BRANCO = 1500.0           # refletância de 15% vira branco: realça solo e vegetação, que refletem pouco no visível
GAMA = 0.8                # clareia os tons médios, como numa foto

cache = {}
with open(os.path.join(DADOS, "metadados.csv"), newline="") as fh:
    linhas = list(csv.DictReader(fh))
for l in linhas:
    if l["chip"] not in cache:
        cache.clear()
        cache[l["chip"]] = rasterio.open(os.path.join(ORIGEM, l["chip"] + "_merged.tif")).read().astype("float32")
    a = cache[l["chip"]]
    i, j = int(l["lin"]), int(l["col"])
    w = a[BANDAS, i:i + JANELA, j:j + JANELA]
    rgb = np.clip(w / BRANCO, 0, 1) ** GAMA
    pasta = os.path.join(DESTINO, l["parte"], l["classe"])
    os.makedirs(pasta, exist_ok=True)
    Image.fromarray((rgb.transpose(1, 2, 0) * 255).round().astype("uint8")).save(os.path.join(pasta, l["arquivo"]))
print(len(linhas), "imagens em cor real salvas em", DESTINO)
