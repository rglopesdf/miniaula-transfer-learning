# Transfer Learning: ensinar uma rede a ver lavouras

Material de uma miniaula de **Aprendizagem Profunda** sobre transferência de aprendizado e fine-tuning,
aplicada à classificação de culturas agrícolas em imagens de satélite.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rglopesdf/miniaula-transfer-learning/blob/main/transfer_learning_lavouras.ipynb)

**[▶ Abrir o notebook direto no Google Colab](https://colab.research.google.com/github/rglopesdf/miniaula-transfer-learning/blob/main/transfer_learning_lavouras.ipynb)**: não precisa instalar nada. No Colab, ative a GPU em *Ambiente de execução → Alterar o tipo de ambiente → GPU (T4)* e execute *Ambiente de execução → Executar tudo*.

## O problema

Classificar recortes de cerca de 1 km × 1 km em quatro classes (`soja`, `milho`, `outras` culturas e
`nao_ha` lavoura) com apenas **600 imagens de treino**. O notebook compara três formas de treinar uma ResNet-18:

| | Estratégia | O que é treinado |
|---|---|---|
| A | Do zero | a rede inteira, com pesos aleatórios |
| B | Extração de características | só a nova camada final |
| C | Fine-tuning | a nova camada final e o último bloco (`layer4`) |

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `transfer_learning_lavouras.ipynb` | Notebook da aula, já executado. Roda no Google Colab sem configuração. |
| `resultados.json` | Resultados da execução registrada no notebook. |
| `dados/` | 1.000 imagens PNG de 32×32 pixels: 600 de treino e 400 de teste, 4 classes equilibradas. |
| `dados_cor_real/` | Os mesmos 1.000 recortes em cor real (vermelho, verde e azul visíveis), para comparação. |
| `dados/metadados.csv` | Origem de cada recorte (chip, posição, classe original, pureza). |
| `scripts/montar_dataset.py` | Script que gerou `dados/` a partir do dataset original. |
| `scripts/gerar_cor_real.py` | Script que gerou `dados_cor_real/` a partir dos mesmos recortes. |
| `plano_de_aula.md` | Plano de aula (objetivos, sequência didática, avaliação). |
| `atividade/` | Atividade avaliativa, gabarito e critérios de correção. |
| `pdf/` | Versões para impressão. |

## Dados

As imagens derivam do conjunto público **HLS Multi-Temporal Crop Classification** (IBM/NASA e Clark
University), licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): imagens Harmonized
Landsat-Sentinel dos EUA em 2022, com rótulos do Cropland Data Layer do USDA.

> Cecil, M.; Kordi, F.; Li, H.; Khallaghi, S.; Alemohammad, H. (2023). *HLS Multi Temporal Crop Classification*.
> https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification · DOI 10.57967/hf/0955

Alterações feitas aqui: recorte de janelas de 32×32 pixels em que uma classe ocupa ao menos 70% da área;
agrupamento das 13 classes originais em 4; composição em falsa-cor (infravermelho próximo, infravermelho de
ondas curtas e vermelho) da segunda data da série. A pasta `dados_cor_real/` traz os mesmos recortes na composição em cor real, da mesma data. Treino e teste usam chips diferentes do dataset original.

## Referências

**Leitura principal**

- Goodfellow, I.; Bengio, Y.; Courville, A. *Deep Learning*. MIT Press, 2016. Seção 15.2 (Transfer Learning and Domain Adaptation).
  O livro está disponível gratuitamente, na íntegra, em https://www.deeplearningbook.org.

**Leituras complementares**

- Yosinski, J. et al. How transferable are features in deep neural networks? *NeurIPS*, 2014.
- LeCun, Y.; Bengio, Y.; Hinton, G. Deep learning. *Nature*, 521, 436–444, 2015.
- Em português: [Deep Learning Book](https://www.deeplearningbook.com.br/indice/) (Data Science Academy), gratuito e online.
  Capítulos 40 a 43, sobre redes convolucionais, a partir de
  [Introdução às Redes Neurais Convolucionais](https://www.deeplearningbook.com.br/introducao-as-redes-neurais-convolucionais/),
  e a seção "Transfer Learning" do [capítulo 77](https://www.deeplearningbook.com.br/modelo-bert-para-processamento-de-linguagem-natural/).

## Declaração de uso de IA

Este material foi preparado com apoio de IA generativa (Claude, da Anthropic) na elaboração dos slides, do código e da atividade. O professor definiu o conteúdo, revisou todo o material e é o responsável pelas informações apresentadas.

O código deste repositório pode ser usado e adaptado livremente para fins de ensino. As imagens em `dados/`
seguem a licença CC BY 4.0 do dataset original.
