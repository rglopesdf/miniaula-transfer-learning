# Atividade avaliativa: Transfer Learning

**Disciplina:** Aprendizagem Profunda · **Tema:** transferência de aprendizado e fine-tuning<br>
**Professor:** Rogério Gomes Lopes · **Duração:** 20 minutos · **Valor:** 10 pontos · Individual, sem consulta

**Nome:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ **Data:** \_\_\_\_ / \_\_\_\_ / \_\_\_\_\_\_

---

## Estudo de caso

Uma cooperativa agrícola quer classificar, a partir de imagens de satélite, recortes de 1 km × 1 km em
quatro classes: **soja**, **milho**, **outras culturas** e **sem lavoura**. A equipe dispõe de **600 imagens
rotuladas** (150 por classe) e decidiu partir de uma **ResNet-18 pré-treinada no ImageNet** (1,28 milhão de
fotografias do cotidiano, 1.000 classes).

## Questão 1 · Conceitos (2 pontos)

Marque **V** (verdadeiro) ou **F** (falso). Cada item vale 0,5 ponto.

( &nbsp; ) **a)** As primeiras camadas de uma CNN aprendem características gerais, como bordas e cores, que servem a muitas tarefas.

( &nbsp; ) **b)** Na extração de características, todos os pesos da rede são atualizados, com taxa de aprendizado baixa.

( &nbsp; ) **c)** No fine-tuning, os pesos pré-treinados são o ponto de partida e continuam a ser ajustados na nova tarefa.

( &nbsp; ) **d)** A camada final da ResNet-18 precisa ser substituída porque ela foi treinada para 1.000 classes, e a nova tarefa tem 4.

## Questão 2 · Decisão (4 pontos)

Para cada cenário, indique **o que você congelaria e o que treinaria** e **justifique** com base em dois
critérios: a quantidade de dados rotulados e a semelhança entre as imagens do problema e as do pré-treino.

**a)** A cooperativa do estudo de caso: 600 imagens de satélite. *(1,5 ponto)*

<br><br><br>

**b)** Uma startup com 800 fotografias de folhas de soja, tiradas com celular, para classificar em
"saudável" ou "doente". *(1,0 ponto)*

<br><br><br>

**c)** Um órgão público com 200 mil imagens de satélite rotuladas, para as mesmas quatro classes. *(1,5 ponto)*

<br><br><br>

## Questão 3 · Código (4 pontos)

Considere o trecho em PyTorch:

```python
model = resnet18(weights="IMAGENET1K_V1")

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(512, 4)

otimizador = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**a)** Quantos parâmetros serão atualizados durante o treino? Mostre o cálculo. *(1 ponto)*

<br><br>

**b)** Escreva as linhas que faltam para também fazer **fine-tuning do último bloco** (`model.layer4`),
incluindo a mudança necessária no otimizador. *(2 pontos)*

<br><br><br><br>

**c)** Por que se recomenda uma taxa de aprendizado **menor** para o `layer4` do que para a nova camada `fc`? *(1 ponto)*

<br><br>
