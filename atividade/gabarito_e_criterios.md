# Gabarito e critérios de correção

**Atividade avaliativa: Transfer Learning** · Aprendizagem Profunda · Prof. Rogério Gomes Lopes

## Objetivos de aprendizagem verificados

| Questão | Objetivo | Nível (Bloom) | Pontos |
|---|---|---|---|
| 1 | Reconhecer o que é reaproveitado e o que é treinado em cada estratégia | Compreender | 2 |
| 2 | Escolher a estratégia de transferência conforme os dados e o domínio | Analisar / Avaliar | 4 |
| 3 | Implementar e explicar o congelamento e o fine-tuning em PyTorch | Aplicar | 4 |

## Questão 1 (2 pontos; 0,5 por item)

**a) V.** As primeiras camadas respondem a padrões simples (bordas, cores), comuns a qualquer imagem.<br>
**b) F.** Na extração de características os pesos transferidos ficam fixos; só a nova camada de saída é treinada. A frase descreve o fine-tuning da rede inteira.<br>
**c) V.**<br>
**d) V.** A saída original tem 1.000 neurônios, um por classe do ImageNet.

## Questão 2 (4 pontos)

Em cada item, metade da nota é para a **estratégia** e metade para a **justificativa** com os dois critérios
(quantidade de dados e semelhança de domínio). Aceitar respostas equivalentes bem justificadas.

**a) Poucos dados, domínio diferente (1,5).** Congelar as camadas iniciais (características gerais), trocar a
`fc` e ajustar o último bloco com taxa de aprendizado baixa, usando aumento de dados. Justificativa: 600
imagens não bastam para treinar a rede inteira sem sobreajuste, mas imagens de satélite diferem das fotos do
ImageNet, então as características mais específicas (últimas camadas) precisam ser adaptadas.
*Também aceitar:* começar só com a nova `fc` (extração de características) e avaliar se o fine-tuning melhora.

**b) Poucos dados, domínio parecido (1,0).** Congelar todo o backbone e treinar apenas a nova camada final (2
saídas). Justificativa: fotos de folhas feitas com celular são fotografias comuns, parecidas com as do
ImageNet; as características pré-treinadas já servem, e com 800 imagens treinar poucos parâmetros evita sobreajuste.

**c) Muitos dados, domínio diferente (1,5).** Fine-tuning amplo (vários blocos ou a rede inteira), usando os
pesos pré-treinados como ponto de partida; treinar do zero também é defensável. Justificativa: com 200 mil
imagens há dados suficientes para adaptar todas as camadas ao novo domínio sem sobreajuste.

| Desempenho no item | Fração da nota |
|---|---|
| Estratégia correta e justificativa com os dois critérios | 100% |
| Estratégia correta, justificativa com apenas um critério | 75% |
| Estratégia correta sem justificativa, ou justificativa correta com estratégia incompleta | 50% |
| Apenas cita termos (congelar, fine-tuning) sem relacionar ao cenário | 25% |
| Em branco ou incoerente | 0% |

## Questão 3 (4 pontos)

**a) (1 ponto)** Apenas a nova `fc`: 512 × 4 pesos + 4 vieses = **2.052 parâmetros**.
0,5 pelo raciocínio (só a `fc` é treinada) e 0,5 pelo cálculo. Quem esquecer os vieses (2.048) recebe 0,75.

**b) (2 pontos)**

```python
for p in model.layer4.parameters():
    p.requires_grad = True

otimizador = torch.optim.Adam([
    {"params": model.fc.parameters(),     "lr": 1e-3},
    {"params": model.layer4.parameters(), "lr": 1e-4},
])
```

1,0 por descongelar o `layer4` (`requires_grad = True`); 1,0 por incluir os parâmetros do `layer4` no
otimizador. Aceitar um único grupo de parâmetros com taxa baixa (por exemplo, `lr=1e-4` para todos).
Erros de sintaxe que não comprometam a lógica não são descontados.

**c) (1 ponto)** Os pesos do `layer4` já contêm conhecimento útil do pré-treino; passos grandes do gradiente
os desfariam (esquecimento do que foi aprendido) e aumentariam o risco de sobreajuste com poucos dados. A `fc`
é nova, começa com valores aleatórios e precisa de passos maiores para aprender. 0,5 para cada ideia
(preservar o pré-treino; a `fc` parte do zero).

## Devolutiva à turma

Corrigir a questão 2 coletivamente, projetando a matriz de decisão da aula, e pedir a dois alunos que
defendam respostas diferentes para o item (a). Na questão 3, executar a resposta no notebook da aula e
comparar a acurácia antes e depois do fine-tuning.
