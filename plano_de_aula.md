# Plano de aula: Transfer Learning aplicado à classificação de lavouras

**Curso:** Tecnologia da Informação · **Disciplina:** Aprendizagem Profunda<br>
**Professor:** Rogério Gomes Lopes · **Duração:** 15 minutos (miniaula) · **Data:** 27/11/2026<br>
**Tópico da ementa:** transferência de aprendizado e fine-tuning para domínios específicos<br>
**Pré-requisito da turma:** redes convolucionais (convolução, pooling, camadas densas)

## Objetivos de aprendizagem

Ao final da aula, o estudante será capaz de:

1. **Definir** transfer learning e **diferenciar** extração de características de fine-tuning.
2. **Explicar** por que as camadas iniciais de uma CNN são reaproveitáveis e as finais são específicas da tarefa.
3. **Decidir** o que congelar e o que treinar, conforme a quantidade de dados e a semelhança de domínio.
4. **Identificar**, em código PyTorch, as linhas que congelam a rede, trocam a camada final e liberam um bloco para ajuste.

## Sequência didática

1. **Problematização:** apresentar o desafio de classificar lavouras em imagens de satélite com poucos exemplos rotulados.
2. **Conceitos:** definir transfer learning e fine-tuning, retomar como uma CNN aprende características e apresentar a ResNet-18 pré-treinada.
3. **Aplicação:** mostrar onde congelar e onde retreinar a rede, o código em PyTorch e o resultado do experimento.
4. **Decisão e síntese:** discutir quando usar cada estratégia, retomar os objetivos e entregar a atividade.

## Metodologia

Aula expositiva dialogada, partindo de um problema real. Os conceitos são apresentados com apoio visual e
perguntas à turma, e a teoria é ligada à prática por meio de um exemplo de código e da análise de seus resultados.

## Recursos

Projetor com entrada HDMI, acesso à internet (para demonstrar o notebook no Google Colab), quadro branco e pincéis.
Slides, notebook e dados em repositório público no GitHub.

## Avaliação

- **Formativa, durante a aula:** perguntas de verificação (qual parte da rede se reaproveita; em qual quadrante
  da matriz está o problema das lavouras).
- **Atividade avaliativa (10 pontos, 20 minutos):** estudo de caso com três questões, sobre conceitos (2 pontos),
  decisão de estratégia em três cenários (4 pontos) e leitura e escrita de código (4 pontos). Gabarito e
  critérios de correção em documento anexo.

## Referências

**Principal**

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge: MIT Press, 2016. Seção 15.2. Disponível em: https://www.deeplearningbook.org.

**Complementares**

- YOSINSKI, J. et al. How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems* 27, 2014.
- LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. *Nature*, v. 521, p. 436–444, 2015.
- DATA SCIENCE ACADEMY. *Deep Learning Book*, capítulos 40 a 43 (redes convolucionais). Disponível em: https://www.deeplearningbook.com.br.

---

**Declaração de uso de IA.** Este material foi preparado com apoio de IA generativa (Claude, da Anthropic) na elaboração dos slides, do código e da atividade. O professor definiu o conteúdo, revisou todo o material e é o responsável pelas informações apresentadas.
