# Perguntas de Preparação

Este guia reúne perguntas prováveis sobre o projeto **MERT Genre Classifier** e respostas curtas para apresentação, banca, entrevista ou demo.

## Resumo em 30 segundos

O projeto classifica gêneros musicais usando o MERT como extrator de embeddings de áudio. O MERT fica congelado, cada música vira um vetor, e classificadores leves, como regressão logística e MLP, aprendem a mapear esses vetores para gêneros. A parte pesada é extrair embeddings; por isso o pipeline tem cache, `--resume` e uma configuração pronta para Colab/CUDA.

## Perguntas Gerais

### 1. Qual problema o projeto resolve?

Ele automatiza a classificação de gêneros musicais a partir do áudio, sem depender de metadados textuais ou tags manuais.

### 2. Qual é a ideia principal?

Usar um modelo pré-treinado em música, o MERT, para gerar embeddings ricos de cada faixa e treinar um classificador menor por cima.

### 3. Por que usar MERT?

Porque ele foi treinado especificamente para entendimento musical. Ele tende a capturar timbre, ritmo, textura, pitch e padrões temporais melhor do que embeddings genéricos.

### 4. O MERT gera música?

Não. Neste projeto, ele é usado como extrator de características. A geração musical não faz parte do escopo.

### 5. Por que não treinar uma rede do zero?

Treinar do zero exigiria muito mais dados, tempo e GPU. Usar MERT congelado aproveita conhecimento já aprendido e reduz o custo do experimento.

## Pipeline

### 6. Como o pipeline funciona?

1. Carrega o dataset `lewtun/music_genres`.
2. Valida labels e splits.
3. Testa a decodificação de uma amostra com `inspect-audio`.
4. Reamostra o áudio para 24 kHz.
5. Extrai embeddings com `m-a-p/MERT-v1-95M`.
6. Salva embeddings em cache `.npz`.
7. Treina regressão logística e MLP.
8. Avalia com macro-F1, accuracy, matriz de confusão e relatório por gênero.

### 7. O que é um embedding?

É um vetor numérico que resume características relevantes do áudio. Em vez de classificar a waveform diretamente, o classificador aprende sobre esses vetores.

### 8. Por que salvar embeddings em cache?

Porque extrair embeddings com MERT é a etapa lenta. Depois de salvar os vetores, dá para treinar e avaliar classificadores rapidamente sem passar o áudio pelo MERT de novo.

### 9. Para que serve o `--resume`?

Ele permite retomar a extração se a execução cair no meio, reaproveitando cache parcial quando disponível.

### 10. Para que serve `inspect-audio`?

Serve para confirmar que o dataset está entregando áudio decodificável antes de carregar o MERT e iniciar uma execução longa.

## Modelo e Treinamento

### 11. Qual versão do MERT foi escolhida?

`m-a-p/MERT-v1-95M`, por ser mais leve que a versão 330M e adequada para começar com menos custo computacional.

### 12. O MERT é fine-tuned?

Não na versão atual. Ele fica congelado. Só os classificadores por cima dos embeddings são treinados.

### 13. Quais classificadores são usados?

Regressão logística como baseline e uma MLP pequena como modelo não linear simples.

### 14. Por que usar regressão logística?

Ela é um baseline rápido, forte e interpretável. Se ela performar bem, significa que os embeddings já separam os gêneros razoavelmente.

### 15. Por que usar MLP?

A MLP consegue capturar relações não lineares nos embeddings sem o custo de fine-tuning do MERT.

### 16. A etapa de treinamento é pesada?

Normalmente não. A etapa pesada é `extract-embeddings`. Depois que os embeddings existem, treinar regressão logística e MLP tende a ser bem mais rápido.

## Dados e Avaliação

### 17. Qual dataset é usado?

O dataset público `lewtun/music_genres`, com áudio, `song_id`, `genre_id` e `genre`.

### 18. Por que usar macro-F1 como métrica principal?

Porque macro-F1 dá peso semelhante para todas as classes, evitando que gêneros mais frequentes escondam desempenho ruim em classes menores.

### 19. Por que também reportar accuracy?

Accuracy é simples de entender e ajuda na leitura geral, mas não deve ser a única métrica se houver desbalanceamento.

### 20. O que a matriz de confusão mostra?

Ela mostra quais gêneros o modelo confunde entre si, ajudando a identificar limites do dataset ou do classificador.

### 21. O problema de gênero musical é sempre single-label?

Não necessariamente. Muitas músicas misturam estilos. A versão atual trata como classificação single-label porque o dataset fornece um gênero principal.

## Execução Local, Colab e CUDA

### 22. Por que usar Colab/CUDA?

Porque inferir embeddings MERT para milhares de faixas pode ser lento em CPU. GPU reduz bastante o tempo da etapa pesada.

### 23. O que roda localmente?

O smoke test local roda com poucos exemplos para validar instalação, pipeline, cache, treino e avaliação.

### 24. O que roda melhor em GPU?

Principalmente `extract-embeddings` no dataset completo.

### 25. Dá para extrair embeddings em uma máquina e treinar em outra?

Sim. Os embeddings `.npz`, labels e relatórios são artefatos portáveis.

### 26. Por que apareceu `AudioDecoder` no Colab?

Versões recentes do `datasets` usam `torchcodec` para decodificar áudio e podem retornar um objeto `AudioDecoder`. O pipeline foi ajustado para ler esse formato.

## Limitações

### 27. Quais são as principais limitações?

- O modelo depende da qualidade e dos rótulos do dataset.
- Gêneros musicais podem ser ambíguos.
- A versão atual não faz fine-tuning do MERT.
- A avaliação inicial depende do split público usado.
- A licença do MERT precisa ser respeitada, especialmente para uso comercial.

### 28. O modelo pode generalizar mal para músicas fora do dataset?

Pode. A generalização depende da diversidade do dataset, da qualidade dos rótulos e da proximidade entre as músicas novas e os dados de treino.

### 29. Como lidar com gêneros muito parecidos?

Analisar a matriz de confusão, considerar labels mais amplos, aumentar dados, testar classificadores diferentes ou migrar para classificação multi-label.

### 30. Por que não usar só espectrograma e CNN?

É uma alternativa válida. A escolha por MERT aproveita um modelo pré-treinado em grande escala e simplifica o pipeline inicial.

## Perguntas Difíceis

### 31. Como você sabe que o MERT está ajudando?

Comparando contra baselines: por exemplo, embeddings simples de áudio, MFCCs, ou uma classificação aleatória/majoritária. A regressão logística sobre MERT já funciona como um bom primeiro teste.

### 32. Como evitar vazamento de dados?

Usando splits separados, treinando apenas no split de treino, avaliando no split de teste e evitando gerar estatísticas de normalização com dados de avaliação.

### 33. O que faria se a macro-F1 fosse baixa?

Eu olharia a matriz de confusão, distribuição por classe, qualidade dos áudios, labels ruidosos e experimentaria pooling, camadas diferentes do MERT, balanceamento e modelos de classificação alternativos.

### 34. O que mudaria numa versão de produção?

Criaria uma API, versionaria modelos e embeddings, adicionaria monitoramento, validação de arquivos de áudio, limites de duração e logging estruturado.

### 35. Qual seria a próxima melhoria técnica?

Comparar camadas do MERT, testar pooling mais rico, treinar um classificador calibrado, medir top-k e, se houver GPU suficiente, experimentar fine-tuning controlado.

## Roteiro de Demo

1. Abrir `site/index.html` e explicar o objetivo.
2. Mostrar o fluxo: dataset, diagnóstico, embeddings, classificador e avaliação.
3. Rodar ou mostrar o comando:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml inspect-audio --split train --index 0
```

4. Mostrar a extração em Colab/CUDA:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml extract-embeddings --split train --resume
```

5. Mostrar os artefatos em `data/processed/`, `models/` e `reports/`.
6. Explicar macro-F1 e matriz de confusão.
7. Fechar com limitações e próximos passos.

## Frases Úteis

- "A decisão principal foi separar representação de classificação."
- "O MERT fica congelado; isso reduz custo e risco de overfitting."
- "A extração é cara, então o cache é parte central do desenho."
- "Macro-F1 é a métrica principal porque trata melhor classes desbalanceadas."
- "O projeto está preparado para rodar localmente em smoke test e escalar a extração em Colab/CUDA."

