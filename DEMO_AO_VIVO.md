# Demo ao Vivo

Este roteiro é para apresentar o projeto localmente na feira, usando embeddings
calculados no Colab.

## 1. Copiar artefatos do Colab

Traga estes arquivos para a mesma estrutura local:

```text
data/processed/full_cuda_train_mert-v1-95m_embeddings.npz
data/processed/full_cuda_test_mert-v1-95m_embeddings.npz
data/processed/full_cuda_labels.json
data/processed/full_cuda_prepare_summary.json
```

Se já tiver treinado no Colab, copie também:

```text
models/full_cuda_logistic.joblib
models/full_cuda_mlp.joblib
reports/
```

Se usou `configs/colab_drive.yaml`, os arquivos ficam persistidos em:

```text
/content/drive/MyDrive/mert-genre-classifier/
```

## 2. Checar se a demo está pronta

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml demo-status
```

O ideal é ver:

```text
ready_for_train: true
ready_for_evaluate: true
ready_for_predict: true
```

Se `ready_for_train` estiver `true`, mas os classificadores ainda não existirem,
rode:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml train
python -m mert_genre_classifier -c configs/full_cuda.yaml evaluate
```

## 3. Demo recomendada para feira

### Parte A: explicar a ideia

Abra:

```text
site/publico.html
```

Explique em três frases:

1. O sistema lê o áudio, não apenas o nome do arquivo.
2. O MERT transforma a música em uma assinatura numérica.
3. Um classificador usa essa assinatura para sugerir o gênero.

### Parte B: mostrar que os dados estão prontos

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml demo-status
```

### Parte C: mostrar avaliação

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml evaluate --model mlp
```

Mostre o `macro_f1`, a `accuracy` e explique que a matriz de confusão fica em
`reports/`.

### Parte D: predizer uma música nova

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml predict caminho/para/musica.wav --model mlp --top-k 5
```

Para essa parte funcionar sem susto, teste antes com os mesmos arquivos que
pretende usar na feira.

## 4. Como deixar a predição ao vivo mais segura

- Use áudios curtos, de preferência 20 a 40 segundos.
- Tenha 2 ou 3 arquivos `.wav` já preparados em uma pasta `sample_data/`.
- Rode uma predição completa antes da apresentação para baixar/cachear o MERT.
- Mantenha a internet disponível, mas não dependa dela na hora.
- Se a predição demorar, explique que a parte pesada é transformar o áudio em
  assinatura numérica.

## 5. Plano B sem internet ou sem MERT local

Se a predição de áudio novo não funcionar localmente, ainda dá para fazer uma
demo boa:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml demo-status
python -m mert_genre_classifier -c configs/full_cuda.yaml evaluate --model mlp
```

Nesse modo, você mostra que os embeddings foram calculados no Colab, que o
classificador foi treinado e que os resultados estão salvos em `reports/`.
