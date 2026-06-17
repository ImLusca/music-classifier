# Runbook

Este projeto separa a etapa pesada de inferencia MERT da etapa leve de treino do
classificador. A extracao de embeddings e a parte que pode demorar muito em CPU.

## 1. Teste local rapido

Use a config pequena para validar ambiente, download do dataset, cache e treino:

```bash
python -m mert_genre_classifier -c configs/local_smoke.yaml prepare-data
python -m mert_genre_classifier -c configs/local_smoke.yaml extract-embeddings --resume
python -m mert_genre_classifier -c configs/local_smoke.yaml train
python -m mert_genre_classifier -c configs/local_smoke.yaml evaluate
```

Essa execucao usa poucos exemplos por split. Ela serve para provar que o
pipeline esta funcionando, nao para medir qualidade final.

## 2. Extracao completa em maquina remota com CUDA

Na maquina com GPU:

```bash
git clone <repo> mert-genre-classifier
cd mert-genre-classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

python -m mert_genre_classifier -c configs/full_cuda.yaml prepare-data
python -m mert_genre_classifier -c configs/full_cuda.yaml extract-embeddings --split train --resume
python -m mert_genre_classifier -c configs/full_cuda.yaml extract-embeddings --split test --resume
```

Se o processo cair no meio, rode o mesmo comando com `--resume`. O cache parcial
sera reutilizado quando possivel.

## 3. Copia e reuso dos embeddings

Os artefatos portaveis ficam em:

```text
data/processed/
models/
reports/
```

Para treinar localmente depois de extrair em CUDA, copie pelo menos:

```text
data/processed/full_cuda_*_mert-v1-95m_embeddings.npz
data/processed/full_cuda_labels.json
data/processed/full_cuda_prepare_summary.json
```

Depois, na maquina local:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml train
python -m mert_genre_classifier -c configs/full_cuda.yaml evaluate
```

## 4. Predicao de um audio local

Depois de treinar:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml predict caminho/para/musica.wav --model mlp --top-k 5
```

O comando carrega MERT para extrair o embedding da faixa e usa o classificador
salvo em `models/`.

## 5. Aviso sobre CPU

Se `extract-embeddings` for executado em CPU com a config completa, o CLI emite
um aviso. O comando ainda pode rodar, mas tende a ser lento. Para resultados
finais, prefira CUDA.

