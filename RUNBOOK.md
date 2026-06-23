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

## 2.1. Colab com Google Drive persistente

Se estiver usando Colab, prefira salvar embeddings e modelos direto no Google
Drive. Assim, se a sessao cair, os artefatos continuam salvos.

Primeira celula no Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Depois, dentro do projeto:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"

python -m mert_genre_classifier -c configs/colab_drive.yaml prepare-data
python -m mert_genre_classifier -c configs/colab_drive.yaml inspect-audio --split train --index 0
python -m mert_genre_classifier -c configs/colab_drive.yaml extract-embeddings --split train --resume
python -m mert_genre_classifier -c configs/colab_drive.yaml extract-embeddings --split test --resume
python -m mert_genre_classifier -c configs/colab_drive.yaml repair-labels
python -m mert_genre_classifier -c configs/colab_drive.yaml train
python -m mert_genre_classifier -c configs/colab_drive.yaml evaluate --model mlp
```

Os arquivos ficarao em:

```text
/content/drive/MyDrive/mert-genre-classifier/data/processed
/content/drive/MyDrive/mert-genre-classifier/models
/content/drive/MyDrive/mert-genre-classifier/reports
```

Se a sessao cair, monte o Drive de novo, reinstale o projeto e rode o mesmo
`extract-embeddings --resume`. O cache parcial sera lido do Drive.

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

Se voce usou `configs/colab_drive.yaml`, copie esses arquivos a partir de:

```text
/content/drive/MyDrive/mert-genre-classifier/
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

Para apresentação ao vivo, veja [DEMO_AO_VIVO.md](DEMO_AO_VIVO.md).

## 5. Troubleshooting: coluna de audio do Hugging Face

Se aparecer um erro parecido com:

```text
Valor de audio inesperado; esperado dict com `array` e `sampling_rate`.
```

atualize o codigo e reinstale o pacote no ambiente remoto:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

O pipeline tenta converter a coluna `audio` com `datasets.Audio` e tambem
aceita valores brutos como `AudioDecoder`, caminho, bytes ou waveform.

Para testar a decodificacao sem carregar o MERT:

```bash
python -m mert_genre_classifier -c configs/full_cuda.yaml inspect-audio --split train --index 0
```

## 6. Troubleshooting: matriz de confusao com accuracy 0

Se a matriz de confusao mostrar quase tudo na linha `Unknown`, ou `accuracy` e
`macro_f1` forem `0.0`, os embeddings provavelmente foram gerados com uma
versao antiga do mapeamento de labels. Atualize o codigo e repare os labels dos
embeddings ja calculados, sem recalcular o MERT:

```bash
python -m pip install -e ".[dev]"
python -m mert_genre_classifier -c configs/full_cuda.yaml repair-labels
python -m mert_genre_classifier -c configs/full_cuda.yaml train
python -m mert_genre_classifier -c configs/full_cuda.yaml evaluate --model mlp
```

O comando `repair-labels` cria um backup `.before-label-repair.npz` antes de
reescrever os labels no arquivo de embeddings.

O pipeline usa `genre_id` antes do texto `genre`, porque algumas versoes do
dataset podem retornar `genre` como `Unknown` mesmo quando o identificador da
classe esta correto.
