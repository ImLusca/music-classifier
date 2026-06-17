# MERT Genre Classifier

Pipeline local para classificar generos musicais usando embeddings congelados do
`m-a-p/MERT-v1-95M` e classificadores leves por cima.

O fluxo esperado e:

1. carregar e validar o dataset publico `lewtun/music_genres`;
2. extrair embeddings com MERT e salvar cache em `data/processed/`;
3. treinar regressao logistica e MLP pequeno;
4. avaliar com `macro-F1` como metrica principal;
5. predizer o genero de um arquivo de audio local.

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Teste local rapido

```bash
python -m mert_genre_classifier -c configs/local_smoke.yaml prepare-data
python -m mert_genre_classifier -c configs/local_smoke.yaml extract-embeddings --resume
python -m mert_genre_classifier -c configs/local_smoke.yaml train
python -m mert_genre_classifier -c configs/local_smoke.yaml evaluate
```

Para execucao completa em CUDA, veja [RUNBOOK.md](RUNBOOK.md).

## Página de apresentação

A vitrine estática do projeto está em [site/index.html](site/index.html). Ela
pode ser aberta diretamente no navegador.
