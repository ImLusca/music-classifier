from __future__ import annotations

import argparse
import json
from typing import Any

from .config import load_config
from .dataset import configured_splits, inspect_audio_sample, prepare_data
from .embeddings import extract_embeddings
from .evaluate import evaluate_classifiers
from .models import train_classifiers
from .paths import labels_path, prepare_summary_path
from .predict import predict_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mert-genre",
        description="Classificador de generos musicais com embeddings MERT congelados.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/local_smoke.yaml",
        help="Caminho para config YAML.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Carrega e valida dataset/labels.")
    prepare.add_argument("--split", help="Split especifico. Omitido valida train/test da config.")
    prepare.add_argument("--max-samples", type=int, help="Limita exemplos por split.")

    inspect = subparsers.add_parser("inspect-audio", help="Inspeciona uma amostra de audio sem carregar MERT.")
    inspect.add_argument("--split", help="Split a inspecionar. Padrao: train_split da config.")
    inspect.add_argument("--index", type=int, default=0, help="Indice da amostra no split.")

    extract = subparsers.add_parser("extract-embeddings", help="Extrai embeddings MERT em cache.")
    extract.add_argument("--split", help="Split especifico. Omitido extrai train/test da config.")
    extract.add_argument("--max-samples", type=int, help="Limita exemplos do split.")
    _add_resume_flags(extract)

    train = subparsers.add_parser("train", help="Treina classificadores sobre embeddings.")
    train.add_argument("--train-split", help="Split de treino. Padrao: train_split da config.")
    train.add_argument("--model", default="all", help="`logistic`, `mlp` ou `all`.")
    train.add_argument("--max-samples", type=int, help="Limita embeddings de treino.")

    evaluate = subparsers.add_parser("evaluate", help="Avalia classificadores salvos.")
    evaluate.add_argument("--split", help="Split de avaliacao. Padrao: eval_split da config.")
    evaluate.add_argument("--model", default="all", help="`logistic`, `mlp` ou `all`.")
    evaluate.add_argument("--max-samples", type=int, help="Limita embeddings de avaliacao.")

    predict = subparsers.add_parser("predict", help="Prediz genero de um arquivo local.")
    predict.add_argument("audio_path", help="Caminho para arquivo de audio.")
    predict.add_argument("--model", default="mlp", help="Classificador salvo a usar.")
    predict.add_argument("--top-k", type=int, default=5, help="Numero de classes a mostrar.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "prepare-data":
        summary = prepare_data(config, split=args.split, max_samples=args.max_samples)
        _print_json(
            {
                "summary_path": str(prepare_summary_path(config)),
                "labels_path": str(labels_path(config)),
                "dataset": summary["dataset"],
                "splits": summary["splits"],
                "num_labels": summary["num_labels"],
                "labels": summary["labels"],
            }
        )
        return 0

    if args.command == "inspect-audio":
        _print_json(inspect_audio_sample(config, split=args.split, index=args.index))
        return 0

    if args.command == "extract-embeddings":
        splits = [args.split] if args.split else configured_splits(config)
        paths = []
        for split in splits:
            path = extract_embeddings(
                config,
                split=split,
                max_samples=args.max_samples,
                resume=args.resume,
            )
            paths.append(str(path))
        _print_json({"embedding_paths": paths})
        return 0

    if args.command == "train":
        saved = train_classifiers(
            config,
            train_split=args.train_split,
            model_name=args.model,
            max_samples=args.max_samples,
        )
        _print_json({"models": {name: str(path) for name, path in saved.items()}})
        return 0

    if args.command == "evaluate":
        results = evaluate_classifiers(
            config,
            split=args.split,
            model_name=args.model,
            max_samples=args.max_samples,
        )
        _print_json(
            {
                name: {
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "num_examples": metrics["num_examples"],
                }
                for name, metrics in results.items()
            }
        )
        return 0

    if args.command == "predict":
        result = predict_audio(
            config,
            audio_path=args.audio_path,
            model_name=args.model,
            top_k=args.top_k,
        )
        _print_json(result)
        return 0

    parser.error(f"Comando desconhecido: {args.command}")
    return 2


def _add_resume_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", dest="resume", action="store_true", default=None)
    group.add_argument("--no-resume", dest="resume", action="store_false")


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
