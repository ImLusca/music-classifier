import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    np = None

from mert_genre_classifier.config import AppConfig, ArtifactsConfig, ClassifierConfig
from mert_genre_classifier.evaluate import evaluate_classifiers
from mert_genre_classifier.models import train_classifiers
from mert_genre_classifier.paths import embeddings_path


@unittest.skipIf(np is None, "numpy/scikit-learn are not installed")
class TrainEvaluateSyntheticTests(unittest.TestCase):
    def test_train_and_evaluate_on_synthetic_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig(
                run_name="synthetic",
                artifacts=ArtifactsConfig(
                    processed_dir=root / "processed",
                    models_dir=root / "models",
                    reports_dir=root / "reports",
                ),
                classifier=ClassifierConfig(
                    mlp_hidden_layer_sizes=(8,),
                    mlp_max_iter=50,
                    mlp_early_stopping=False,
                ),
            )
            config.ensure_artifact_dirs()

            labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
            embeddings = np.array(
                [
                    [0.0, 0.1],
                    [0.1, 0.0],
                    [0.2, 0.1],
                    [2.0, 2.1],
                    [2.1, 2.0],
                    [2.2, 2.1],
                ],
                dtype=np.float32,
            )
            for split in (config.dataset.train_split, config.dataset.eval_split):
                np.savez_compressed(
                    embeddings_path(config, split),
                    embeddings=embeddings,
                    labels=labels,
                    item_ids=np.array([str(i) for i in range(len(labels))]),
                    label_names=np.array(["A", "B"]),
                )

            saved = train_classifiers(config, model_name="logistic")
            results = evaluate_classifiers(config, model_name="logistic")

        self.assertIn("logistic", saved)
        self.assertGreaterEqual(results["logistic"]["macro_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()

