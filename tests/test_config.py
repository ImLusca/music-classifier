import tempfile
import unittest
from pathlib import Path

try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    yaml = None

from mert_genre_classifier.config import load_config


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class ConfigTests(unittest.TestCase):
    def test_load_config_converts_paths_and_hidden_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
run_name: test_run
seed: 123
classifier:
  mlp_hidden_layer_sizes: [32, 16]
artifacts:
  processed_dir: tmp_processed
  models_dir: tmp_models
  reports_dir: tmp_reports
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.run_name, "test_run")
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.classifier.mlp_hidden_layer_sizes, (32, 16))
        self.assertEqual(config.artifacts.processed_dir, Path("tmp_processed"))


if __name__ == "__main__":
    unittest.main()

