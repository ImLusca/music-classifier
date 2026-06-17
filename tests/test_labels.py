import unittest

from mert_genre_classifier.config import AppConfig
from mert_genre_classifier.dataset import build_label_map, label_index_for_row


class LabelMapTests(unittest.TestCase):
    def test_build_label_map_orders_by_source_id(self):
        rows = [
            {"genre": "Rock", "genre_id": 2},
            {"genre": "Electronic", "genre_id": 0},
            {"genre": "Jazz", "genre_id": 1},
            {"genre": "Rock", "genre_id": 2},
        ]

        label_map = build_label_map(rows, label_column="genre", label_id_column="genre_id")

        self.assertEqual(label_map["labels"], ["Electronic", "Jazz", "Rock"])
        self.assertEqual(label_map["label_to_index"]["Rock"], 2)
        self.assertEqual(label_map["source_id_to_index"]["0"], 0)

    def test_label_index_for_row_prefers_label_text(self):
        config = AppConfig()
        label_map = {
            "labels": ["Electronic", "Rock"],
            "label_to_index": {"Electronic": 0, "Rock": 1},
            "source_id_to_index": {"7": 1},
        }

        index = label_index_for_row({"genre": "Rock", "genre_id": 7}, label_map, config)

        self.assertEqual(index, 1)


if __name__ == "__main__":
    unittest.main()

