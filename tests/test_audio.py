import unittest

from mert_genre_classifier.audio import audio_from_dataset_value


class AudioValueTests(unittest.TestCase):
    def test_decoded_hf_audio_mapping(self):
        audio, sample_rate = audio_from_dataset_value(
            {"array": [0.0, 0.1, -0.1], "sampling_rate": 44100}
        )

        self.assertEqual(audio, [0.0, 0.1, -0.1])
        self.assertEqual(sample_rate, 44100)

    def test_waveform_with_fallback_sample_rate(self):
        audio, sample_rate = audio_from_dataset_value([0.0, 0.1], fallback_sample_rate=24000)

        self.assertEqual(audio, [0.0, 0.1])
        self.assertEqual(sample_rate, 24000)

    def test_unexpected_value_has_type_in_error(self):
        with self.assertRaisesRegex(ValueError, "recebido int"):
            audio_from_dataset_value(123)


if __name__ == "__main__":
    unittest.main()

