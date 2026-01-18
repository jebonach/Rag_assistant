import unittest

from src.subchunking import split_for_embeddings


class TestSubchunking(unittest.TestCase):
    def test_split_max_chars(self) -> None:
        text = ("a" * 5000) + " [ 1 ]"
        max_chars = 1000
        overlap = 100
        chunks = split_for_embeddings(text, max_chars=max_chars, overlap=overlap, page=1)
        self.assertTrue(chunks, "Expected at least one subchunk")
        for ch in chunks:
            self.assertLessEqual(len(ch["text"]), max_chars)


if __name__ == "__main__":
    unittest.main()
