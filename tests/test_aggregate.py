import unittest

from src.retrievers.aggregate import aggregate_subchunk_hits


class TestAggregate(unittest.TestCase):
    def test_aggregate_max_score_and_snippet(self) -> None:
        hits = [
            {"page": 1, "sub_id": 0, "snippet": "first", "score": 0.2},
            {"page": 1, "sub_id": 1, "snippet": "best", "score": 0.9},
            {"page": 2, "sub_id": 0, "snippet": "other", "score": 0.5},
        ]
        page_text_by_page = {1: "page one text", 2: "page two text"}
        page_chunk_id_by_page = {1: "book_p1", 2: "book_p2"}

        rows = aggregate_subchunk_hits(
            hits=hits,
            page_text_by_page=page_text_by_page,
            page_chunk_id_by_page=page_chunk_id_by_page,
            top_k=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["page"], 1)
        self.assertAlmostEqual(rows[0]["score"], 0.9)
        self.assertEqual(rows[0]["best_snippet"], "best")
        self.assertEqual(rows[0]["text"], "page one text")
        self.assertEqual(rows[0]["chunk_id"], "book_p1")


if __name__ == "__main__":
    unittest.main()
