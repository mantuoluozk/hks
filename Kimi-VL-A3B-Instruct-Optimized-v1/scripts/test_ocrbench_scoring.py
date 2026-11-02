"""Synthetic checks of the upstream OCRBench normalization rules; not accuracy."""
import unittest
from eval_ocrbench import grade


class ScoringTests(unittest.TestCase):
    def test_case_insensitive_text_and_answer_alias(self):
        row = {"dataset": "IIIT5K", "answer": ["Centre", "Center"]}
        self.assertTrue(grade(row, {"output": "The word is CENTER.", "error": None}))

    def test_mathematical_case_is_preserved(self):
        row = {"dataset": "HME100k", "answer": ["x + Y"]}
        self.assertTrue(grade(row, {"output": "x+Y", "error": None}))
        self.assertFalse(grade(row, {"output": "x+y", "error": None}))

    def test_errors_do_not_receive_credit(self):
        row = {"dataset": "IIIT5K", "answer": ["CENTRE"]}
        self.assertFalse(grade(row, {"output": "CENTRE", "error": "timeout"}))

    def test_normal_text_spaces_are_not_deleted(self):
        row = {"dataset": "docVQA", "answer": ["New York"]}
        self.assertTrue(grade(row, {"output": "NEW\nYORK", "error": None}))
        self.assertFalse(grade(row, {"output": "NewYork", "error": None}))


if __name__ == "__main__":
    unittest.main()
