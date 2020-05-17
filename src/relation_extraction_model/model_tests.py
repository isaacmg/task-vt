import unittest
from re_model import RelationExtractModel
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent / "transformer_dir"
class TestCoreMethods(unittest.TestCase):
    def setUp(self):
        self.model = RelationExtractModel(BASE_DIR, BASE_DIR / "weight.pickle")
    def test_inference(self):
        self.assertTrue(self.model.predict("Here is some text"))

if __name__ == '__main__':
    unittest.main()
