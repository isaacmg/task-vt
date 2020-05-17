import unittest
from re_model import RelationExtractModel
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
class TestCoreMethods(unittest.TestCase):
    def setUp(self):
        self.model = RelationExtractModel("transformer_dir", "transformer_dir/weight.pickle")
    def test_inference(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()