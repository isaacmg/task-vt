import unittest
from re_model import RelationExtractModel
class TestCoreMethods(unittest.TestCase):
    def setUp(self):
        self.model = RelationExtractModel("test_data", "test_data/weight.pickle")
    def test_inference(sef):
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()