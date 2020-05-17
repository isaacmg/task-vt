import unittest
from re_model import RelationExtractModel
class TestCoreMethods(unittest.TestCase):
    def setUp(self):
        self.model = RelationExtractModel("test_data", "test_data/weight.pickle")
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

if __name__ == '__main__':
    unittest.main()