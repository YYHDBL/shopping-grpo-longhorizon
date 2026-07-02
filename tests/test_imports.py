import unittest


class ImportsTest(unittest.TestCase):
    def test_package_imports(self):
        import shopping_grpo

        self.assertTrue(shopping_grpo.__version__)


if __name__ == "__main__":
    unittest.main()
