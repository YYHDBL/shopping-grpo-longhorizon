import unittest

from scripts.dryrun_shop_interaction import run_fake_dryrun


class DryrunShopInteractionTest(unittest.TestCase):
    def test_fake_dryrun_executes_search_tool(self):
        result = run_fake_dryrun(query="乳胶枕")

        self.assertEqual(result["tool"], "search_products")
        self.assertEqual(result["action"], "search[乳胶枕]")
        self.assertEqual(result["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
