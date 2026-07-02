import unittest

from shopping_grpo.shop_http_env import ShopHttpEnv


class ShopHttpEnvTest(unittest.TestCase):
    def test_build_search_url(self):
        env = ShopHttpEnv(base_url="http://127.0.0.1:7001")
        env.session_id = "fixed_0"

        url = env.build_action_url("search[乳胶枕]")

        self.assertIn("%E4%B9%B3%E8%83%B6%E6%9E%95", url)
        self.assertTrue(url.endswith("/1"))

    def test_build_click_url_rejects_without_page_state(self):
        env = ShopHttpEnv(base_url="http://127.0.0.1:7001")
        env.session_id = "fixed_0"

        self.assertIsNone(env.build_action_url("click[Buy Now]"))

    def test_parse_observation_maps_form_button_to_click_action(self):
        env = ShopHttpEnv(base_url="http://127.0.0.1:7001")
        parsed = env.parse_observation(
            '<form action="/done/fixed_0/abc/{}">'
            '<button type="submit">Buy Now</button>'
            "</form>"
        )

        self.assertIn("Buy Now", parsed["available_actions"])
        self.assertEqual(parsed["action_urls"]["Buy Now"], "http://127.0.0.1:7001/done/fixed_0/abc/{}")


if __name__ == "__main__":
    unittest.main()
