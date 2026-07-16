import inspect
import unittest

from shopping_grpo import shop_http_env


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class ShopAgentEnvTest(unittest.TestCase):
    def test_structured_api_environment_is_exposed(self):
        self.assertTrue(hasattr(shop_http_env, "ShopAgentEnv"))

    def test_structured_api_environment_exposes_lifecycle_methods(self):
        for name in ("reset", "step", "release"):
            self.assertTrue(hasattr(shop_http_env.ShopAgentEnv, name))

    def test_transport_can_be_injected_for_api_tests(self):
        parameters = inspect.signature(shop_http_env.ShopAgentEnv).parameters
        self.assertIn("transport", parameters)

    def test_reset_step_and_release_use_structured_api(self):
        transport = FakeTransport(
            [
                {"result": {"instruction": "找乳胶枕", "env_idx": 7, "idx": 0}},
                {"result": {"instruction": "搜索结果", "done": False, "reward": 0.0}},
                {
                    "result": {
                        "instruction": "购买完成",
                        "done": True,
                        "reward": 0.0,
                        "reward_detail": {"r_type": 0, "r_att": 0, "r_option": 0, "r_price": 0},
                        "purchase": {"asin": "wrong"},
                        "goal": {"instruction_text": "hidden"},
                    }
                },
                {"result": {"message": "Environment 7 is already free"}},
            ]
        )
        env = shop_http_env.ShopAgentEnv("http://shop.test", timeout=12, transport=transport)

        reset = env.reset(0)
        search = env.step("search[乳胶枕]")
        terminal = env.step("click[Buy Now]")
        env.release()

        self.assertEqual(reset["instruction"], "找乳胶枕")
        self.assertFalse(search["done"])
        self.assertTrue(terminal["done"])
        self.assertEqual(terminal["reward"], 0.0)
        self.assertEqual(terminal["purchase"], {"asin": "wrong"})
        self.assertEqual(terminal["goal"], {"instruction_text": "hidden"})
        self.assertIsNone(env.env_idx)
        self.assertEqual(
            [payload for _, payload, _ in transport.calls],
            [
                {"action": "reset", "idx": 0},
                {"action": "interact", "env_idx": 7, "response": "search[乳胶枕]"},
                {"action": "interact", "env_idx": 7, "response": "click[Buy Now]"},
                {"action": "release_one", "env_idx": 7},
            ],
        )

    def test_zero_reward_terminal_purchase_stops_further_steps(self):
        transport = FakeTransport(
            [
                {"result": {"instruction": "找乳胶枕", "env_idx": 2, "idx": 0}},
                {
                    "result": {
                        "instruction": "购买完成",
                        "done": True,
                        "reward": 0.0,
                        "reward_detail": {"r_type": 0, "r_att": 0, "r_option": 0, "r_price": 0},
                        "purchase": {"asin": "wrong"},
                        "goal": {"instruction_text": "hidden"},
                    }
                },
            ]
        )
        env = shop_http_env.ShopAgentEnv(transport=transport)

        terminal = env.reset(0)
        terminal = env.step("click[Buy Now]")

        self.assertTrue(terminal["done"])
        self.assertEqual(terminal["reward"], 0.0)
        with self.assertRaises(shop_http_env.ShopEnvironmentStateError):
            env.step("search[乳胶枕]")

    def test_environment_error_is_distinct_from_http_error(self):
        env = shop_http_env.ShopAgentEnv(
            transport=FakeTransport([{"result": {"error": "reset action requires idx parameter"}}])
        )

        with self.assertRaises(shop_http_env.ShopEnvironmentError):
            env.reset(0)

    def test_context_manager_releases_after_http_error(self):
        transport = FakeTransport(
            [
                {"result": {"instruction": "找乳胶枕", "env_idx": 3, "idx": 0}},
                OSError("connection reset"),
                {"result": {"message": "Environment 3 has been released"}},
            ]
        )

        with self.assertRaises(shop_http_env.ShopHttpError):
            with shop_http_env.ShopAgentEnv(transport=transport) as env:
                env.reset(0)
                env.step("search[乳胶枕]")

        self.assertEqual(
            [payload["action"] for _, payload, _ in transport.calls],
            ["reset", "interact", "release_one"],
        )
        self.assertIsNone(env.env_idx)

    def test_failed_release_keeps_lease_for_recovery(self):
        """释放请求未送达时，客户端必须保留租约编号供上层恢复。"""
        transport = FakeTransport(
            [
                {"result": {"instruction": "找乳胶枕", "env_idx": 3, "idx": 0}},
                OSError("connection reset"),
            ]
        )
        env = shop_http_env.ShopAgentEnv(transport=transport)
        env.reset(0)

        with self.assertRaises(shop_http_env.ShopHttpError):
            env.release()

        self.assertEqual(env.env_idx, 3)


if __name__ == "__main__":
    unittest.main()
