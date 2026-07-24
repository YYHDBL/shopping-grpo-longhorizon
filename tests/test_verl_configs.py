"""veRL 配置必须始终引用唯一的 Shopping tool schema。"""

import unittest

from scripts.generate_verl_shop_configs import build_tool_config
from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS


class VerlConfigTest(unittest.TestCase):
    def test_tool_config_uses_every_canonical_schema_once(self):
        config = build_tool_config()
        schemas = [item["tool_schema"] for item in config["tools"]]
        self.assertEqual(schemas, SHOP_TOOL_SCHEMAS)
        self.assertTrue(all(item["class_name"].endswith("ShopSimulatorTool") for item in config["tools"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
