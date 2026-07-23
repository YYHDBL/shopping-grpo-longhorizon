"""veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class VerlCompatTest(unittest.TestCase):
    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
            },
        ):
            from shopping_grpo.verl_compat import install_torch_padding_fallback

            install_torch_padding_fallback()

        self.assertEqual(attention._get_attention_functions(), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
