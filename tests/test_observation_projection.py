"""Observation projection preserves actionable state and bounds visible tokens."""

import json
import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.environment.projection import (
    ObservationProjectionError,
    TRUNCATION_MARKER,
    project_observation,
)
from shopping_grpo.environment.observation import render_structured_observation


def search_page(product_count=12, page=1):
    buttons = ["back to search", "next >"]
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        "query: useful product",
        "normalized_query: useful product",
        (
            f"Page {page} of 2 (Total results: 40; "
            f"ranks {(page - 1) * 20 + 1}-{(page - 1) * 20 + product_count})"
        ),
        "格式: rank|asin|price|brand|category|key_attributes|title",
    ]
    for index in range(product_count):
        asin = f"{100000000000 + index}"
        lines.append(
            f"{index + 1}|{asin}|{index + 1}.0|brand|category|attr|"
            f"product title {index} " + "x" * 40
        )
        buttons.append(asin)
    lines.append(f"products_shown: {product_count}")
    return (
        "\n".join(lines)
        + "\n\n搜索功能是否可用: False"
        + "\n\n可点击的按钮: "
        + json.dumps(buttons, ensure_ascii=False)
    )


class ObservationProjectionTest(unittest.TestCase):
    def test_search_projection_preserves_every_current_page_product(self):
        raw = search_page(product_count=20)
        visible, meta = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )

        self.assertLessEqual(len(visible), 1200)
        self.assertTrue(meta.truncated)
        self.assertTrue(meta.critical_footer_preserved)
        self.assertIn(TRUNCATION_MARKER, visible)
        self.assertEqual(
            {button.casefold() for button in clickable_buttons(visible) if not button.isdigit()},
            {"back to search", "next >"},
        )
        self.assertEqual(
            set(product_ids(visible)),
            {button for button in clickable_buttons(visible) if button.isdigit()},
        )
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_guard_accepts_last_product_instead_of_creating_blind_spot(self):
        raw = search_page(product_count=20)
        visible, _ = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )
        last_asin = product_ids(raw)[-1]

        self.assertIsNone(
            action_reject_reason("open_product", {"asin": last_asin}, visible)
        )

    def test_second_environment_page_preserves_products_21_through_40(self):
        raw = search_page(product_count=20, page=2)
        visible, _ = project_observation(
            "next_page",
            raw,
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )

        self.assertIn("Page 2 of 2", visible)
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_capacity_mismatch_fails_instead_of_silently_dropping_products(self):
        raw = search_page(product_count=20)
        with self.assertRaisesRegex(ObservationProjectionError, "page capacity"):
            project_observation(
                "search_products",
                raw,
                count_tokens=len,
                token_budget=1200,
                search_top_k=10,
            )

    def test_short_product_page_is_identity_projection(self):
        raw = (
            "Product [SEP] price: 20 [SEP] Buy Now"
            "\n\n搜索功能是否可用: False"
            '\n\n可点击的按钮: ["back to search", "buy now"]'
        )
        visible, meta = project_observation(
            "open_product",
            raw,
            count_tokens=len,
            token_budget=448,
        )

        self.assertEqual(visible, raw)
        self.assertFalse(meta.truncated)

    def test_generic_projection_keeps_complete_footer(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=300,
        )

        self.assertLessEqual(len(visible), 300)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertIn("TAIL_SPECIFICATION", visible)
        self.assertTrue(meta.critical_footer_preserved)

    def test_long_page_without_action_footer_fails_closed(self):
        with self.assertRaisesRegex(ObservationProjectionError, "action footer"):
            project_observation(
                "unknown",
                "x" * 1000,
                count_tokens=len,
                generic_token_budget=128,
            )

    def test_structured_search_projection_preserves_all_twenty_products(self):
        products = [
            {
                "rank": index,
                "asin": f"{index:012d}",
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index in range(1, 21)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": [
                    "back to search",
                    "next >",
                    *[product["asin"] for product in products],
                ],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 2,
                "total_results": 40,
                "rank_start": 1,
                "rank_end": 20,
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )
        self.assertEqual(product_ids(visible), product_ids(raw))
        self.assertLessEqual(len(visible), 1400)

    def test_structured_projection_preserves_mixed_catalog_id_lengths(self):
        asins = ["12345678", "123456789", "1234567890", "35842622441", "123456789012"]
        products = [
            {
                "rank": index,
                "asin": asin,
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index, asin in enumerate(asins, start=1)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", *asins],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 1,
                "total_results": len(products),
                "rank_start": 1,
                "rank_end": len(products),
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=700,
            search_top_k=20,
        )

        self.assertEqual(product_ids(visible), asins)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
