"""Structured, answer-free ShopSimulator state for the canonical Agent renderer."""

from __future__ import annotations

import os

OBSERVATION_VERSION = "shopping-observation-v2"


def _compact_list(value, limit=10):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item) for item in value[:limit] if str(item).strip()]


def product_summary(product: dict, *, rank: int | None = None) -> dict:
    pricing = product.get("Price")
    if pricing is None:
        pricing = product.get("pricing")
    result = {
        "asin": str(product.get("asin", "")),
        "title": str(product.get("title") or product.get("Title") or ""),
        "brand": str(product.get("brand") or product.get("shop_name") or ""),
        "category": str(product.get("category") or ""),
        "price": pricing,
        "key_attributes": _compact_list(
            product.get("attribute") or product.get("Attributes") or []
        ),
    }
    if rank is not None:
        result["rank"] = int(rank)
    return result


def build_observation_state(
    *,
    page_type: str,
    session: dict,
    product_item_dict: dict,
    available_actions: dict,
) -> dict:
    """Build one public state; the task goal is intentionally not accepted."""
    clickables = [str(value) for value in available_actions.get("clickables", [])]
    state = {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "search_available": bool(available_actions.get("has_search_bar")),
        "actions": clickables,
    }
    if page_type == "search_results":
        all_asins = session.get("search_result_asins") or []
        page_asins = session.get("current_page_asins") or []
        rank_by_asin = {
            asin: index for index, asin in enumerate(all_asins, start=1)
        }
        state.update(
            {
                "query": " ".join(session.get("keywords") or []),
                "normalized_query": session.get("normalized_query") or "",
                "page": int(session.get("page") or 1),
                "total_pages": int(session.get("total_pages") or 1),
                "total_results": int(session.get("total_results") or 0),
                "rank_start": min(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "rank_end": max(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "products": [
                    product_summary(
                        product_item_dict[asin],
                        rank=rank_by_asin[asin],
                    )
                    for asin in page_asins
                    if asin in product_item_dict and asin in rank_by_asin
                ],
            }
        )
    elif page_type in {"product_detail", "information_subpage"}:
        asin = session.get("asin")
        product = product_item_dict.get(asin, {})
        state["product"] = product_summary(product)
        state["selected_options"] = dict(session.get("options") or {})
        state["available_options"] = {
            str(key): _compact_list(values, limit=100)
            for key, values in (product.get("options") or {}).items()
        }
        if session.get("selected_price") is not None:
            state["selected_price"] = session["selected_price"]
        if os.environ.get("SHOPSIM_PUBLIC_SUPPORT") == "1" and page_type == "product_detail":
            # Public catalog fields only. Never inspect goal/instructions/reward.
            selected = state["selected_options"]
            state["public_details"] = {
                "description": product.get("Description") or "",
                "features": product.get("BulletPoints") or [],
                "unselected_axes": [k for k,v in state["available_options"].items() if v and k not in selected],
                "option_prices": {v: p for v,p in (product.get("option_to_price") or {}).items() if p is not None} if sum(len(v) for v in state["available_options"].values()) <= 20 else {},
                "price_note": "单项报价只作比较，多轴最终组合以选择后的price为准；空白描述不代表功能满足。",
            }
            from web_agent_site.engine.public_option_evidence import option_evidence
            evidence=option_evidence(product)
            if evidence: state['public_details']['option_evidence']=evidence
        if page_type == "information_subpage":
            subpage = str(session.get("subpage") or "information")
            field = {
                "description": "Description",
                "features": "BulletPoints",
                "reviews": "Reviews",
                "attributes": "Attributes",
            }.get(subpage.casefold())
            state["subpage"] = subpage
            state["content"] = product.get(field, "") if field else ""
    return state


def page_type_from_name(page_name: str) -> str:
    return {
        "": "search_home",
        "search_results": "search_results",
        "item_page": "product_detail",
        "item_sub_page": "information_subpage",
        "done": "terminal",
    }.get(page_name, "unknown")
