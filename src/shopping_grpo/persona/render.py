# 画像渲染：把原始 user_persona dict 渲染成给模型看的文本
# 泄漏控制：「最近14天搜索关键词」由商品—指令反向生成，天然回声目标查询，
# 渲染时无条件剔除；收藏/加购在当前数据全为空，同样剔除行为清单字段。
from __future__ import annotations

from typing import Mapping

PERSONA_RENDER_VERSION = "persona-render-v1"
DROPPED_BEHAVIOR_FIELDS = (
    "最近14天搜索关键词",
    "最近14天收藏商品",
    "最近14天加购商品",
    "最近14天关注店铺",
)


def _clean_persona(persona: Mapping) -> dict:
    cleaned = dict(persona)
    behavior = dict(cleaned.get("行为特征") or {})
    for field in DROPPED_BEHAVIOR_FIELDS:
        behavior.pop(field, None)
    cleaned["行为特征"] = behavior
    return cleaned


def render_persona(persona: Mapping) -> str:
    """渲染画像文本；输入须为 user_persona dict，空画像返回空串。"""
    if not isinstance(persona, Mapping) or not persona:
        return ""
    cleaned = _clean_persona(persona)
    lines = ["[用户画像]"]
    region = cleaned.get("地区信息") or {}
    if region:
        lines.append(
            f"地区：{region.get('省份', '')}{region.get('城市', '')}"
        )
    demographic = cleaned.get("人口属性") or {}
    if demographic:
        parts = [
            str(demographic.get(key))
            for key in ("性别", "年龄段", "消费等级", "会员等级")
            if demographic.get(key)
        ]
        if parts:
            lines.append("人口属性：" + "，".join(part.strip(" ,") for part in parts))
    behavior = cleaned.get("行为特征") or {}
    behavior_parts = []
    if behavior.get("日均浏览时长"):
        behavior_parts.append(f"日均浏览 {behavior['日均浏览时长']} 秒")
    if behavior.get("近7天访问次数"):
        behavior_parts.append(f"近7天访问 {behavior['近7天访问次数']} 次")
    if behavior.get("常用设备"):
        behavior_parts.append(f"常用设备 {behavior['常用设备']}")
    if behavior_parts:
        lines.append("行为特征：" + "，".join(behavior_parts))
    trading = cleaned.get("交易特征") or {}
    if trading:
        parts = []
        if trading.get("近30天消费金额"):
            parts.append(f"近30天消费 {trading['近30天消费金额']} 元")
        if trading.get("平均客单价"):
            parts.append(f"平均客单价 {trading['平均客单价']} 元")
        if trading.get("支付方式偏好"):
            parts.append(f"偏好 {trading['支付方式偏好']}")
        if parts:
            lines.append("交易特征：" + "，".join(parts))
    interest = cleaned.get("兴趣偏好") or {}
    if interest:
        category = interest.get("类目偏好")
        if isinstance(category, Mapping) and category:
            high = [name for name, level in category.items() if str(level) == "高"]
            if high:
                lines.append("类目偏好（高）：" + "、".join(high))
        brands = interest.get("品牌偏好")
        if isinstance(brands, list) and brands:
            names = [
                item.get("品牌名称")
                for item in brands
                if isinstance(item, Mapping) and item.get("品牌名称")
            ]
            if names:
                lines.append("品牌偏好：" + "、".join(names))
        attributes = interest.get("商品属性偏好") or {}
        if isinstance(attributes, Mapping):
            for key, label in (("风格", "风格偏好"), ("颜色", "颜色偏好"), ("材质", "材质偏好")):
                values = attributes.get(key)
                if isinstance(values, list) and values:
                    lines.append(f"{label}：" + "、".join(str(v) for v in values))
            price_band = attributes.get("价格区间")
            if isinstance(price_band, Mapping):
                lines.append(
                    f"常购价格带：{price_band.get('最小值', '?')}-{price_band.get('最大值', '?')} 元"
                )
    return "\n".join(lines)
