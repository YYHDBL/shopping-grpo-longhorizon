"""Conservative public-text checks. No task IDs, target options, or inferred facts.

These checks only cover named patterns; PASS does not certify all natural language.
"""

import re

from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE, comparison


def selected_checks(instruction, selected_options):
    text = str(instruction)
    selected = " | ".join(map(str, selected_options.values())).casefold()
    results = []

    def add(status, requirement, actual, rule):
        results.append(
            comparison(
                status,
                comparator="public-selected-v1:" + rule,
                required=requirement,
                actual=actual,
                source_field="instruction_text|selected_options",
                evidence={"instruction": text, "selected_options": dict(selected_options)},
            )
        )

    # Explicit numeric upper/lower bounds: only one selected value in the same unit.
    for m in re.finditer(
        r"(?:尺寸|大小)(?:要|需)?(小于|低于|大于|超过)(\d+(?:\.\d+)?)(寸|厘米|毫米)", text
    ):
        vals = re.findall(r"(\d+(?:\.\d+)?)" + re.escape(m[3]), selected)
        if len(vals) != 1:
            add(UNVERIFIABLE, m[0], vals, "size-bound")
            continue
        actual, bound = float(vals[0]), float(m[2])
        ok = actual < bound if m[1] in ("小于", "低于") else actual > bound
        add(PASS if ok else FAIL, m[0], actual, "size-bound")
    # Explicit mass only; a single selected mass, with equivalent public units.
    mass_units = {"kg": 1000, "千克": 1000, "公斤": 1000, "g": 1, "克": 1, "斤": 500}
    for m in re.finditer(
        r"规格(?:为|要)(\d+(?:\.\d+)?)\s*(kg|千克|公斤|g|克|斤)(?![a-z])", text, re.IGNORECASE
    ):
        values = []
        for found in re.finditer(
            r"(?<![\d.零一二三四五六七八九十百千万两-])(\d+(?:\.\d+)?|一|两|二|三|四|五|六|七|八|九|十)\s*(kg|千克|公斤|g|克|斤)(?![a-z])",
            selected,
            re.IGNORECASE,
        ):
            raw = found[1]
            number = (
                float(raw)
                if raw[0].isdigit()
                else {
                    "一": 1,
                    "两": 2,
                    "二": 2,
                    "三": 3,
                    "四": 4,
                    "五": 5,
                    "六": 6,
                    "七": 7,
                    "八": 8,
                    "九": 9,
                    "十": 10,
                }[raw]
            )
            values.append(number * mass_units[found[2].casefold()])
        # Do not assign a hidden tolerance to an approximate mass requirement.
        if re.match(r"左右|上下|大概", text[m.end() :]):
            add(UNVERIFIABLE, m[0], values, "mass-exact")
            continue
        target = float(m[1]) * mass_units[m[2].casefold()]
        status = UNVERIFIABLE if len(values) != 1 else PASS if values[0] == target else FAIL
        add(status, m[0], values, "mass-exact")
    # Explicit same-object size opposition, not a comparison to the hidden target.
    for m in re.finditer(r"(小型|大型)(低音炮)", text):
        opposite = "大" if m[1] == "小型" else "小"
        if re.search(opposite + r"(?:型)?" + m[2], selected):
            add(FAIL, m[0], selected, "component-size-opposition")
    # Selected variant overrides the listing-level feature claim.
    if re.search(r"自动休眠|闲置时.*低功耗", text) and re.search(
        r"不(?:支持)?休眠|无休眠", selected
    ):
        add(FAIL, "自动休眠/闲置低功耗", selected, "selected-negation")
    # A named color must be evidenced by the selected variant. Shades/codes stay
    # unknown: e.g. cream is not automatically equated to white or called wrong.
    for m in re.finditer(
        r"颜色(?:要|为|是)(?:经典的)?(军绿色|白色|黑色|黄色|蓝色|红色|桔色)", text
    ):
        color_values = [
            str(v).casefold()
            for k, v in selected_options.items()
            if "颜色" in k or k in ("color", "colour")
        ]
        actual = " | ".join(color_values)
        # Explicit base-color morpheme: 海军蓝 is blue; 黄爱心 is yellow.
        # Cream has no 白 evidence and stays unknown. Do not infer colors from codes.
        color_token = m[1].removesuffix("色")
        if color_token not in actual:
            add(UNVERIFIABLE, m[0], actual, "literal-color-evidence")
    # Listing is an assembly, selected item is a component. Do not infer that a
    # sensor includes the requested knob from the listing title/category alone.
    if (
        re.search(r"(?:一款|的)空调旋钮", text)
        and re.search(r"温度传感器", selected)
        and "旋钮" not in selected
    ):
        add(UNVERIFIABLE, "空调旋钮", selected, "component-not-assembly-proof")
    return results


def product_checks(product, instruction, selected_options):
    """Only explicit requirement patterns; missing proof never invents a failure."""
    from web_agent_site.engine.comparators import compare_core_functions

    text = str(instruction)
    selected = " | ".join(map(str, selected_options.values())).casefold()
    checks = []
    from web_agent_site.engine.public_option_evidence import selected_option_evidence

    images = selected_option_evidence(product, selected_options)
    image_colors = {c for image in images for c in image["facts"].get("visible_color_families", [])}
    if re.search(r"100\s*%\s*遮光", text):
        for image in images:
            actual = image["facts"].get("blackout_percent")
            if actual is not None and actual < 100:
                checks.append(
                    comparison(
                        FAIL,
                        comparator="selected-image-blackout-v1",
                        required="100%遮光",
                        actual=actual,
                        source_field="instruction_text|selected_option_image",
                        evidence=image,
                    )
                )
    if "深蓝色" in text and image_colors and "蓝" not in image_colors:
        checks.append(
            comparison(
                FAIL,
                comparator="selected-image-color-family-v1",
                required="深蓝色（至少应有蓝色）",
                actual=sorted(image_colors),
                source_field="instruction_text|selected_option_image",
                evidence=images,
            )
        )
    # Explicit single-protocol image text is evidence; a missing field is not.
    if re.search(r"(?<!不)(?<!无需)(?<!不需要)支持\s*sata\s*(?:读取|协议)", text, re.IGNORECASE):
        for image in images:
            exclusive = image["facts"].get("exclusive_storage_protocols")
            if exclusive and "sata" not in [str(p).casefold() for p in exclusive]:
                checks.append(
                    comparison(
                        FAIL,
                        comparator="selected-image-exclusive-protocol-v1",
                        required="支持SATA读取/协议",
                        actual=exclusive,
                        source_field="instruction_text|selected_option_image",
                        evidence=image,
                    )
                )

    def unknown(required, rule, actual=selected):
        checks.append(
            comparison(
                UNVERIFIABLE,
                comparator="public-evidence-v2:" + rule,
                required=required,
                actual=actual,
                source_field="instruction_text|public_product|selected_options",
                evidence={
                    "instruction": text,
                    "selected_options": dict(selected_options),
                    "reason": "No public proof for this requirement; absence is not negation.",
                },
            )
        )

    for pattern, claims in [
        (r"医用级别", ["医用级"]),
        (r"实时成像", ["实时成像", "可视化"]),
        (r"包装不容易破", ["包装耐破", "防破损包装", "耐撕裂包装"]),
        (r"头层牛皮", ["头层牛皮"]),
        (r"不用开孔安装", ["免开孔", "不用开孔"]),
        (r"高精度热敏元件", ["高精度热敏元件"]),
        (r"配备Type-C数据线", ["type-c数据线", "type-c线", "type-c双线"]),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if pattern == r"配备Type-C数据线" and any(
            "usb-c" in i["facts"].get("included_cable_connectors", []) for i in images
        ):
            continue
        enriched = dict(product)
        # Selected evidence is relevant, but do not use any unselected option.
        enriched["full_description"] = str(product.get("full_description", "")) + " " + selected
        results = [compare_core_functions([claim], enriched) for claim in claims]
        if any(r["status"] == PASS for r in results):
            continue
        if any((r.get("evidence") or {}).get("negated") for r in results):
            failed = next(r for r in results if (r.get("evidence") or {}).get("negated"))
            checks.append(failed)
        else:
            unknown(
                match[0],
                "explicit-claim",
                {
                    "title": product.get("title"),
                    "attributes": product.get("attribute"),
                    "selected": selected,
                },
            )
    # Specific scope mismatch: listing-level components do not prove the bundle.
    if "小吃推车" in text and selected.strip() == "汤锅":
        bundled = any(
            i["facts"].get("shipment_scope_explicit") is True
            and {"车架", "汤锅", "配件"} <= set(i["facts"].get("included_components", []))
            for i in images
        )
        if not bundled:
            unknown("小吃推车/车架及汤锅", "component-bundle")
    if "完整一套美牙仪" in text and not re.search(r"套装|套盒|美牙仪", selected):
        unknown("完整一套美牙仪", "component-bundle")
    if "深蓝色" in text:
        colors = " | ".join(str(v) for k, v in selected_options.items() if "颜色" in k)
        if not re.search(r"深蓝|藏蓝|海军蓝", colors) and not (
            image_colors and "蓝" not in image_colors
        ):
            unknown("深蓝色", "color-code", colors)
    if "蓝色和白色搭配" in text:
        colors = " | ".join(str(v) for k, v in selected_options.items() if "颜色" in k)
        if not (("蓝" in colors and "白" in colors) or {"蓝", "白"} <= image_colors):
            unknown("蓝白搭配", "two-colors", colors)
    # Do not convert grams to millilitres without density or explicit package evidence.
    if re.search(r"净含量\s*\d+(?:\.\d+)?\s*(?:毫升|ml)", text, re.IGNORECASE):
        content = " | ".join(str(v) for k, v in selected_options.items() if "净含量" in k)
        if re.search(r"\d\s*(?:g|克)(?![a-z])", content, re.IGNORECASE) and not re.search(
            r"毫升|ml", content, re.IGNORECASE
        ):
            unknown("要求体积单位，所选仅质量单位", "mass-volume-not-convertible", content)
    return checks
