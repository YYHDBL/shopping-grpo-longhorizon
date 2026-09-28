# 裁决者盲判校准考卷：DeepSeek V4.1 Flash 充当人工裁决角色
# 盲判：只给需求与商品，不给 Jev 判定，独立四分类；判后与 Jev 对比出一致率
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.rollout import OpenAIChatClient  # noqa: E402

OUT = ROOT / "outputs/calibration"

PROMPT = """你是购物商品满足度裁判。根据用户需求和商品信息判断商品对需求的满足程度，四选一：
- fully：商品满足用户需求中的全部明确约束，包括品类、属性、规格和预算
- partial：商品满足部分主要需求，但至少一项明确约束不满足或与需求不符
- does_not：商品与用户需求的核心品类或主要约束明显不符
- unknown：给出的信息不足以判断

判定要点：
- 只依据给出的信息判断，不推测未给出的细节
- 用户明确点名的约束（品类、产地、材质、功能、图案、预算、规格等）都算明确约束，缺一个不能给 fully
- 价格口径："X 元以内/以下"= <=X；"X 元左右/上下"允许约 ±10% 浮动；"X-Y 元"按闭区间
- 同一约束的同义表述视为满足；模糊主观偏好（如"摸起来更软"）不作扣分依据

只输出一行，格式：判定|不超过 30 字的理由"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def main():
    load_env_file(ROOT / ".env")
    questions = json.loads((OUT / "exam_50.json").read_text(encoding="utf-8"))
    client = OpenAIChatClient(
        model=os.environ["TEACHER2_MODEL"],
        base_url=os.environ["TEACHER2_BASE_URL"],
        api_key=os.environ["TEACHER2_API_KEY"],
        temperature=0.0,
        max_tokens=200,
        timeout=120,
        thinking=False,
    )
    client.extra_headers = {"x-opencode-session": str(uuid.uuid4())}

    results = []
    for q in questions:
        content = (f"用户需求：{q['instruction']}\n\n商品信息：\n{q['product']}")
        message = client.complete(
            [{"role": "user", "content": PROMPT + "\n\n" + content}], []
        )
        text = message.get("content") or ""
        match = re.match(r"\s*(fully|partial|does_not|unknown)\s*\|(.*)", text)
        if match:
            choice, reason = match.group(1), match.group(2).strip()
        else:
            choice, reason = "parse_error", text[:60]
        full_name = {
            "fully": "fully_satisfies",
            "partial": "partially_satisfies",
            "does_not": "does_not_satisfy",
        }.get(choice)
        results.append({
            "number": q["number"],
            "pool": q["pool"],
            "jev": q["jev"],
            "judge": choice,
            "judge_full": full_name or choice,
            "agree": full_name == q["jev"],
            "reason": reason,
        })
        print(f"{q['number']:2d}. jev={q['jev']:<18s} judge={choice:<14s}"
              f" {'一致' if full_name == q['jev'] else '分歧'}  {reason}", flush=True)

    (OUT / "judge_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    agree = sum(1 for r in results if r["agree"])
    print(f"\n整体一致率：{agree}/{len(results)} = {agree/len(results):.0%}")
    jev_fully = [r for r in results if r["jev"] == "fully_satisfies"]
    fully_ok = sum(1 for r in jev_fully if r["judge"] == "fully")
    print(f"Jev 满分精确率：{fully_ok}/{len(jev_fully)}"
          f" = {fully_ok/max(len(jev_fully),1):.0%}")
    print("\n分歧清单：")
    for r in results:
        if not r["agree"]:
            print(f"  第{r['number']}题 [{r['pool']}] jev={r['jev']} "
                  f"judge={r['judge_full']}：{r['reason']}")


if __name__ == "__main__":
    main()
