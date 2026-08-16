#!/usr/bin/env python3
"""Build a self-contained HTML report from one Shopping GRPO evaluation run."""

from __future__ import annotations

import collections
import json
import statistics
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TOOL_LABELS = {
    "search_products": "搜索商品",
    "open_product": "打开商品",
    "select_option": "选择规格",
    "view_description": "查看描述",
    "view_features": "查看特点",
    "view_reviews": "查看评价",
    "prev_page": "返回上一页",
    "back_to_search": "返回搜索",
    "next_page": "下一页",
    "buy_now": "购买",
    "finish_without_purchase": "主动结束",
}
REWARD_LABELS = {
    "gold_purchase": "严格成功购买",
    "partial_alternative_purchase": "部分满足购买",
    "wrong_purchase": "错误购买",
    "repeat_loop": "重复循环",
    "max_steps": "达到步数上限",
    "early_abstain": "过早放弃",
    "unknown": "未完成 / 无终局",
}


def _load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percent(value, total):
    return round(100 * value / total, 2) if total else 0.0


def build_data(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trajectories = _load_jsonl(run_dir / "trajectories.jsonl")
    protocol = summary.get("protocol") or {}
    model = str(protocol.get("model") or run_dir.name or "unknown-model")
    strict_ids = {int(task_id) for task_id in summary["strict_success_task_ids"]}
    rows = []
    tool_counts = collections.Counter()
    tool_task_counts = collections.Counter()
    guard_counts = collections.Counter()
    step_values = []

    for trajectory in trajectories:
        task_id = int(trajectory["task_id"])
        terminal = trajectory.get("terminal_result") or {}
        reward_detail = terminal.get("reward_detail") or {}
        reward_type = reward_detail.get("reward_type") or "unknown"
        tools = [step.get("tool_name", "unknown") for step in trajectory.get("steps", [])]
        guards = [item.get("reason", "unknown") for item in trajectory.get("blocked_tool_calls", [])]
        tool_counts.update(tools)
        tool_task_counts.update(set(tools))
        guard_counts.update(guards)
        step_count = len(trajectory.get("steps", []))
        step_values.append(step_count)
        query = (trajectory.get("initial_result") or {}).get("instruction", "")
        rows.append(
            {
                "task_id": task_id,
                "query": query,
                "status": trajectory.get("status", "unknown"),
                "done": bool(trajectory.get("done")),
                "strict": task_id in strict_ids,
                "purchase_success": bool(reward_detail.get("purchase_success")),
                "reward": round(float(trajectory.get("final_reward", 0.0)), 4),
                "reward_type": reward_type,
                "termination": terminal.get("termination_reason") or "",
                "steps": step_count,
                "tools": tools,
                "tool_sequence": " → ".join(TOOL_LABELS.get(tool, tool) for tool in tools),
                "guard_count": len(guards),
                "guard_reasons": guards,
                "error": (trajectory.get("error") or {}).get("type", ""),
            }
        )

    rows.sort(key=lambda row: row["task_id"])
    total = len(rows)
    reward_counts = collections.Counter(row["reward_type"] for row in rows)
    status_counts = collections.Counter(row["status"] for row in rows)
    step_buckets = []
    max_steps = int(protocol.get("max_steps") or max(step_values or [35]))
    for label, low, high in (("1–5 步", 1, 5), ("6–10 步", 6, 10), ("11–20 步", 11, 20), (f"21–{max_steps} 步", 21, max_steps)):
        bucket = [row for row in rows if low <= row["steps"] <= high]
        step_buckets.append(
            {
                "label": label,
                "tasks": len(bucket),
                "strict": sum(row["strict"] for row in bucket),
                "strict_rate": _percent(sum(row["strict"] for row in bucket), len(bucket)),
                "mean_reward": round(statistics.mean(row["reward"] for row in bucket), 4) if bucket else 0.0,
            }
        )

    sorted_steps = sorted(step_values)
    reward_values = [row["reward"] for row in rows]
    charts = {
        "outcomes": [
            {"key": "gold_purchase", "label": "严格成功购买", "value": reward_counts["gold_purchase"]},
            {"key": "partial_alternative_purchase", "label": "部分满足购买", "value": reward_counts["partial_alternative_purchase"]},
            {"key": "wrong_purchase", "label": "错误购买", "value": reward_counts["wrong_purchase"]},
            {"key": "repeat_loop", "label": "重复循环", "value": reward_counts["repeat_loop"]},
            {"key": "max_steps", "label": "达到步数上限", "value": reward_counts["max_steps"]},
            {"key": "early_abstain", "label": "过早放弃", "value": reward_counts["early_abstain"]},
            {"key": "unknown", "label": "未完成 / 无终局", "value": reward_counts["unknown"]},
        ],
        "tools": [
            {"key": key, "label": TOOL_LABELS.get(key, key), "value": value, "tasks": tool_task_counts[key]}
            for key, value in tool_counts.most_common()
        ],
        "guards": [
            {"key": key, "label": key, "value": value}
            for key, value in guard_counts.most_common()
        ],
        "steps": step_buckets,
    }
    compact_summary = {
        "total": total,
        "completed": summary["completed_tasks"],
        "done": summary["done_tasks"],
        "done_rate": summary["done_rate"],
        "strict": summary["strict_successes"],
        "strict_rate": summary["strict_success_rate"],
        "purchase": summary["purchase_successes"],
        "purchase_rate": summary["purchase_success_rate"],
        "mean_reward": summary["mean_final_reward"],
        "weighted_score": summary["mean_weighted_score"],
        "average_steps": summary["average_steps"],
        "median_steps": statistics.median(step_values),
        "min_steps": min(step_values),
        "max_steps": max(step_values),
        "reward_min": min(reward_values),
        "reward_max": max(reward_values),
        "reward_median": statistics.median(reward_values),
        "guard_total": sum(guard_counts.values()),
        "guard_rate": _percent(sum(guard_counts.values()), total),
        "guard_per_task": round(sum(guard_counts.values()) / total, 4) if total else 0.0,
        "status_counts": dict(status_counts),
        "reward_counts": dict(reward_counts),
        "guard_counts": dict(guard_counts),
        "tool_counts": dict(tool_counts),
    }
    return {
        "meta": {
            "model": model,
            "benchmark": protocol.get("benchmark", ""),
            "max_steps": max_steps,
            "max_tokens": protocol.get("max_tokens", ""),
            "temperature": protocol.get("temperature", ""),
            "top_p": protocol.get("top_p", ""),
            "reward_contract": protocol.get("reward_contract", summary.get("reward_contract", "")),
        },
        "summary": compact_summary,
        "charts": charts,
        "rows": rows,
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>评测报告</title>
  <style>
    :root { --bg:#f5f7fb; --ink:#172033; --muted:#68748a; --line:#e4e8f0; --card:#fff; --blue:#4969e8; --green:#16a36a; --amber:#d99116; --red:#d94d5c; --purple:#855bd5; --cyan:#2aa7b8; --shadow:0 12px 30px #26375712; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:linear-gradient(135deg,#f8faff 0%,var(--bg) 55%,#f5f8ff 100%); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
    .page { max-width:1400px; margin:0 auto; padding:32px 24px 56px; }
    header { display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin-bottom:24px; }
    h1 { margin:0 0 5px; font-size:31px; letter-spacing:-.03em; }
    h2 { margin:0 0 16px; font-size:18px; }
    h3 { margin:0 0 10px; font-size:15px; }
    .subtitle,.muted { color:var(--muted); }
    .tag { display:inline-block; padding:5px 10px; border-radius:99px; background:#e9eeff; color:#3d56c3; font-weight:650; font-size:12px; }
    .grid { display:grid; gap:16px; }
    .kpis { grid-template-columns:repeat(6,minmax(0,1fr)); margin-bottom:16px; }
    .two { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px; box-shadow:var(--shadow); }
    .kpi .value { font-size:28px; font-weight:760; letter-spacing:-.04em; }
    .kpi .label { color:var(--muted); margin-top:4px; font-size:12px; }
    .kpi.green .value { color:var(--green); } .kpi.blue .value { color:var(--blue); } .kpi.amber .value { color:var(--amber); }
    .kpi.red .value { color:var(--red); } .kpi.purple .value { color:var(--purple); } .kpi.cyan .value { color:var(--cyan); }
    .bar-row { display:grid; grid-template-columns:150px 1fr 55px; gap:10px; align-items:center; margin:11px 0; }
    .bar-label { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .bar-track { height:10px; background:#edf0f6; border-radius:10px; overflow:hidden; }
    .bar { height:100%; border-radius:10px; background:var(--blue); min-width:2px; }
    .bar.green { background:var(--green); } .bar.amber { background:var(--amber); } .bar.red { background:var(--red); }
    .bar.purple { background:var(--purple); } .bar.cyan { background:var(--cyan); }
    .bar-value { text-align:right; font-weight:650; }
    .insight { border-left:4px solid var(--blue); background:#f8faff; padding:12px 14px; border-radius:10px; margin:10px 0; }
    .insight strong { color:#2e4ac1; }
    .stat-line { display:flex; justify-content:space-between; gap:16px; padding:9px 0; border-bottom:1px solid var(--line); }
    .stat-line:last-child { border-bottom:0; }
    .table-wrap { overflow:auto; }
    table { border-collapse:collapse; width:100%; min-width:760px; }
    th,td { padding:10px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:12px; white-space:nowrap; background:#fbfcfe; position:sticky; top:0; z-index:1; }
    th button { border:0; background:transparent; color:inherit; font:inherit; cursor:pointer; padding:0; }
    td.num, th.num { text-align:right; }
    .pill { display:inline-block; padding:3px 8px; border-radius:99px; font-size:12px; white-space:nowrap; background:#eef1f6; color:#526078; }
    .pill.ok { color:#08794e; background:#e3f7ee; } .pill.warn { color:#97600b; background:#fff2d6; }
    .pill.bad { color:#a3293c; background:#ffe7ea; } .pill.neutral { color:#626d81; background:#eef0f5; }
    .controls { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px; }
    input,select { border:1px solid var(--line); border-radius:9px; background:#fff; color:var(--ink); padding:9px 11px; font:inherit; }
    input { min-width:270px; flex:1; }
    .small { font-size:12px; color:var(--muted); }
    .query { max-width:420px; min-width:260px; }
    footer { color:var(--muted); text-align:center; padding-top:28px; font-size:12px; }
    @media (max-width:1100px) { .kpis { grid-template-columns:repeat(3,minmax(0,1fr)); } }
    @media (max-width:760px) { .page { padding:22px 13px 40px; } header { display:block; } h1 { font-size:25px; } .kpis,.two { grid-template-columns:1fr; } .bar-row { grid-template-columns:125px 1fr 48px; } .card { padding:16px; } }
  </style>
</head>
<body>
<main class="page">
  <header>
    <div><div class="tag" id="report-tag"></div><h1 id="report-title"></h1><div class="subtitle" id="report-subtitle"></div></div>
    <div class="small" id="report-meta"></div>
  </header>

  <section class="grid kpis">
    <div class="card kpi green"><div class="value" id="kpi-strict"></div><div class="label">严格成功率</div></div>
    <div class="card kpi blue"><div class="value" id="kpi-purchase"></div><div class="label">购买成功率</div></div>
    <div class="card kpi cyan"><div class="value" id="kpi-done"></div><div class="label">完成终局率</div></div>
    <div class="card kpi purple"><div class="value" id="kpi-reward"></div><div class="label">平均最终 Reward</div></div>
    <div class="card kpi amber"><div class="value" id="kpi-steps"></div><div class="label">平均步数</div></div>
    <div class="card kpi red"><div class="value" id="kpi-guard"></div><div class="label">Guard 拒绝次数</div></div>
  </section>

  <section class="grid two">
    <div class="card"><h2>最终结果构成</h2><div id="outcome-chart"></div><div class="small">严格成功以 Reward v3 的 gold_purchase 计；未完成表示模型停止时没有终局 Reward。</div></div>
    <div class="card"><h2>步数区间与严格成功</h2><div id="step-chart"></div><div class="small">每个柱同时显示该区间任务量和严格成功率。</div></div>
  </section>

  <section class="grid two" style="margin-top:16px">
    <div class="card"><h2>工具调用分布</h2><div id="tool-chart"></div></div>
    <div class="card"><h2>主要问题定位</h2><div id="insights"></div><div id="guard-chart"></div></div>
  </section>

  <section class="grid two" style="margin-top:16px">
    <div class="card"><h2>关键统计</h2><div id="stats"></div></div>
    <div class="card"><h2>协议与数据完整性</h2><div id="protocol"></div></div>
  </section>

  <section class="card" style="margin-top:16px">
    <h2>任务明细</h2>
    <div class="controls"><input id="query-filter" placeholder="搜索 task_id 或用户需求"><select id="outcome-filter"><option value="all">全部结果</option><option value="gold_purchase">严格成功购买</option><option value="partial_alternative_purchase">部分满足购买</option><option value="wrong_purchase">错误购买</option><option value="repeat_loop">重复循环</option><option value="max_steps">达到步数上限</option><option value="early_abstain">过早放弃</option><option value="unknown">未完成 / 无终局</option></select><span class="small" id="row-count"></span></div>
    <div class="table-wrap"><table><thead><tr><th><button data-sort="task_id">Task ID ↕</button></th><th>用户需求</th><th><button data-sort="reward_type">结果 ↕</button></th><th class="num"><button data-sort="steps">步数 ↕</button></th><th class="num"><button data-sort="reward">Reward ↕</button></th><th class="num"><button data-sort="guard_count">Guard ↕</button></th><th>动作序列</th></tr></thead><tbody id="task-table"></tbody></table></div>
  </section>
  <footer id="report-footer"></footer>
</main>
<script>
const REPORT_DATA = __REPORT_DATA__;
const S = REPORT_DATA.summary;
const M = REPORT_DATA.meta;
const fmtPct = v => `${(Number(v) * 100).toFixed(1)}%`;
const fmt = v => Number(v).toFixed(3);
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[char]);
const show = value => value === '' || value === null || value === undefined ? '—' : value;
const outcomeLabel = key => ({gold_purchase:'严格成功购买',partial_alternative_purchase:'部分满足购买',wrong_purchase:'错误购买',repeat_loop:'重复循环',max_steps:'达到步数上限',early_abstain:'过早放弃',unknown:'未完成 / 无终局'})[key] || key;
const outcomeClass = key => key === 'gold_purchase' ? 'ok' : ['wrong_purchase','repeat_loop','max_steps','early_abstain'].includes(key) ? 'bad' : key === 'partial_alternative_purchase' ? 'warn' : 'neutral';
const barClass = key => ({gold_purchase:'green',partial_alternative_purchase:'amber',wrong_purchase:'red',repeat_loop:'red',max_steps:'purple',early_abstain:'red',unknown:'cyan'})[key] || '';
function setText(id, value) { document.getElementById(id).textContent = value; }
setText('report-tag', `Shopping GRPO · ${M.reward_contract || '评测报告'}`);
setText('report-title', `${M.model} 评测报告`);
setText('report-subtitle', `最大 ${show(M.max_steps)} 步 · 每回合最多 ${show(M.max_tokens)} tokens · 温度 ${show(M.temperature)} · top-p ${show(M.top_p)}`);
document.getElementById('report-meta').innerHTML = `任务数：${S.total}<br>基准：${esc(M.benchmark || '未记录')}`;
setText('report-footer', `报告基于 ${S.total} 条 ${M.model} 实际评测轨迹生成；页面使用任务级摘要，不修改原始轨迹。`);
document.title = `${M.model} 评测报告`;
setText('kpi-strict', `${S.strict}/${S.total} · ${fmtPct(S.strict_rate)}`);
setText('kpi-purchase', `${S.purchase}/${S.total} · ${fmtPct(S.purchase_rate)}`);
setText('kpi-done', `${S.done}/${S.total} · ${fmtPct(S.done_rate)}`);
setText('kpi-reward', fmt(S.mean_reward));
setText('kpi-steps', `${S.average_steps.toFixed(2)}（中位数 ${S.median_steps}）`);
setText('kpi-guard', `${S.guard_total}（${S.guard_per_task.toFixed(2)}/题）`);

function renderBars(targetId, items, maxValue, valueText, classFn) {
  const target = document.getElementById(targetId);
  target.innerHTML = items.map(item => `<div class="bar-row"><div class="bar-label" title="${item.label}">${item.label}</div><div class="bar-track"><div class="bar ${classFn ? classFn(item) : ''}" style="width:${Math.max(2, item.value / maxValue * 100)}%"></div></div><div class="bar-value">${valueText(item)}</div></div>`).join('');
}
renderBars('outcome-chart', REPORT_DATA.charts.outcomes, Math.max(...REPORT_DATA.charts.outcomes.map(x=>x.value)), item => `${item.value}（${fmtPct(item.value/S.total)}）`, item => barClass(item.key));
renderBars('tool-chart', REPORT_DATA.charts.tools, Math.max(...REPORT_DATA.charts.tools.map(x=>x.value)), item => `${item.value}`, item => '');
renderBars('guard-chart', REPORT_DATA.charts.guards, Math.max(...REPORT_DATA.charts.guards.map(x=>x.value)), item => `${item.value}`, item => 'red');

document.getElementById('step-chart').innerHTML = REPORT_DATA.charts.steps.map(item => `<div class="bar-row"><div class="bar-label">${item.label}</div><div class="bar-track"><div class="bar green" style="width:${Math.max(2,item.strict_rate)}%"></div></div><div class="bar-value">${item.strict}/${item.tasks}</div></div>`).join('');
document.getElementById('insights').innerHTML = [
  `<div class="insight"><strong>总体：</strong>严格成功 ${S.strict} 题（${fmtPct(S.strict_rate)}）；另有 ${S.done-S.strict} 题完成了环境终局但没有严格满足目标，${S.total-S.done} 题在模型输出结束时没有终局。</div>`,
  `<div class="insight"><strong>交互合法性：</strong>共 ${S.guard_total} 次 Guard 拒绝，其中“点击不在上一条 observation”${S.guard_counts.click_not_in_previous_observation || 0} 次，“当前页面不可搜索”${S.guard_counts.search_not_available_on_current_page || 0} 次。</div>`,
  `<div class="insight"><strong>长程控制：</strong>${(S.reward_counts.repeat_loop || 0)+(S.reward_counts.max_steps || 0)} 题因循环或达到 ${show(M.max_steps)} 步上限结束；步数范围 ${S.min_steps}–${S.max_steps}，中位数 ${S.median_steps}。</div>`,
  `<div class="insight"><strong>购买决策：</strong>共调用购买 ${S.tool_counts.buy_now || 0} 次，最终严格成功 ${S.strict} 次；购买动作之后仍有 ${S.tool_counts.buy_now-S.strict} 次不是严格成功。</div>`
].join('');

document.getElementById('stats').innerHTML = [
  ['最终 Reward 范围', `${S.reward_min.toFixed(3)} ～ ${S.reward_max.toFixed(3)}`],
  ['Reward 中位数', S.reward_median.toFixed(3)],
  ['平均加权得分', S.weighted_score.toFixed(3)],
  ['平均工具动作数', (Object.values(S.tool_counts).reduce((a,b)=>a+b,0)/S.total).toFixed(2)],
  ['搜索次数 / 题', ((S.tool_counts.search_products || 0)/S.total).toFixed(2)],
  ['打开商品次数 / 题', ((S.tool_counts.open_product || 0)/S.total).toFixed(2)],
  ['选择规格次数 / 题', ((S.tool_counts.select_option || 0)/S.total).toFixed(2)],
  ['无终局任务', `${S.total-S.done}（${fmtPct((S.total-S.done)/S.total)}）`]
].map(([k,v]) => `<div class="stat-line"><span class="muted">${k}</span><strong>${v}</strong></div>`).join('');
document.getElementById('protocol').innerHTML = [
  ['模型',M.model], ['任务数',`${S.total}`], ['Reward 契约',M.reward_contract || '未记录'], ['温度 / top-p',`${show(M.temperature)} / ${show(M.top_p)}`], ['最大环境步数',`${show(M.max_steps)}`], ['最大生成 tokens',`${show(M.max_tokens)}`], ['上下文 tokenizer','由评测参数决定'], ['数据校验',`${S.total} 条任务级轨迹`]
].map(([k,v]) => `<div class="stat-line"><span class="muted">${k}</span><strong>${v}</strong></div>`).join('');

let sortKey = 'task_id'; let sortAsc = true;
const queryFilter = document.getElementById('query-filter'); const outcomeFilter = document.getElementById('outcome-filter');
function renderTable() {
  const query = queryFilter.value.trim().toLowerCase(); const outcome = outcomeFilter.value;
  const filtered = REPORT_DATA.rows.filter(row => (!query || String(row.task_id).includes(query) || row.query.toLowerCase().includes(query)) && (outcome === 'all' || row.reward_type === outcome));
  filtered.sort((a,b) => { const av=a[sortKey], bv=b[sortKey]; const x=typeof av==='string'?av.localeCompare(bv):av-bv; return sortAsc?x:-x; });
  document.getElementById('row-count').textContent = `显示 ${filtered.length} / ${REPORT_DATA.rows.length} 题`;
  document.getElementById('task-table').innerHTML = filtered.map(row => `<tr><td>${row.task_id}</td><td class="query" title="${esc(row.query)}">${esc(row.query)}</td><td><span class="pill ${outcomeClass(row.reward_type)}">${outcomeLabel(row.reward_type)}</span></td><td class="num">${row.steps}</td><td class="num">${row.reward.toFixed(3)}</td><td class="num">${row.guard_count}</td><td class="small">${esc(row.tool_sequence || '—')}</td></tr>`).join('');
}
document.querySelectorAll('[data-sort]').forEach(button => button.addEventListener('click', () => { const key=button.dataset.sort; sortAsc = key === sortKey ? !sortAsc : true; sortKey=key; renderTable(); }));
queryFilter.addEventListener('input', renderTable); outcomeFilter.addEventListener('change', renderTable); renderTable();
</script>
</body>
</html>
'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="从一次评测结果生成自包含 HTML 报告")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "glm-5.2",
        help="包含 summary.json 和 trajectories.jsonl 的评测目录",
    )
    parser.add_argument("--output", type=Path, help="HTML 输出路径，默认 <run-dir>/report.html")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report_path = args.output or args.run_dir / "report.html"
    data = json.dumps(build_data(args.run_dir), ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(HTML.replace("__REPORT_DATA__", data), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
