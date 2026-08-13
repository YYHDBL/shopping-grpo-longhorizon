#!/usr/bin/env python3
"""Build a self-contained comparison report for Final-200 evaluation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTOGRAM_EDGES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
MODEL_ANALYSIS = {
    "qwen3.7-max": {
        "portrait": "整体最稳，搜索召回通常够用，找到后也敢下单；代价是偶尔太果断，买了第一个看着差不多的候选。",
        "bullets": [
            "51 个 bad case：12 个买了相似款，15 个找对商品但规格没选对或没落到购买记录。",
            "8 个在重复搜索或 35 步里耗尽，6 个触发上下文/生成长度限制，4 个最终价格超预算。",
            "已知目标 ASIN 的 38 个失败里，22 个其实打开过目标，另 15 个目标已经在搜索结果里，只是没点开。",
            "例：9545 跳过首屏第 1 名的精确硬盘盒，点第 9 名通用款就买；20117 点过黑色和 42.5 码，但购买记录只留下尺码；14928 找到 19 元目标后继续逛，最后反而买了 23 元超预算商品。",
        ],
        "fix": "保留两三个候选再决策；买前重新读取 selected_options 和实际价格，别把“点过”当成“已生效”。",
    },
    "deepseek-v4-flash": {
        "portrait": "不是搜不到，而是找到以后还不放心，来回比来比去，常把自己绕死。",
        "bullets": [
            "61 个 bad case 中有 22 个 repeat_loop；28 个至少发生过一次“打开商品后没查信息就退”。",
            "32 个失败购买里，18 个是目标商品但规格没选对，8 个买了相似品，6 个踩预算硬门槛。",
            "33 个 bad task 出现 52 次守卫拦截，多数是点击旧页面里的 ASIN 或在商品页直接搜索。",
            "例：11773 已选中正确狗粮规格又回去重选；3025 把外径 16mm 选成 14mm；14877 把 Quinny Yezz Air 买成相似的 Zapp Xpress。",
        ],
        "fix": "正确候选第二次出现就强制决策；打开商品后至少核对价格、规格和一项详情再决定退回。",
    },
    "glm-5.2": {
        "portrait": "主要不是搜不到，而是都走到门口了，却没把最后一步做完。",
        "bullets": [
            "75 个 bad case 中 49 个以 assistant_final 直接停掉：8 个只搜不点、6 个刚打开就停、6 个看过详情没选规格、29 个已深入核验或选好规格仍没买。",
            "另有 10 个找对商品但规格/购买记录不对，12 个循环或耗尽 35 步，3 个超预算，只有 1 个是明显买错相似品。",
            "例：8187 首屏第 1 名就是正确木梳，也识别出正确规格，却没调用 select_option；4154 已逐项确认 LC16PB 满足要求，却没调用 buy_now；10633 说要返回商品页，但没有真的调用返回工具。",
            "它常把 Description、Features、Reviews 全看一遍，即使页面没新增信息；成功轨迹平均 9.31 步，比 Qwen Max 多约 1.9 步。",
        ],
        "fix": "环境还没 terminal 时，纯文字输出应自动重试成一个合法工具调用；信息页没新增证据就别机械全看。",
    },
    "qwen3.7-flash": {
        "portrait": "比较敢买，但经常第一个看着差不多就下单，主要输在规格没核对。",
        "bullets": [
            "76 个 bad case；38 个失败购买中有 28 个只比较了 1 个唯一候选，32 个存在规格/选项不匹配。",
            "25 个 repeat_loop，23 个出现“打开后立刻返回”；46 个 bad task 至少一次工具调用被守卫拦截。",
            "9 个 max_steps、3 个 ContextBudgetError；纯英文搜索在中文商城里也造成至少 10 个明显失败。",
            "例：20117 只看一双 AJ1，漏选黑色就买；5510 要 S-2只装却选 L-2只装；15991 连看 6 款男士防晒，全都没开 description/features。",
        ],
        "fix": "先用中文短关键词，至少比较两款；下单前逐轴核对颜色、型号、尺码、数量和选后价格。",
    },
    "deepseek-v4-pro": {
        "portrait": "最大短板不是购物判断，而是工具参数格式；其次也是找到答案后继续折腾。",
        "bullets": [
            "77 个 bad case 中 29 个直接死于 invalid_action_limit；139 次守卫拒绝来自给无参工具乱塞 name、reason 或 {\"{}\":{}}。",
            "18 个 repeat_loop，至少 16 个已经出现过很强甚至正确的候选；29 个 bad case出现只看标题就退。",
            "21 个失败购买里，15 个是目标商品但规格没锁全，4 个买相似品，2 个超预算。",
            "例：18522 连续三次 view_description 都传错参数；8187 已选中正确木梳规格仍继续搜索；20117 尺码选对但颜色规格没保留。",
        ],
        "fix": "所有无参工具只能发严格的 {}；第一次被拒绝后按错误重写，禁止原样重试。",
    },
    "qwen3.7-plus": {
        "portrait": "它不是主要输在乱买，而是太犹豫：一直找、一直退、迟迟不买。",
        "bullets": [
            "125 个 bad case 里 113 个根本没下单；70 个 repeat_loop，36 个 assistant_length_limit。",
            "64 个 bad case 出现只看标题就退，共 137 次；114 个 bad task 至少一次被守卫拦截，共 312 次。",
            "失败购买只有 12 个，真正买错 ASIN 仅 1 个；更大的问题是比较太多却不收口。",
            "例：4154 已选好 LC16PB 军绿色仍重复搜索；1212 对 3 个淋浴底盘完整重复 description/features/reviews；9918、11000 搜一次就因输出长度中断。",
        ],
        "fix": "设置停机规则：最多比较三款；满足硬条件、预算和规格就买，并给最终工具调用预留输出长度。",
    },
}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _histogram(values: list[float]) -> list[int]:
    counts = [0] * (len(HISTOGRAM_EDGES) - 1)
    for value in values:
        for index, (lower, upper) in enumerate(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])):
            if lower <= value < upper:
                counts[index] += 1
                break
    return counts


def _reward_type(row: dict) -> str:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return detail.get("reward_type") or terminal.get("reward_type") or "unknown"


def _model_data(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(run_dir / "trajectories.jsonl")
    rewards = [float(row.get("final_reward") or 0.0) for row in rows]
    protocol = summary.get("protocol") or {}
    outcome_counts = summary.get("reward_type_counts") or Counter(_reward_type(row) for row in rows)
    success_ids = summary.get("strict_success_task_ids") or [
        int(row["task_id"]) for row in rows if _reward_type(row) == "gold_purchase"
    ]
    return {
        "key": run_dir.name,
        "name": protocol.get("model") or run_dir.name,
        "report": f"{run_dir.name}/report.html",
        "tasks": len(rows),
        "successes": int(summary.get("strict_successes", len(success_ids))),
        "success_rate": float(summary.get("strict_success_rate", len(success_ids) / len(rows))),
        "purchase_rate": float(summary.get("purchase_success_rate", 0.0)),
        "reward_valid_rate": float(summary.get("reward_valid_rate", 0.0)),
        "average_steps": float(summary.get("average_steps", 0.0)),
        "outcomes": dict(outcome_counts),
        "statuses": summary.get("status_counts") or dict(Counter(row.get("status", "unknown") for row in rows)),
        "guards": summary.get("guard_reason_counts") or {},
        "success_ids": success_ids,
        "reward": {
            "mean": statistics.fmean(rewards),
            "median": statistics.median(rewards),
            "stddev": statistics.pstdev(rewards),
            "min": min(rewards),
            "q1": _quantile(rewards, 0.25),
            "q3": _quantile(rewards, 0.75),
            "max": max(rewards),
            "histogram": _histogram(rewards),
        },
    }


def build_comparison_data(evaluation_dir: Path) -> dict:
    run_dirs = sorted(
        path
        for path in evaluation_dir.iterdir()
        if path.is_dir()
        and (path / "summary.json").is_file()
        and (path / "trajectories.jsonl").is_file()
    )
    models = [_model_data(path) for path in run_dirs]
    if not models:
        raise ValueError(f"no evaluation runs found under {evaluation_dir}")
    all_task_ids = set().union(*(set(model["success_ids"]) for model in models))
    for path in run_dirs:
        all_task_ids.update(int(row["task_id"]) for row in _load_jsonl(path / "trajectories.jsonl"))
    solved_by = Counter(
        sum(task_id in set(model["success_ids"]) for model in models) for task_id in all_task_ids
    )
    best = max(models, key=lambda model: model["success_rate"])
    return {
        "models": models,
        "outcome_order": [
            "gold_purchase",
            "valid_alternative_purchase",
            "partial_alternative_purchase",
            "wrong_purchase",
            "repeat_loop",
            "max_steps",
            "early_abstain",
            "graceful_stop",
            "reward_unverifiable",
            "unknown",
        ],
        "histogram_labels": [
            f"{HISTOGRAM_EDGES[index]:.1f}～{min(HISTOGRAM_EDGES[index + 1], 1.0):.1f}"
            for index in range(len(HISTOGRAM_EDGES) - 1)
        ],
        "agreement": [{"models": count, "tasks": solved_by.get(count, 0)} for count in range(len(models) + 1)],
        "all_failed_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if not any(task_id in set(model["success_ids"]) for model in models)
        ),
        "all_succeeded_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if all(task_id in set(model["success_ids"]) for model in models)
        ),
        "best_model": best["name"],
        "analysis": MODEL_ANALYSIS,
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final-200 多模型轨迹与 Bad Case 综合报告</title>
<style>
:root{--ink:#142033;--muted:#64748b;--line:#dbe4ef;--bg:#f3f6fa;--blue:#2563eb;--green:#16a34a;--amber:#d97706;--red:#dc2626;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1440px;margin:auto;padding:28px}.hero,.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px #0f172a0a}
.hero{padding:28px;margin-bottom:18px;background:linear-gradient(135deg,#fff 55%,#e8f0ff)}h1{margin:0 0 8px;font-size:30px}h2{font-size:20px;margin:0 0 14px}h3{margin:0 0 8px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:18px 0}.card{padding:20px;overflow:auto}.wide{grid-column:1/-1}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin-top:18px}.kpi{padding:14px;border-radius:12px;background:#f8fafc;border:1px solid var(--line)}.kpi b{display:block;font-size:24px}
table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#fff}th{color:var(--muted);font-size:12px}
.bar-row{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:9px 0}.track{height:12px;background:#eef2f7;border-radius:99px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:99px}.small{font-size:12px}.hist{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;align-items:end;margin:12px 0}.hist-name{align-self:center}.hist-bin{height:92px;background:#f1f5f9;display:flex;align-items:end;border-radius:5px 5px 0 0;overflow:hidden}.hist-bin i{display:block;width:100%;background:var(--violet);min-height:2px}.hist-labels{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;color:var(--muted);font-size:10px}.hist-labels span{writing-mode:vertical-rl;height:58px}
.stack{display:flex;height:18px;border-radius:99px;overflow:hidden;background:#eef2f7}.seg{height:100%}.outcome-row{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:12px 0}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
.notes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.note{padding:14px;border:1px solid var(--line);border-radius:12px;background:#fbfdff}.note ul{padding-left:20px;margin:7px 0}.task-ids{word-break:break-all;padding:12px;background:#f8fafc;border-radius:10px;color:#475569}.links a{display:inline-block;margin:5px 10px 5px 0;padding:7px 11px;border-radius:9px;background:#eff6ff;color:#1d4ed8;text-decoration:none}
@media(max-width:900px){main{padding:14px}.grid{grid-template-columns:1fr}.kpis,.notes{grid-template-columns:1fr 1fr}.hist{grid-template-columns:100px repeat(10,minmax(15px,1fr))}.hist-labels{grid-template-columns:100px repeat(10,minmax(15px,1fr))}}
</style></head><body><main>
<section class="hero"><div class="muted">Shopping Agent · Final-200</div><h1>多模型轨迹与 Bad Case 综合报告</h1><p>同一套 200 题、6 个模型。先看成功率和 Reward 分布，再看失败轨迹为什么会错、强模型强在哪里。</p><div class="kpis" id="kpis"></div><div class="links" id="links"></div></section>
<div class="grid">
<section class="card"><h2>严格成功率</h2><div id="success-bars"></div></section>
<section class="card"><h2>任务共识难度</h2><p class="muted">横轴含义：一道题被多少个模型做对。</p><div id="agreement-bars"></div></section>
<section class="card wide"><h2>描述性统计</h2><p class="muted">Reward 统计包含全部 200 条；未形成可验证终局的轨迹按其记录值（通常为 0）计入。</p><div id="reward-reading"></div><table><thead><tr><th>模型</th><th>严格成功</th><th>购买成功率</th><th>Reward 有效率</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小</th><th>P25</th><th>P75</th><th>最大</th><th>平均步数</th></tr></thead><tbody id="stats"></tbody></table></section>
<section class="card wide"><h2>Reward 分布</h2><p class="muted">每一小柱是一个 0.2 宽区间；紫柱越高，落在该 Reward 区间的任务越多。</p><div id="histograms"></div><div class="hist-labels" id="hist-labels"></div></section>
<section class="card wide"><h2>终局类型分布</h2><div class="legend" id="legend"></div><div id="outcomes"></div></section>
<section class="card wide"><h2>核心结论：强模型强在哪</h2><div id="strengths"></div></section>
<section class="card wide"><h2>各模型主要 Bad Case</h2><div class="notes" id="model-notes"></div></section>
<section class="card wide"><h2>跨模型共性</h2><div id="common-notes"></div><h3>所有模型都没做对的任务（<span id="all-failed-count"></span>）</h3><div class="task-ids" id="all-failed"></div><h3 style="margin-top:16px">所有模型都做对的任务（<span id="all-succeeded-count"></span>）</h3><div class="task-ids" id="all-succeeded"></div></section>
<section class="card wide"><h2>需要谨慎解读的评测数据</h2><div id="data-notes"></div></section>
</div></main>
<script>
const D=__REPORT_DATA__;
const COLORS={gold_purchase:'#16a34a',valid_alternative_purchase:'#4ade80',partial_alternative_purchase:'#f59e0b',wrong_purchase:'#ef4444',repeat_loop:'#7c3aed',max_steps:'#db2777',early_abstain:'#94a3b8',graceful_stop:'#38bdf8',reward_unverifiable:'#a16207',unknown:'#cbd5e1'};
const LABELS={gold_purchase:'正确购买',valid_alternative_purchase:'有效替代品',partial_alternative_purchase:'部分匹配',wrong_purchase:'买错商品',repeat_loop:'重复循环',max_steps:'步数耗尽',early_abstain:'过早放弃',graceful_stop:'主动停止',reward_unverifiable:'无法核验',unknown:'未形成终局'};
const pct=x=>(x*100).toFixed(1)+'%'; const n=x=>Number(x).toFixed(3); const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sorted=[...D.models].sort((a,b)=>b.success_rate-a.success_rate); const total=D.models[0].tasks;
document.querySelector('#kpis').innerHTML=`<div class="kpi"><span>模型</span><b>${D.models.length}</b></div><div class="kpi"><span>每模型任务</span><b>${total}</b></div><div class="kpi"><span>最高严格成功率</span><b>${pct(sorted[0].success_rate)}</b><small>${esc(sorted[0].name)}</small></div><div class="kpi"><span>全模型共同失败</span><b>${D.all_failed_task_ids.length}</b></div>`;
document.querySelector('#links').innerHTML=sorted.map(m=>`<a href="${encodeURI(m.report)}">${esc(m.name)} 单模型报告</a>`).join('');
function bars(target,items,max,label){document.querySelector(target).innerHTML=items.map(x=>`<div class="bar-row"><span>${esc(x.name)}</span><div class="track"><div class="fill" style="width:${x.value/max*100}%"></div></div><b>${label(x)}</b></div>`).join('')}
bars('#success-bars',sorted.map(m=>({name:m.name,value:m.success_rate,count:m.successes})),1,x=>`${x.count}/${total}`);
bars('#agreement-bars',D.agreement.map(x=>({name:`${x.models} 个模型`,value:x.tasks})),Math.max(...D.agreement.map(x=>x.tasks)),x=>x.value);
document.querySelector('#stats').innerHTML=sorted.map(m=>`<tr><td><a href="${encodeURI(m.report)}">${esc(m.name)}</a></td><td>${m.successes}/${m.tasks}（${pct(m.success_rate)}）</td><td>${pct(m.purchase_rate)}</td><td>${pct(m.reward_valid_rate)}</td><td>${n(m.reward.mean)}</td><td>${n(m.reward.median)}</td><td>${n(m.reward.stddev)}</td><td>${n(m.reward.min)}</td><td>${n(m.reward.q1)}</td><td>${n(m.reward.q3)}</td><td>${n(m.reward.max)}</td><td>${m.average_steps.toFixed(2)}</td></tr>`).join('');
document.querySelector('#reward-reading').innerHTML=`<p>分布很明显是“两头多、中间少”：正确购买直接落在 1.0，循环、步数耗尽和买错则集中在负分。${esc(sorted[0].name)} 的 P25 仍有 ${n(sorted[0].reward.q1)}，说明至少四分之三任务没有掉到低分区；${esc(sorted[sorted.length-1].name)} 的中位数是 ${n(sorted[sorted.length-1].reward.median)}、P25 是 ${n(sorted[sorted.length-1].reward.q1)}，一半任务连正向终局都没有。标准差越大表示越不稳：本批最高是 ${esc([...sorted].sort((a,b)=>b.reward.stddev-a.reward.stddev)[0].name)}（${n([...sorted].sort((a,b)=>b.reward.stddev-a.reward.stddev)[0].reward.stddev)}）。</p>`;
document.querySelector('#histograms').innerHTML=sorted.map(m=>{const max=Math.max(...m.reward.histogram);return `<div class="hist"><b class="hist-name">${esc(m.name)}</b>${m.reward.histogram.map(v=>`<div class="hist-bin" title="${v} 条"><i style="height:${v/max*100}%"></i></div>`).join('')}</div>`}).join('');
document.querySelector('#hist-labels').innerHTML='<b></b>'+D.histogram_labels.map(x=>`<span>${x}</span>`).join('');
document.querySelector('#legend').innerHTML=D.outcome_order.map(k=>`<span><i style="background:${COLORS[k]}"></i>${LABELS[k]}</span>`).join('');
document.querySelector('#outcomes').innerHTML=sorted.map(m=>`<div class="outcome-row"><b>${esc(m.name)}</b><div class="stack">${D.outcome_order.map(k=>`<div class="seg" title="${LABELS[k]}：${m.outcomes[k]||0}" style="width:${(m.outcomes[k]||0)/m.tasks*100}%;background:${COLORS[k]}"></div>`).join('')}</div></div>`).join('');
const best=sorted[0], weakest=sorted[sorted.length-1];
document.querySelector('#strengths').innerHTML=`<p><b>${esc(best.name)}</b> 是这批里最稳的：严格成功 ${best.successes}/${best.tasks}（${pct(best.success_rate)}），平均只走 ${best.average_steps.toFixed(2)} 步。它不是“想得更久”，而是更常在前几次搜索里锁定靠谱候选，利用商品页已有的标题、属性、规格和价格，够用就选、选完就买。</p><p>和 GLM-5.2 逐题比，两者共同做对 114 题；Qwen Max 单独做对 35 题，GLM 单独做对 11 题。Qwen Max 的成功轨迹平均 7.44 步，GLM 是 9.31 步。GLM 的主要损失是 49 条 assistant_final——不少轨迹已经想清楚下一步，甚至选好规格，却没有真的发工具调用。</p><p>最明显的差距在“能不能收尾”：${esc(best.name)} 的重复循环只有 ${best.outcomes.repeat_loop||0} 条、未形成终局 ${best.outcomes.unknown||0} 条；${esc(weakest.name)} 分别是 ${weakest.outcomes.repeat_loop||0} 和 ${weakest.outcomes.unknown||0}。强模型少走回头路，也更少在已经接近答案时卡住。不过 Qwen Max 偶尔太果断，会把近似商品当答案；改进方向是加一遍轻量规格检查，不是学 GLM 把所有空页签都看一遍。</p>`;
document.querySelector('#model-notes').innerHTML=sorted.map(m=>{const a=D.analysis[m.key];return `<article class="note"><h3>${esc(m.name)}</h3><p><b>${esc(a.portrait)}</b></p><ul>${a.bullets.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><p><b>怎么改：</b>${esc(a.fix)}</p></article>`}).join('');
document.querySelector('#common-notes').innerHTML=`<p><b>第一类是“看标题就退”。</b>候选标题不够像时，模型常常打开后立即返回，没有继续看完整属性、规格轴和变体价；这在 Qwen Plus（64 个 bad case）、DeepSeek Pro（29 个）和 DeepSeek Flash（28 个）里尤其明显。</p><p><b>第二类是“找到后不收口”。</b>正确候选已经出现，模型仍换同义搜索词、重复打开商品或来回切规格。强模型 Qwen Max 只有 4 个 repeat_loop，Qwen Plus 有 70 个，差距主要就在这里。</p><p><b>第三类是“规格轴没管住”。</b>型号、颜色、尺码、容量、数量经常只选一部分，或者点过但最终购买状态没保留。预算也应按最终变体价核对，而不是拿列表价格凭感觉。</p><p><b>第四类是“页面状态没跟上”。</b>模型拿旧页面的 ASIN/按钮继续点，或给无参工具乱传参数。DeepSeek Pro 的 29 条 invalid_action_limit 是最集中的系统性问题。</p><p>${D.all_failed_task_ids.length} 道题六个模型全败，说明这部分通常有大量近似品，必须靠详情或精确规格区分；${D.all_succeeded_task_ids.length} 道题六个模型全对，说明基础搜索和明显匹配项并不是主要瓶颈。</p>`;
document.querySelector('#data-notes').innerHTML=`<p>raw bad 数不能全当成模型真实错误。逐轨迹检查发现几类疑似标签/比较器问题：</p><ul><li><b>需求与 gold 冲突：</b>5703 用户明确要 7 号机针，gold 却要 8 号；21785 用户要 2XL，gold 却是 XL；5510 用户要 L 码，gold 却含 S-2只装。</li><li><b>需求没写、gold 却强制：</b>4786 没写尺码却要求 XL女175；3368 只要求高度 15cm 以下，9cm 合理但 gold 强制 12cm。</li><li><b>字符归一化：</b>11168、5904、2352、12860 存在 ➕、爱心、大小写等字符串看起来等价却匹配失败。</li><li><b>口语预算被当硬上限：</b>“60 元出头”买 62、“40 左右”买 45、“170 上下”买 171 都会被判失败。报告保留官方严格成功率，但这些案例不宜直接归因于模型推理。</li></ul>`;
document.querySelector('#all-failed-count').textContent=D.all_failed_task_ids.length;document.querySelector('#all-failed').textContent=D.all_failed_task_ids.join(', ');document.querySelector('#all-succeeded-count').textContent=D.all_succeeded_task_ids.length;document.querySelector('#all-succeeded').textContent=D.all_succeeded_task_ids.join(', ');
</script></body></html>'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成 Final-200 多模型综合 HTML 报告")
    parser.add_argument(
        "--evaluation-dir", type=Path, default=ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.evaluation_dir / "comparison-report.html"
    data = json.dumps(build_comparison_data(args.evaluation_dir), ensure_ascii=False, separators=(",", ":"))
    output.write_text(HTML.replace("__REPORT_DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
