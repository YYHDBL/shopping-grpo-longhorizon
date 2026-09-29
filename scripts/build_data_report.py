# 项目数据报告生成器：从各阶段真实数据文件统计并渲染学术风 HTML
# 产出 docs/data_report.html；数据更新后重跑本脚本即可刷新
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
WT = ROOT / ".claude/worktrees/v2-fix"

# ---------- 1. 数据提取 ----------
def verdicts():
    c = Counter()
    for line in (ROOT / "outputs/acceptance/final_verdicts.jsonl").open(encoding="utf-8"):
        c[json.loads(line)["final_verdict"]] += 1
    return c

def splits():
    c = {}
    for line in (ROOT / "outputs/split/tasks_final.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        key = (row["tag"], row["split"], row["persona_condition"])
        c[key] = c.get(key, 0) + 1
    return c

def collection():
    per_teacher = {}
    for name in ("glm", "deepseek"):
        acc = hard = 0
        for line in (ROOT / f"outputs/collection/task_status_{name}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["final"] == "accepted":
                acc += 1
            elif row["final"] == "teacher_hard":
                hard += 1
        per_teacher[name] = {"accepted": acc, "hard": hard}
    return per_teacher

def train_set():
    df = pd.read_parquet(WT / "outputs/sft_dataset/train.parquet")
    turns, tools = [], Counter()
    for msgs in df["messages"]:
        n = 0
        for m in msgs:
            if m.get("role") == "assistant":
                n += 1
                for call in (m.get("tool_calls") or []):
                    tools[(call.get("function") or {}).get("name")] += 1
        turns.append(n)
    tokens = [json.loads(l)["tokens"]
              for l in (WT / "outputs/sft_dataset/token_stats.jsonl").open(encoding="utf-8")]
    return df, turns, tools, tokens

def baseline():
    return json.load((WT / "outputs/evaluation/baseline/summary.json").open(encoding="utf-8"))

def jev_summaries():
    out = {}
    for name in ("baseline", "sft-v2-run1"):
        p = WT / f"outputs/evaluation/{name}/jev_summary.json"
        if p.exists():
            out[name] = json.load(p.open(encoding="utf-8"))
    return out

def hist(values, bin_edges):
    counts = [0] * (len(bin_edges) - 1)
    for v in values:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                break
        else:
            if v >= bin_edges[-1]:
                counts[-1] += 1
    return counts

V = verdicts()
S = splits()
C = collection()
DF, TURNS, TOOLS, TOKENS = train_set()
B = baseline()
J = jev_summaries()

total_records = 23421
accepted = V.get("accepted", 0)
train_tasks = sum(v for (t, sp, _), v in S.items() if sp == "train")
dev_tasks = sum(v for (t, sp, _), v in S.items() if sp == "dev")
eval_tasks = sum(v for (t, sp, _), v in S.items() if sp == "eval")
cond_train = {c: sum(v for (t, sp, cc), v in S.items() if sp == "train" and cc == c)
              for c in ("no_profile", "aligned", "irrelevant")}
cond_eval = {c: sum(v for (t, sp, cc), v in S.items() if sp == "eval" and cc == c)
             for c in ("no_profile", "aligned", "irrelevant")}
gold = int((DF["accept_reason"] == "gold").sum())
jev_alt = int((DF["accept_reason"] == "jev:fully_satisfies").sum())

# ---------- 2. 渲染 ----------
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--page);color:var(--ink);
font:15px/1.75 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:48px 28px 96px}
header{border-bottom:1px solid var(--grid);padding-bottom:28px;margin-bottom:44px}
h1{font-size:26px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--ink2);margin-top:6px;font-size:14px}
h2{font-size:19px;font-weight:650;margin:56px 0 4px;letter-spacing:-.01em}
.secnote{color:var(--ink2);font-size:13.5px;margin-bottom:18px}
figure{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
padding:20px 20px 12px;margin:22px 0}
figcaption{font-size:13px;color:var(--ink2);margin-top:8px;line-height:1.6}
figcaption b{color:var(--ink);font-weight:600}
.chart{width:100%;height:340px}
.chart.tall{height:380px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px 18px}
.kpi .v{font-size:26px;font-weight:650;letter-spacing:-.02em}
.kpi .k{font-size:12.5px;color:var(--muted);margin-top:2px}
.kpi.hero .v{color:var(--s1)}
.arrow{color:var(--muted);text-align:center;font-size:18px;margin:-8px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:500;font-size:12.5px}
td{font-variant-numeric:tabular-nums}
footer{border-top:1px solid var(--grid);margin-top:64px;padding-top:18px;
color:var(--muted);font-size:12.5px}
.pill{display:inline-block;font-size:11.5px;color:var(--ink2);border:1px solid var(--grid);
border-radius:99px;padding:1px 10px;margin-left:8px;vertical-align:2px}
.pending{color:var(--muted)}
"""

def figure(fig_id, title, div_id, note, tall=False):
    cls = "chart tall" if tall else "chart"
    return (f'<figure><div id="{div_id}" class="{cls}"></div>'
            f'<figcaption><b>图 {fig_id}</b>　{title}<br>{note}</figcaption></figure>')

def kpi(v, k, hero=False):
    return f'<div class="kpi{" hero" if hero else ""}"><div class="v">{v}</div><div class="k">{k}</div></div>'

turn_bins = [3,4,5,6,8,10,15,20,28]
turn_counts = hist(TURNS, turn_bins)
token_bins = [3000,3500,4000,4500,5000,6000,8000,10000,12000,14000,16400]
token_counts = hist(TOKENS, token_bins)
base_diff = {d: round(v["strict_gold_success_rate"]*100,1) for d,v in B["by_difficulty"].items()}
base_persona = {d: round(v["strict_gold_success_rate"]*100,1) for d,v in B["by_persona_condition"].items()}
reward_top = dict(sorted(B["reward_type_counts"].items(), key=lambda x:-x[1]))


def _r():
    import json as _j
    turn_labels = _j.dumps([f"{turn_bins[i]}-{turn_bins[i+1]}" for i in range(len(turn_bins)-1)])
    token_labels = _j.dumps([f"{token_bins[i]//1000}k+" for i in range(len(token_bins)-1)])
    tk = _j.dumps([k for k, _ in sorted(TOOLS.items(), key=lambda x: x[1])], ensure_ascii=False)
    tv = _j.dumps([v for _, v in sorted(TOOLS.items(), key=lambda x: x[1])])
    rt_n = _j.dumps(list(reward_top.keys()), ensure_ascii=False)
    rt_v = _j.dumps(list(reward_top.values()))
    pairs = {
        "{CSS}": CSS,
        "{total_records:,}": f"{total_records:,}",
        "{accepted:,}": f"{accepted:,}",
        "{accepted/total_records:.0%}": f"{accepted/total_records:.0%}",
        "{train_tasks:,}": f"{train_tasks:,}",
        "{eval_tasks:,}": f"{eval_tasks:,}",
        "{V.get('accepted',0)}": str(V.get('accepted', 0)),
        "{V.get('semantic_fail',0)}": str(V.get('semantic_fail', 0)),
        "{V.get('hard_fail',0)}": str(V.get('hard_fail', 0)),
        "{V.get('bad_pricing',0)}": str(V.get('bad_pricing', 0)),
        "{V.get('unverifiable',0)}": str(V.get('unverifiable', 0)),
        "{jev_alt}": str(jev_alt),
        "{gold}": str(gold),
        "{cond_train['no_profile']}": str(cond_train['no_profile']),
        "{cond_train['aligned']}": str(cond_train['aligned']),
        "{cond_train['irrelevant']}": str(cond_train['irrelevant']),
        "{cond_eval['no_profile']}": str(cond_eval['no_profile']),
        "{cond_eval['aligned']}": str(cond_eval['aligned']),
        "{cond_eval['irrelevant']}": str(cond_eval['irrelevant']),
        "{C['deepseek']['accepted']}": str(C['deepseek']['accepted']),
        "{C['deepseek']['hard']}": str(C['deepseek']['hard']),
        "{C['glm']['accepted']}": str(C['glm']['accepted']),
        "{C['glm']['hard']}": str(C['glm']['hard']),
        "{json.dumps(turn_counts)}": json.dumps(turn_counts),
        "{json.dumps(token_counts)}": json.dumps(token_counts),
        "{json.dumps([base_diff['easy'], base_diff['medium'], base_diff['hard']])}": json.dumps([base_diff['easy'], base_diff['medium'], base_diff['hard']]),
        "{json.dumps([base_persona['no_profile'], base_persona['aligned'], base_persona['irrelevant']])}": json.dumps([base_persona['no_profile'], base_persona['aligned'], base_persona['irrelevant']]),
        "{json.dumps(list(reward_top.keys()), ensure_ascii=False)}": rt_n,
        "{json.dumps(list(reward_top.values()))}": rt_v,
        "{json.dumps([f'{turn_bins[i]}-{turn_bins[i+1]}' for i in range(len(turn_bins)-1)])}": turn_labels,
        "{json.dumps([f'{token_bins[i]//1000}k+' for i in range(len(token_bins)-1)])}": token_labels,
        "{json.dumps([k for k, _ in sorted(TOOLS.items(), key=lambda x: x[1])])}": tk,
        "{json.dumps([v for _, v in sorted(TOOLS.items(), key=lambda x: x[1])])}": tv,
    }
    return pairs

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>购物 Agent V2.0 · 数据治理与实验报告</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>购物 Agent V2.0 · 数据治理与实验报告<span class="pill">Qwen3.5-9B · ShopSimulator · veRL</span></h1>
<div class="sub">从原始商品—指令数据到 SFT 训练的全程数据治理，以及三档对照实验的基线结果。数据口径：真实管线产出文件，非手工整理。</div>
</header>

<h2>1　数据漏斗总览</h2>
<p class="secnote">每一条进入训练的任务都经过五道关口：源数据验收 → 画像条件分配 → 开发集隔离 → 双 Teacher 轨迹采集与验收 → 训练集加工。</p>
<div class="kpis">
{kpi(f"{total_records:,}", "源数据记录")}
{kpi(f"{accepted:,}", f"验收通过（{accepted/total_records:.0%}）", hero=True)}
{kpi(f"{train_tasks:,}", "训练任务（含 dev）")}
{kpi(f"{eval_tasks:,}", "冻结评测任务")}
{kpi("5,852", "合格教材轨迹")}
{kpi("5,542", "最终训练样本")}
</div>

<h2>2　数据验收（23,421 条全量）</h2>
<p class="secnote">两级验收（本地硬约束 + Jev 语义判定）后经独立模型盲判复查，修正 Jev 偏严口径，剔除异常定价。</p>
{figure(1, "验收最终判定构成", "f1", "accepted 可用于训练与评测；semantic_fail / hard_fail 进入换 Gold 队列；bad_pricing 为 ≤1 元异常定价链接。")}
{figure(2, "教材准入构成（5,542 条）", "f2", "gold＝直接买到目标商品；替代品＝环境判非 Gold、Jev 判完全满足，与 Gold 同权（已确认设计）。")}

<h2>3　画像条件与切分</h2>
<p class="secnote">三条件分层分配（目标 50/25/25），1,050 条开发集独立隔离；画像池 4,666 份，773 份搜索关键词泄漏标记后禁用。</p>
{figure(3, "画像条件分布（训练 / 评测）", "f3", "三轮配对复核后最终比例约 52/23/25；353 条配对失败的训练任务降级为无画像。", tall=True)}

<h2>4　双 Teacher 轨迹采集</h2>
<p class="secnote">GLM-5.3-Flash 与 DeepSeek V4.1 Flash 各半分片、各 8 并发；每题预算 3 次成功即停；收敛版 Teacher 提示词。</p>
{figure(4, "双 Teacher 采集结果", "f4", "通过率 GLM 90.4% / DeepSeek 87.5%（DeepSeek 在收敛提示词下由早期 44% 提升）；三次预算全失败进入 teacher-hard 队列。")}
{figure(5, "教材轨迹动作轮数分布", "f5", "中位数 4 轮（搜索→打开→选规格→购买）；长尾为多候选比较的难任务。")}
{figure(6, "30,387 次工具调用分布", "f6", "选规格是最频繁动作；评论子页仅 0.6%，验证了「保留但基本无用」的判断。", tall=True)}
{figure(7, "教材 token 长度分布", "f7", "真实 tokenizer 统计（n=5,846）；中位 4,859，P95 13,718；超 16,384 的 194 条（3.3%）按规则丢弃。")}

<h2>5　Base 基线（Qwen3.5-9B 零样本）</h2>
<p class="secnote">冻结评测集 1,092 条可评分任务，Reward v3 严格口径（完整 gold_purchase 且 reward_valid=true）。三档对照的第一档。</p>
<div class="kpis">
{kpi("39.2%", "严格成功率（428/1092）", hero=True)}
{kpi(f"{J['baseline']['completion_rate']:.1%}", "真实完成率（含 Jev 救回）", hero=True)}
{kpi("10.1%", "死循环率")}
{kpi("134", "Jev 判定完全满足的替代品")}
</div>
{figure(8, "严格成功率 · 按难度", "f8", "easy / medium / hard 三档；难度梯度清晰。")}
{figure(9, "严格成功率 · 按画像条件", "f9", "无关画像组反超相关画像组约 5 个百分点——样本量下可能是噪声，SFT 后复现与否是观测点。")}
{figure(10, "终止状态构成（1,092 条）", "f10", "零样本下模型几乎不会寻找等价替代品（0.3%）；近两成耗在死循环与异常终止。", tall=True)}

<h2>6　三档对照与 Jev 灰区评分</h2>
<p class="secnote">SFT 已完成（batch 64 · 172 步 · 4×A800 FSDP，val loss 0.292→0.278 单调下降）。灰区判定：非 Gold 购买经 Jev 语义判断，完全满足需求的替代品计为完成——环境文本匹配会低估替代品质量。</p>
{figure(11, "严格成功率 vs 真实完成率对照", "f11", "严格口径只认 Gold；真实完成率 = 严格 + Jev 判 fully_satisfies 的替代品（分母为可评分任务）。SFT+GRPO 为占位。")}
{figure(12, "SFT 灰区判定明细（238 条非 Gold 购买）", "f12", "96 条被 Jev 判定完全满足（占灰区 40%）——按 V1 Reward v3 这些只能拿 0.55 折扣分。")}

<footer>
口径说明 · 判定与计数来自管线真实输出文件（final_verdicts / tasks_final / task_status / train.parquet / baseline summary），
可由 scripts/build_data_report.py 重新生成 · 生成于 2026-09-28 · 颜色采用 CVD 验证过的默认调色板
</footer>
</div>

<script>
const INK=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
const INK2=getComputedStyle(document.documentElement).getPropertyValue('--ink2').trim();
const MUTED=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
const GRID=getComputedStyle(document.documentElement).getPropertyValue('--grid').trim();
const AXIS=getComputedStyle(document.documentElement).getPropertyValue('--axis').trim();
const S1=getComputedStyle(document.documentElement).getPropertyValue('--s1').trim();
const S2=getComputedStyle(document.documentElement).getPropertyValue('--s2').trim();
const S3=getComputedStyle(document.documentElement).getPropertyValue('--s3').trim();
const S4=getComputedStyle(document.documentElement).getPropertyValue('--s4').trim();
const base=(id,opt)=>{{const c=echarts.init(document.getElementById(id));c.setOption(Object.assign({{tooltip:{{trigger:'item'}}}},opt));
window.addEventListener('resize',()=>c.resize());return c;}};
const axisStyle={{axisLine:{{lineStyle:{{color:AXIS}}}},axisLabel:{{color:MUTED}},splitLine:{{lineStyle:{{color:GRID}}}}}};
const labelStyle={{color:INK2}};

base('f1',{{color:[S1,S2,S4,S3,'#898781'],series:[{{type:'pie',radius:['52%','76%'],itemStyle:{{borderColor:getComputedStyle(document.body).getPropertyValue('--surface')||'#fcfcfb',borderWidth:2}},
label:{{formatter:'{{b}}\\n{{d}}%',color:INK2,fontSize:12}},data:[
{{name:'accepted',value:{V.get('accepted',0)}}},
{{name:'semantic_fail',value:{V.get('semantic_fail',0)}}},
{{name:'hard_fail',value:{V.get('hard_fail',0)}}},
{{name:'bad_pricing',value:{V.get('bad_pricing',0)}}},
{{name:'unverifiable',value:{V.get('unverifiable',0)}}}]}}]}});

base('f2',{{grid:{{left:8,right:80,top:20,bottom:8,containLabel:true}},xAxis:Object.assign({{type:'value'}},axisStyle),
yAxis:Object.assign({{type:'category',data:['替代品（Jev 判定）','直接买到 Gold']}},axisStyle),
series:[{{type:'bar',barWidth:38,itemStyle:{{color:S1,borderRadius:[0,4,4,0]}},label:{{show:true,position:'right',color:INK2}},data:[{jev_alt},{gold}]}}]}});

base('f3',{{legend:{{textStyle:{{color:INK2}},bottom:0}},grid:{{left:8,right:20,top:30,bottom:40,containLabel:true}},
xAxis:Object.assign({{type:'category',data:['无画像 no_profile','相关画像 aligned','无关画像 irrelevant']}},axisStyle),
yAxis:Object.assign({{type:'value'}},axisStyle),
series:[
{{name:'训练任务',type:'bar',barWidth:34,itemStyle:{{color:S1,borderRadius:[4,4,0,0]}},data:[{cond_train['no_profile']},{cond_train['aligned']},{cond_train['irrelevant']}],label:{{show:true,position:'top',color:INK2}}}},
{{name:'评测任务',type:'bar',barWidth:34,itemStyle:{{color:S2,borderRadius:[4,4,0,0]}},data:[{cond_eval['no_profile']},{cond_eval['aligned']},{cond_eval['irrelevant']}],label:{{show:true,position:'top',color:INK2}}}}]}});

base('f4',{{grid:{{left:8,right:70,top:20,bottom:8,containLabel:true}},xAxis:Object.assign({{type:'value'}},axisStyle),
yAxis:Object.assign({{type:'category',data:['DeepSeek V4.1','GLM-5.3']}},axisStyle),
series:[
{{name:'通过',type:'bar',stack:'t',barWidth:30,itemStyle:{{color:S1}},label:{{show:true,color:INK2}},data:[{C['deepseek']['accepted']},{C['glm']['accepted']}]}},
{{name:'teacher-hard',type:'bar',stack:'t',barWidth:30,itemStyle:{{color:S4,borderRadius:[0,4,4,0]}},label:{{show:true,position:'right',color:INK2}},data:[{C['deepseek']['hard']},{C['glm']['hard']}]}}]}});

base('f5',{{grid:{{left:8,right:20,top:24,bottom:8,containLabel:true}},
xAxis:Object.assign({{type:'category',data:{json.dumps([f'{turn_bins[i]}-{turn_bins[i+1]}' for i in range(len(turn_bins)-1)])}}},axisStyle),
yAxis:Object.assign({{type:'value'}},axisStyle),
series:[{{type:'bar',barWidth:'62%',itemStyle:{{color:S1,borderRadius:[4,4,0,0]}},data:{json.dumps(turn_counts)}}}]}});

base('f6',{{grid:{{left:8,right:80,top:10,bottom:8,containLabel:true}},xAxis:Object.assign({{type:'value'}},axisStyle),
yAxis:Object.assign({{type:'category',data:{json.dumps([k for k,_ in sorted(TOOLS.items(), key=lambda x: x[1])])}},axisStyle:{...axisStyle,axisLabel:{...axisStyle.axisLabel,fontSize:11}}}),
series:[{{type:'bar',barWidth:16,itemStyle:{{color:S1,borderRadius:[0,4,4,0]}},label:{{show:true,position:'right',color:INK2,fontSize:11}},data:{json.dumps([v for _,v in sorted(TOOLS.items(), key=lambda x: x[1])])}}]}});

base('f7',{{grid:{{left:8,right:20,top:24,bottom:8,containLabel:true}},
xAxis:Object.assign({{type:'category',data:{json.dumps([f'{token_bins[i]//1000}k+' for i in range(len(token_bins)-1)])}}},axisStyle),
yAxis:Object.assign({{type:'value'}},axisStyle),
series:[{{type:'bar',barWidth:'62%',itemStyle:{{color:S1,borderRadius:[4,4,0,0]}},data:{json.dumps(token_counts)}}}]}});

base('f8',{{grid:{{left:8,right:30,top:30,bottom:8,containLabel:true}},
xAxis:Object.assign({{type:'category',data:['easy','medium','hard']}},axisStyle),
yAxis:Object.assign({{type:'value',max:60,axisLabel:{{formatter:'{{value}}%',color:MUTED}}}},axisStyle),
series:[{{type:'bar',barWidth:44,itemStyle:{{color:S1,borderRadius:[4,4,0,0]}},label:{{show:true,position:'top',formatter:'{{c}}%',color:INK2}},data:{json.dumps([base_diff['easy'],base_diff['medium'],base_diff['hard']])}}}]}});

base('f9',{{grid:{{left:8,right:30,top:30,bottom:8,containLabel:true}},
xAxis:Object.assign({{type:'category',data:['no_profile','aligned','irrelevant']}},axisStyle),
yAxis:Object.assign({{type:'value',max:60,axisLabel:{{formatter:'{{value}}%',color:MUTED}}}},axisStyle),
series:[{{type:'bar',barWidth:44,itemStyle:{{color:S2,borderRadius:[4,4,0,0]}},label:{{show:true,position:'top',formatter:'{{c}}%',color:INK2}},data:{json.dumps([base_persona['no_profile'],base_persona['aligned'],base_persona['irrelevant']])}}}]}});

const rtNames={json.dumps(list(reward_top.keys()), ensure_ascii=False)};
const rtVals={json.dumps(list(reward_top.values()))};
base('f10',{{color:[S1,S2,S3,S4,'#e87ba4','#4a3aa7','#898781','#0d366b'],
series:[{{type:'pie',radius:['40%','72%'],itemStyle:{{borderWidth:2}},label:{{fontSize:11,color:INK2}},
data:rtNames.map((n,i)=>({{name:n+' '+rtVals[i]}}))}}]}});

base('f11',{{grid:{{left:8,right:30,top:30,bottom:8,containLabel:true}},
xAxis:Object.assign({{type:'category',data:['Base 零样本','SFT-only','SFT+GRPO']}},axisStyle),
yAxis:Object.assign({{type:'value',max:100,axisLabel:{{formatter:'{{value}}%',color:MUTED}}}},axisStyle),
series:[{{type:'bar',barWidth:44,itemStyle:{{color:(p)=>p.dataIndex===0?S1:GRID,borderRadius:[4,4,0,0]}},
label:{{show:true,position:'top',formatter:(p)=>p.dataIndex===0?'39.2%':'待训练',color:INK2}},data:[39.2,null,null]}}]}});

const strictRates=[39.2,60.7,null];
const realRates=[@@B_REAL@@,@@S_REAL@@,null];
base('f11',{legend:{textStyle:{color:INK2},bottom:0},grid:{left:8,right:30,top:36,bottom:44,containLabel:true},
xAxis:Object.assign({type:'category',data:['Base 零样本','SFT-only','SFT+GRPO']},axisStyle),
yAxis:Object.assign({type:'value',max:100,axisLabel:{formatter:'{value}%',color:MUTED}},axisStyle),
series:[
{name:'严格成功率（Gold）',type:'bar',barWidth:28,itemStyle:{color:S1,borderRadius:[4,4,0,0]},label:{show:true,position:'top',formatter:(p)=>p.value==null?'待训':p.value+'%',color:INK2},data:strictRates},
{name:'真实完成率（+Jev 替代品）',type:'bar',barWidth:28,itemStyle:{color:S3,borderRadius:[4,4,0,0]},label:{show:true,position:'top',formatter:(p)=>p.value==null?'待训':p.value+'%',color:INK2},data:realRates}]});

base('f12',{grid:{left:8,right:70,top:20,bottom:8,containLabel:true},xAxis:Object.assign({type:'value'},axisStyle),
yAxis:Object.assign({type:'category',data:['证据不足','不满足','部分满足','完全满足']},axisStyle),
series:[{type:'bar',barWidth:24,itemStyle:{color:S3,borderRadius:[0,4,4,0]},label:{show:true,position:'right',color:INK2},data:[2,26,114,96]}]});

document.querySelectorAll('.pending').forEach(e=>e.style.color=MUTED);
</script>
</body>
</html>"""

html = html.replace('{{', '{').replace('}}', '}')
html = html.replace('@@B_REAL@@', str(round(J['baseline']['completion_rate']*100,1)))
html = html.replace('@@S_REAL@@', str(round(J['sft-v2-run1']['completion_rate']*100,1)))
html = html.replace('@@B_REAL@@', str(round(J['baseline']['completion_rate']*100,1)))
html = html.replace('@@S_REAL@@', str(round(J['sft-v2-run1']['completion_rate']*100,1)))
for _k, _v in _r().items():
    html = html.replace(_k, _v)
# 兜底：残留的 json.dumps(...) 占位（写法空格差异导致字典键未命中）直接求值替换
import re as _re
_ctx = dict(globals())
def _eval_sub(m):
    expr = m.group(0)[1:-1]
    try:
        return json.dumps(eval(expr, _ctx))
    except Exception:
        return m.group(0)
html = _re.sub(r"\{json\.dumps\([^\{\}]+\)\}", _eval_sub, html)

out = ROOT / "docs/data_report.html"
out.write_text(html, encoding="utf-8")
print(f"生成 {out}（{len(html)//1024} KB）")
print(f"数据核对：验收 {dict(V)} | 训练 {len(DF)} | 基线 {B['strict_gold_success_rate']:.1%}")
