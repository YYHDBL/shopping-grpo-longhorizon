"""Self-contained evaluator dashboard for feed-shopping trajectories.

The generated HTML has no network dependencies.  It is deliberately an
evaluator view: actions, delayed events and reward components are visible, but
private latent simulator state is never serialized.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from shopping_grpo.feed.manifest import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from shopping_grpo.feed.schema import iter_jsonl


DEMO_SCHEMA_VERSION = "feed-demo-v1"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _trajectory_from_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Normalize the supported log/result envelopes for the browser."""
    nested = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    transitions = row.get("transitions")
    if transitions is None:
        transitions = row.get("trajectory")
    if transitions is None:
        transitions = nested.get("transitions")
    transitions = transitions if isinstance(transitions, (list, tuple)) else []
    episode_id = str(
        row.get("episode_id")
        or row.get("task_id")
        or nested.get("episode_id")
        or f"episode-{index:04d}"
    )
    policy = str(
        row.get("policy")
        or row.get("policy_name")
        or row.get("behavior_policy")
        or "unknown"
    )
    summary = row.get("summary")
    if not isinstance(summary, Mapping):
        summary = nested.get("summary") if isinstance(nested.get("summary"), Mapping) else {}
    episode_return = row.get("episode_return")
    if episode_return is None:
        episode_return = summary.get("episode_return", nested.get("episode_return", 0.0))
    return {
        "episode_id": episode_id,
        "policy": policy,
        "episode_return": float(episode_return or 0.0),
        "summary": _json_safe(summary),
        "transitions": _json_safe(transitions),
    }


def build_demo_payload(
    rows: Iterable[Mapping[str, Any]],
    *,
    evaluation_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trajectories = [_trajectory_from_row(row, index) for index, row in enumerate(rows)]
    if not trajectories:
        raise ValueError("at least one trajectory is required to build the demo")
    trajectories.sort(key=lambda row: (row["policy"], row["episode_id"]))
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "evaluation_summary": _json_safe(evaluation_summary or {}),
        "trajectories": trajectories,
    }


def render_demo_html(payload: Mapping[str, Any], *, title: str = "Feed Agent Lab") -> str:
    """Return a standalone interactive trajectory viewer."""
    if payload.get("schema_version") != DEMO_SCHEMA_VERSION:
        raise ValueError("unsupported demo payload")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Ctext y='.9em' font-size='90'%3E%E2%86%97%3C/text%3E%3C/svg%3E">
  <style>
    :root {{ color-scheme: light; --ink:#10233f; --muted:#5d6f86; --line:#9fb1c5;
      --paper:#eaf1f8; --panel:#f8fbff; --navy:#101a2b; --cyan:#19a994;
      --coral:#e95f5c; --yellow:#f2c14e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink);
      font:14px/1.5 "Avenir Next","Segoe UI",sans-serif; }}
    header {{ display:grid; grid-template-columns:minmax(0,1.4fr) minmax(220px,.6fr);
      gap:28px; align-items:end; padding:34px clamp(18px,5vw,72px) 28px;
      color:#eef7ff; background:var(--navy); border-bottom:5px solid var(--cyan); }}
    h1 {{ margin:2px 0 8px; max-width:820px; font:800 clamp(30px,5vw,58px)/.98
      "Arial Narrow","Avenir Next Condensed",sans-serif; letter-spacing:-.045em; }}
    .kicker {{ color:#64deca; text-transform:uppercase; letter-spacing:.19em;
      font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .sub {{ color:#b8c6d8; max-width:760px; }}
    .thesis {{ border-left:1px solid #58708e; padding-left:18px; color:#d5e0ec;
      font:600 15px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .thesis b {{ display:block; color:var(--yellow); font-size:28px; }}
    main {{ padding:22px clamp(18px,5vw,72px) 56px; }}
    .controls {{ display:grid; grid-template-columns:1fr 1fr minmax(220px,2fr); gap:12px;
      align-items:end; margin-bottom:18px; }}
    label {{ display:grid; gap:6px; color:var(--muted); font:700 10px/1.2
      ui-monospace,SFMono-Regular,Menlo,monospace; text-transform:uppercase; letter-spacing:.1em; }}
    select,input {{ width:100%; color:var(--ink); background:var(--panel); border:1px solid var(--line);
      border-radius:2px; padding:10px; accent-color:var(--cyan); }}
    select:focus-visible,input:focus-visible,.frame:focus-visible {{ outline:3px solid var(--yellow);
      outline-offset:2px; }}
    .filmstrip {{ position:relative; display:flex; gap:7px; overflow-x:auto; margin:10px 0 0;
      padding:21px 12px; background:var(--navy); border-radius:3px; scrollbar-color:var(--cyan) var(--navy); }}
    .filmstrip::before,.filmstrip::after {{ content:""; position:absolute; left:0; right:0; height:7px;
      background:repeating-linear-gradient(90deg,transparent 0 8px,#d7e3f0 8px 17px,transparent 17px 25px); opacity:.55; }}
    .filmstrip::before {{ top:5px; }} .filmstrip::after {{ bottom:5px; }}
    .frame {{ position:relative; flex:0 0 58px; height:45px; border:1px solid #59718e;
      background:#1a2a42; color:#9fb4cb; cursor:pointer; font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .frame:hover {{ border-color:#a7c9e8; }} .frame.active {{ color:#091521; background:#64deca;
      border-color:#64deca; }} .frame.has-buy::after,.frame.has-return::after {{ content:"";
      position:absolute; width:7px; height:7px; right:4px; top:4px; border-radius:50%; background:var(--yellow); }}
    .frame.has-return::after {{ background:var(--coral); }}
    .scoreboard {{ display:flex; flex-wrap:wrap; border-block:1px solid var(--line); margin:18px 0; }}
    .metric {{ min-width:150px; flex:1; padding:11px 14px; border-right:1px solid var(--line); }}
    .metric:last-child {{ border-right:0; }} .metric b {{ display:block; font:800 20px/1.2
      ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--cyan); }}
    .metric span {{ color:var(--muted); font:700 9px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;
      text-transform:uppercase; letter-spacing:.08em; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:3px; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr); gap:14px; }}
    .panel {{ padding:16px; min-height:180px; overflow:auto; }}
    .panel h2 {{ margin:-16px -16px 14px; padding:8px 12px; color:#dce8f5; background:#263a55;
      font:700 10px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.11em; text-transform:uppercase; }}
    pre {{ white-space:pre-wrap; word-break:break-word; margin:0; color:#203754;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
    .chip {{ border:1px solid var(--line); padding:4px 8px; border-radius:999px; }}
    .event-click,.event-cart,.event-purchase,.event-retained {{ color:#087f70; }}
    .event-return,.event-skip,.event-repeat_exposure {{ color:#b83f44; }}
    .timeline {{ border-left:3px solid var(--cyan); padding-left:12px; display:grid; gap:8px;
      font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .empty {{ color:var(--muted); font-style:italic; }}
    @media(max-width:800px) {{ header,.controls,.grid {{ grid-template-columns:1fr; }}
      .scoreboard {{ grid-template-columns:1fr 1fr; }} }}
    @media(prefers-reduced-motion:reduce) {{ * {{ scroll-behavior:auto!important; }} }}
  </style>
</head>
<body>
<header><div><div class="kicker">Causal feed instrument · {DEMO_SCHEMA_VERSION}</div><h1>{safe_title}</h1>
<div class="sub">在同一条固定 Feed 上回放策略决策、延迟购买与退货，并检查奖励归因。</div></div>
<div class="thesis"><b>t → t+k</b>一次商业介入，必须为它后来留下的结果负责。</div></header>
<main>
  <section class="controls">
    <label>Policy<select id="policy"></select></label>
    <label>Episode<select id="episode"></select></label>
    <label>Feed step <span id="stepLabel"></span><input id="step" type="range" min="0" value="0"></label>
  </section>
  <nav id="filmstrip" class="filmstrip" aria-label="Feed causal filmstrip"></nav>
  <section class="scoreboard">
    <div class="metric"><b id="return">—</b><span>episode return</span></div>
    <div class="metric"><b id="position">—</b><span>feed position</span></div>
    <div class="metric"><b id="decision">—</b><span>When</span></div>
    <div class="metric"><b id="eventsCount">—</b><span>events this step</span></div>
  </section>
  <section class="grid">
    <article class="panel"><h2>Current video / public observation</h2><pre id="observation"></pre></article>
    <article class="panel"><h2>When · What · How</h2><pre id="action"></pre></article>
    <article class="panel"><h2>Delayed event timeline</h2><div id="events" class="timeline"></div></article>
    <article class="panel"><h2>Reward components (evaluator only)</h2><pre id="reward"></pre></article>
  </section>
</main>
<script id="feed-data" type="application/json">{encoded}</script>
<script>
const data=JSON.parse(document.getElementById('feed-data').textContent);
const byId=id=>document.getElementById(id), policy=byId('policy'), episode=byId('episode'), step=byId('step');
const pretty=value=>JSON.stringify(value??{{}},null,2);
const policies=[...new Set(data.trajectories.map(x=>x.policy))];
policy.innerHTML=policies.map(x=>`<option>${{x}}</option>`).join('');
function available(){{return data.trajectories.filter(x=>x.policy===policy.value)}}
function fillEpisodes(){{const rows=available();episode.innerHTML=rows.map((x,i)=>`<option value="${{i}}">${{x.episode_id}}</option>`).join('');render();}}
function render(){{const rows=available(),run=rows[Number(episode.value)||0];if(!run)return;
 const transitions=run.transitions||[],max=Math.max(0,transitions.length-1);step.max=max;step.value=Math.min(Number(step.value)||0,max);
 const i=Number(step.value),t=transitions[i]||{{}},events=t.events||t.user_events||[],reward=t.reward||t.reward_breakdown||{{}};
 byId('filmstrip').innerHTML=transitions.map((item,index)=>{{const itemEvents=item.events||item.user_events||[],types=itemEvents.map(e=>e.event_type||e.type),state=index===i?' active':'',flag=types.includes('return')?' has-return':types.includes('purchase')?' has-buy':'';return `<button class="frame${{state}}${{flag}}" data-step="${{index}}" aria-label="Go to feed step ${{index+1}}">${{String(index+1).padStart(2,'0')}}</button>`}}).join('');
 byId('filmstrip').querySelectorAll('.frame').forEach(button=>button.addEventListener('click',()=>{{step.value=button.dataset.step;render()}}));
 byId('filmstrip').querySelector('.active')?.scrollIntoView({{block:'nearest',inline:'center'}});
 byId('return').textContent=Number(run.episode_return||0).toFixed(3);byId('position').textContent=transitions.length?`${{i+1}} / ${{transitions.length}}`:'0 / 0';
 byId('stepLabel').textContent=transitions.length?`${{i+1}}/${{transitions.length}}`:'—';byId('decision').textContent=t.action?.decision||'—';
 byId('eventsCount').textContent=events.length;byId('observation').textContent=pretty(t.observation||t.pre_observation||{{}});byId('action').textContent=pretty(t.action||{{}});
 byId('reward').textContent=pretty(reward);byId('events').innerHTML=events.length?events.map(e=>`<div class="event-${{e.event_type||e.type}}"><b>${{e.event_type||e.type}}</b> · source ${{e.source_step??'—'}} → realized ${{e.step??'—'}}${{e.product_id?' · '+e.product_id:''}}</div>`).join(''):'<div class="empty">No user event</div>';
}}
policy.addEventListener('change',()=>{{step.value=0;fillEpisodes()}});episode.addEventListener('change',()=>{{step.value=0;render()}});step.addEventListener('input',render);
fillEpisodes();
</script>
</body></html>"""


def write_demo(
    logs_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_summary_path: str | Path | None = None,
    title: str = "Feed Agent Lab",
) -> Path:
    rows = list(iter_jsonl(logs_path))
    summary: Mapping[str, Any] | None = None
    if evaluation_summary_path is not None:
        loaded = json.loads(Path(evaluation_summary_path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("evaluation summary must be a JSON object")
        summary = loaded
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_demo_html(build_demo_payload(rows, evaluation_summary=summary), title=title),
        encoding="utf-8",
    )
    source_logs = Path(logs_path)
    manifest = {
        "schema_version": "feed-demo-manifest-v1",
        "demo_schema_version": DEMO_SCHEMA_VERSION,
        "inputs": {
            "logs": {
                "path": source_logs.name,
                "sha256": sha256_file(source_logs),
            },
        },
        "output": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        },
        "network_dependencies": False,
    }
    if evaluation_summary_path is not None:
        summary_path = Path(evaluation_summary_path)
        manifest["inputs"]["evaluation_summary"] = {
            "path": summary_path.name,
            "sha256": sha256_file(summary_path),
        }
    manifest["manifest_content_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    write_json_atomic(destination.with_suffix(".manifest.json"), manifest)
    return destination


__all__ = [
    "DEMO_SCHEMA_VERSION",
    "build_demo_payload",
    "render_demo_html",
    "write_demo",
]
