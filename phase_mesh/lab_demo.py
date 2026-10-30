from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .config import MeshConfig
from .registry import PhaseMeshRegistry, render_domain_report
from .runtime import CognitiveMeshRuntime


DEFAULT_SOLVES = (
    ("arithmetic", "8 plus 9"),
    ("code", "def add(a, b): return a + b"),
    ("json", '{"ok": true}'),
    ("memory", "recall registry"),
)


def build_lab_demo(
    *,
    out_dir: str | Path = "runs/lab-demo",
    registry_dir: str | Path | None = None,
    context_tokens: Sequence[int] = (512, 2048, 8192),
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "numpy",
    pin_strength: float = 0.25,
    residual_carry: float = 0.08,
    fit_registry: bool = True,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    registry_path = Path(registry_dir) if registry_dir is not None else out_path / "registry"
    paths = {
        "summary_json": str(out_path / "summary.json"),
        "index_html": str(out_path / "index.html"),
        "summary_md": str(out_path / "summary.md"),
    }

    started = time.perf_counter()
    if fit_registry or not (registry_path / "manifest.json").exists():
        registry = PhaseMeshRegistry()
        fit = registry.fit(registry_path)
    else:
        registry = PhaseMeshRegistry.load(registry_path)
        fit = _read_json(registry_path / "manifest.json")

    registry = PhaseMeshRegistry.load(registry_path)
    probe = registry.probe()
    solves = [
        {
            "label": label,
            "prompt": prompt,
            "result": registry.solve(prompt),
        }
        for label, prompt in DEFAULT_SOLVES
    ]
    context = run_context_sweep(
        token_counts=context_tokens,
        size=size,
        steps=steps,
        seed=seed,
        backend=backend,
        pin_strength=pin_strength,
        residual_carry=residual_carry,
    )
    context_control = run_context_sweep(
        token_counts=context_tokens,
        size=size,
        steps=steps,
        seed=seed,
        backend=backend,
        pin_strength=0.0,
        residual_carry=residual_carry,
    )
    controls = {
        "context_pin_on": context,
        "context_pin_off": context_control,
        "mean_pin_on_gradient": mean_gradient(context),
        "mean_pin_off_gradient": mean_gradient(context_control),
        "gradient_reduction": gradient_reduction(context_control, context),
        "separation_passed": all(row["passed"] for row in context) and any(not row["passed"] for row in context_control),
    }
    artifact_sizes = {
        "registry_bytes": directory_size(registry_path),
        "out_dir_bytes": directory_size(out_path),
    }
    payload = {
        "type": "phase-mesh-lab-demo",
        "version": 1,
        "status": "pass" if probe.get("passed") and controls["separation_passed"] else "red",
        "elapsed_s": time.perf_counter() - started,
        "config": {
            "registry_dir": str(registry_path),
            "size": int(size),
            "steps": int(steps),
            "seed": int(seed),
            "backend": str(backend),
            "pin_strength": float(pin_strength),
            "residual_carry": float(residual_carry),
            "context_tokens": [int(item) for item in context_tokens],
        },
        "fit": fit,
        "probe": probe,
        "solves": solves,
        "context_sweep": context,
        "controls": controls,
        "artifact_sizes": artifact_sizes,
        "paths": paths,
        "claims": [
            "Domain gates are measured and reproducible from this artifact.",
            "Pinning ablation compares the same synthetic context rows with phase pinning on and off.",
            "Synthetic context rows report phase-gradient retention, not natural-language recall.",
            "Arithmetic solves decode operation and operands, then use an exact resolver.",
            "This is a PhaseMesh substrate demo, not a general LLM claim.",
        ],
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "index.html").write_text(render_lab_demo_html(payload), encoding="utf-8")
    (out_path / "summary.md").write_text(render_domain_report(probe, artifact_dir=registry_path), encoding="utf-8")
    return payload


def run_context_sweep(
    *,
    token_counts: Sequence[int],
    size: int,
    steps: int,
    seed: int,
    backend: str,
    pin_strength: float,
    residual_carry: float,
) -> list[dict[str, Any]]:
    rows = []
    for token_count in token_counts:
        runtime = CognitiveMeshRuntime(
            MeshConfig(
                width=int(size),
                height=int(size),
                max_steps=int(steps),
                seed=int(seed),
                laplacian_backend=str(backend),
                phase_pin_strength=float(pin_strength),
                phase_residual_carry=float(residual_carry),
            )
        )
        tokens = [f"ctx_{index:06d}" for index in range(max(1, int(token_count)))]
        target_index = min(5, len(tokens) - 1)
        target = tokens[target_index]
        prompt = " ".join(tokens) + f"\nQuestion: stabilize around early token {target}"
        start = time.perf_counter()
        run = runtime.resonate(prompt)
        elapsed = time.perf_counter() - start
        rows.append({
            "token_count": int(token_count),
            "target_index": int(target_index),
            "target_token": target,
            "gradient": float(run.metrics.gradient),
            "coherence": float(run.metrics.coherence),
            "energy": float(run.metrics.energy),
            "steps_used": int(len(run.history)),
            "elapsed_s": float(elapsed),
            "passed": bool(run.metrics.gradient < 0.05),
        })
    return rows


def render_lab_demo_html(payload: dict[str, Any]) -> str:
    probe = payload.get("probe", {})
    domains = probe.get("domains", {})
    context = payload.get("context_sweep", [])
    controls = payload.get("controls", {})
    solves = payload.get("solves", [])
    status = str(payload.get("status", "unknown")).upper()
    config = payload.get("config", {})
    data_json = json.dumps(payload, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhaseMesh Lab Demo</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080a0d;
      --panel: #11161b;
      --panel2: #161d24;
      --line: #2c3640;
      --text: #eef4f8;
      --muted: #96a5b2;
      --green: #63e6a8;
      --blue: #7cc7ff;
      --amber: #ffd166;
      --red: #ff6b80;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 32px; line-height: 1.1; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ color: var(--muted); margin: 8px 0 0; line-height: 1.45; }}
    .badge {{ border: 1px solid var(--line); background: var(--panel2); border-radius: 8px; padding: 10px 12px; font-weight: 750; color: var(--green); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .card, section {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 14px; }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .card strong {{ display: block; font-size: 24px; overflow-wrap: anywhere; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; margin-top: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #d7e7f5; }}
    .pass {{ color: var(--green); font-weight: 750; }}
    .fail {{ color: var(--red); font-weight: 750; }}
    .chart {{ height: 220px; display: grid; align-items: end; grid-auto-flow: column; gap: 12px; border: 1px solid var(--line); background: #0b0f13; border-radius: 8px; padding: 12px; }}
    .bar {{ display: grid; grid-template-rows: 1fr auto; gap: 8px; height: 100%; }}
    .bar i {{ display: block; align-self: end; min-height: 2px; border-radius: 6px 6px 0 0; background: linear-gradient(180deg, var(--blue), var(--green)); }}
    .bar small {{ color: var(--muted); text-align: center; font-size: 11px; }}
    .solves {{ display: grid; gap: 10px; }}
    .solve {{ background: #0d1217; border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .solve pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 8px 0 0; color: #dbe7ef; font-size: 12px; }}
    .notes {{ margin-top: 14px; }}
    .notes li {{ color: var(--muted); margin: 6px 0; }}
    @media (max-width: 900px) {{
      header, .grid, .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>PhaseMesh Lab Demo</h1>
        <p>A reproducible domain-gate artifact: fit, probe, solve, context-gradient sweep, and artifact size in one local run.</p>
      </div>
      <div class="badge">{escape(status)}</div>
    </header>
    <div class="cards">
      <div class="card"><span>Domains Passing</span><strong>{domain_pass_count(domains)}/{len(domains)}</strong></div>
      <div class="card"><span>Context Rows Passing</span><strong>{context_pass_count(context)}/{len(context)}</strong></div>
      <div class="card"><span>Pinning Reduction</span><strong>{float(controls.get("gradient_reduction", 0.0)):.1f}x</strong></div>
      <div class="card"><span>Registry Size</span><strong>{format_bytes(payload.get("artifact_sizes", {}).get("registry_bytes", 0))}</strong></div>
    </div>
    <div class="grid">
      <section>
        <h2>Domain Gates</h2>
        {render_domain_table(domains)}
      </section>
      <section>
        <h2>Pinning Ablation</h2>
        {render_context_ablation_table(controls)}
      </section>
    </div>
    <div class="grid">
      <section>
        <h2>Live Solves</h2>
        <div class="solves">{render_solves(solves)}</div>
      </section>
      <section>
        <h2>Run Config</h2>
        {render_config_table(config)}
        <ul class="notes">{render_claims(payload.get("claims", []))}</ul>
      </section>
    </div>
    <script type="application/json" id="phase-mesh-data">{escape(data_json)}</script>
  </main>
</body>
</html>
"""


def render_domain_table(domains: dict[str, Any]) -> str:
    rows = []
    for name, result in sorted(domains.items()):
        metrics = result.get("metrics", {})
        rows.append(
            "<tr>"
            f"<td><code>{escape(name)}</code></td>"
            f"<td>{status_class(bool(result.get('passed')))}</td>"
            f"<td>{escape(short_metrics(metrics))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Domain</th><th>Gate</th><th>Key metrics</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_context_chart(rows: list[dict[str, Any]]) -> str:
    max_gradient = max([float(row.get("gradient", 0.0)) for row in rows] + [0.05])
    bars = []
    for row in rows:
        gradient = float(row.get("gradient", 0.0))
        height = max(2.0, min(100.0, gradient / max_gradient * 100.0))
        bars.append(
            "<div class=\"bar\">"
            f"<i style=\"height:{height:.1f}%\"></i>"
            f"<small>{int(row.get('token_count', 0))}</small>"
            "</div>"
        )
    return "<div class=\"chart\">" + "".join(bars) + "</div>"


def render_context_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{int(row.get('token_count', 0))}</td>"
            f"<td>{float(row.get('gradient', 0.0)):.4f}</td>"
            f"<td>{float(row.get('coherence', 0.0)):.4f}</td>"
            f"<td>{float(row.get('elapsed_s', 0.0)):.3f}s</td>"
            f"<td>{status_class(bool(row.get('passed')))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Tokens</th><th>Gradient</th><th>Coherence</th><th>Latency</th><th>Gate</th></tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"


def render_context_ablation_table(controls: dict[str, Any]) -> str:
    on_rows = {int(row.get("token_count", 0)): row for row in controls.get("context_pin_on", [])}
    off_rows = {int(row.get("token_count", 0)): row for row in controls.get("context_pin_off", [])}
    table_rows = []
    for token_count in sorted(set(on_rows) | set(off_rows)):
        on = on_rows.get(token_count, {})
        off = off_rows.get(token_count, {})
        on_gradient = float(on.get("gradient", 0.0))
        off_gradient = float(off.get("gradient", 0.0))
        ratio = off_gradient / on_gradient if on_gradient > 0.0 else 0.0
        table_rows.append(
            "<tr>"
            f"<td>{token_count}</td>"
            f"<td>{off_gradient:.4f}</td>"
            f"<td>{on_gradient:.4f}</td>"
            f"<td>{ratio:.1f}x</td>"
            f"<td>{status_class(bool(on.get('passed')) and not bool(off.get('passed')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Tokens</th><th>Pin Off Gradient</th><th>Pin On Gradient</th>"
        "<th>Reduction</th><th>Control</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )


def render_solves(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for row in rows:
        result = row.get("result", {})
        data = result.get("result", {}).get("data", {})
        decoded = data.get("factor_readout") or data.get("decoded") or {}
        blocks.append(
            "<div class=\"solve\">"
            f"<strong>{escape(row.get('label', 'solve'))}: {escape(result.get('answer', ''))}</strong>"
            f"<pre>{escape(row.get('prompt', ''))}</pre>"
            f"<pre>{escape(json.dumps(decoded, indent=2, sort_keys=True))}</pre>"
            "</div>"
        )
    return "".join(blocks)


def render_config_table(config: dict[str, Any]) -> str:
    rows = []
    for key, value in sorted(config.items()):
        rows.append(f"<tr><td><code>{escape(key)}</code></td><td>{escape(json.dumps(value))}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def render_claims(claims: list[str]) -> str:
    return "".join(f"<li>{escape(item)}</li>" for item in claims)


def short_metrics(metrics: dict[str, Any]) -> str:
    preferred = [
        "factorized_result_accuracy",
        "direct_result_accuracy",
        "factor_mean_accuracy",
        "factor_min_accuracy",
        "exact_json_accuracy",
        "exact_ast_accuracy",
        "exact_recall",
        "accuracy",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            parts.append(f"{key}={format_metric(metrics[key])}")
    return ", ".join(parts[:5])


def status_class(passed: bool) -> str:
    label = "pass" if passed else "fail"
    return f"<span class=\"{label}\">{label}</span>"


def domain_pass_count(domains: dict[str, Any]) -> int:
    return sum(1 for result in domains.values() if result.get("passed"))


def context_pass_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("passed"))


def mean_gradient(rows: list[dict[str, Any]]) -> float:
    gradients = [float(row.get("gradient", 0.0)) for row in rows]
    return sum(gradients) / len(gradients) if gradients else 0.0


def gradient_reduction(off_rows: list[dict[str, Any]], on_rows: list[dict[str, Any]]) -> float:
    on_gradient = mean_gradient(on_rows)
    off_gradient = mean_gradient(off_rows)
    return off_gradient / on_gradient if on_gradient > 0.0 else 0.0


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    units = ("B", "KB", "MB", "GB")
    index = 0
    while amount >= 1024.0 and index < len(units) - 1:
        amount /= 1024.0
        index += 1
    if index == 0:
        return f"{int(amount)} {units[index]}"
    return f"{amount:.1f} {units[index]}"


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
