from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import MeshConfig
from .consolidation import BasinTracker
from .encoding import TextPhaseEncoder
from .field import PhaseFieldMesh
from .runtime import CognitiveMeshRuntime


def config_from_env() -> MeshConfig:
    size = int(os.getenv("PHASE_MESH_SIZE", "128"))
    return MeshConfig(
        width=size,
        height=size,
        max_steps=int(os.getenv("PHASE_MESH_STEPS", "320")),
        seed=int(os.getenv("PHASE_MESH_SEED", "7")),
        laplacian_backend=os.getenv("PHASE_MESH_BACKEND", "auto"),
        phase_pin_strength=float(os.getenv("PHASE_MESH_PIN", "0.25")),
        phase_residual_carry=float(os.getenv("PHASE_MESH_RESIDUAL_CARRY", "0.08")),
    )


def persistence_enabled() -> bool:
    return os.getenv("PHASE_MESH_PERSIST", "1") not in {"0", "false", "False", "no"}


def state_paths() -> tuple[Path, Path]:
    state_dir = Path(os.getenv("PHASE_MESH_STATE_DIR", "runs/service-state"))
    return state_dir / "topology.q8.npz", state_dir / "basins.json"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await load_persistent_state()
    try:
        yield
    finally:
        await save_persistent_state()


app = FastAPI(title="Phase-Field Cognitive Mesh", version="0.2.0", lifespan=lifespan)
runtime = CognitiveMeshRuntime(config_from_env())
runtime_lock = asyncio.Lock()
jobs: dict[str, dict[str, Any]] = {}
service_state_error: str | None = None


class ResonateRequest(BaseModel):
    text: str = Field(min_length=1)
    steps: int | None = Field(default=None, ge=1, le=2000)
    expected: str | None = None
    learn: bool = False


class LearnRequest(BaseModel):
    text: str = Field(min_length=1)
    expected: str | None = None
    rounds: int = Field(default=4, ge=1, le=64)
    steps: int | None = Field(default=None, ge=1, le=2000)


class ThinkRequest(BaseModel):
    text: str = Field(min_length=1)
    max_budget: int = Field(default=200, ge=1, le=5000)
    min_steps: int | None = Field(default=None, ge=0, le=5000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    expected: str | None = None
    learn: bool = False
    verifier_control: bool = False
    stream_interval: int = Field(default=4, ge=1, le=128)


class JobResponse(BaseModel):
    job_id: str
    status: str
    poll: str


async def load_persistent_state() -> None:
    global service_state_error
    if not persistence_enabled():
        return
    topology_path, basins_path = state_paths()
    try:
        if topology_path.exists():
            mesh = PhaseFieldMesh.load_quantized(topology_path)
            install_mesh(mesh)
        if basins_path.exists():
            runtime.basin_tracker = BasinTracker.load(basins_path)
    except Exception as exc:  # pragma: no cover - service safety net
        service_state_error = str(exc)


async def save_persistent_state() -> None:
    if persistence_enabled():
        save_state()


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)


@app.get("/demo", response_class=HTMLResponse)
def demo() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    topology_path, basins_path = state_paths()
    return {
        "ok": service_state_error is None,
        "state_error": service_state_error,
        "step": runtime.mesh.step_index,
        "shape": list(runtime.mesh.config.shape),
        "backend": runtime.mesh.config.laplacian_backend,
        "pin_strength": runtime.mesh.config.phase_pin_strength,
        "state": {
            "enabled": persistence_enabled(),
            "topology_path": str(topology_path),
            "topology_exists": topology_path.exists(),
            "basins_path": str(basins_path),
            "basins_exists": basins_path.exists(),
        },
    }


@app.get("/state")
def state() -> dict[str, Any]:
    topology_path, basins_path = state_paths()
    return {
        "mesh": runtime.mesh.metrics().to_dict(),
        "basins": runtime.discover_basins(),
        "paths": {
            "topology": str(topology_path),
            "basins": str(basins_path),
        },
    }


@app.post("/state/save")
def save_state_endpoint() -> dict[str, Any]:
    return save_state()


@app.get("/comparison/latest")
def latest_comparison() -> dict[str, Any]:
    path = Path(os.getenv("PHASE_MESH_COMPARISON_PATH", "runs/frontier-compare/results.json"))
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}
    return {"status": "ok", "path": str(path), "payload": payload}


@app.post("/resonate")
async def resonate(request: ResonateRequest) -> dict[str, Any]:
    async with runtime_lock:
        run = await asyncio.to_thread(
            runtime.resonate,
            request.text,
            steps=request.steps,
            expected=request.expected,
            learn=request.learn,
        )
        if request.learn and persistence_enabled():
            save_state()
    return run.to_dict()


@app.post("/learn")
async def learn(request: LearnRequest) -> dict[str, Any]:
    async with runtime_lock:
        result = await asyncio.to_thread(
            runtime.learn,
            request.text,
            expected=request.expected,
            rounds=request.rounds,
            steps=request.steps,
        )
        if persistence_enabled():
            save_state()
        return result


@app.get("/think")
async def think_query(
    text: str,
    max_budget: int = 200,
    min_steps: int | None = None,
    temperature: float = 0.0,
    expected: str | None = None,
    learn: bool = False,
    verifier_control: bool = False,
) -> dict[str, Any]:
    request = ThinkRequest(
        text=text,
        max_budget=max_budget,
        min_steps=min_steps,
        temperature=temperature,
        expected=expected,
        learn=learn,
        verifier_control=verifier_control,
    )
    return await think(request)


@app.post("/think")
async def think(request: ThinkRequest) -> dict[str, Any]:
    async with runtime_lock:
        run = await asyncio.to_thread(
            runtime.think,
            request.text,
            max_budget=request.max_budget,
            min_steps=request.min_steps,
            temperature=request.temperature,
            expected=request.expected,
            learn=request.learn,
            verifier_control=request.verifier_control,
        )
        if request.learn and persistence_enabled():
            save_state()
    return run.to_dict()


@app.get("/think/stream")
async def think_stream_query(
    text: str,
    max_budget: int = 200,
    min_steps: int | None = None,
    temperature: float = 0.0,
    expected: str | None = None,
    learn: bool = False,
    verifier_control: bool = False,
    stream_interval: int = 4,
) -> StreamingResponse:
    request = ThinkRequest(
        text=text,
        max_budget=max_budget,
        min_steps=min_steps,
        temperature=temperature,
        expected=expected,
        learn=learn,
        verifier_control=verifier_control,
        stream_interval=stream_interval,
    )
    return await think_stream(request)


@app.post("/think/stream")
async def think_stream(request: ThinkRequest) -> StreamingResponse:
    async with runtime_lock:
        local_runtime = clone_runtime()
    return StreamingResponse(
        stream_runtime_events(local_runtime, request),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.get("/basins")
def basins() -> dict[str, Any]:
    return runtime.discover_basins()


@app.post("/jobs/resonate")
async def queue_resonate(request: ResonateRequest) -> JobResponse:
    job_id = create_job("resonate", request.model_dump())
    asyncio.create_task(resolve_resonate_job(job_id, request))
    return JobResponse(job_id=job_id, status="queued", poll=f"/jobs/{job_id}")


@app.post("/jobs/learn")
async def queue_learn(request: LearnRequest) -> JobResponse:
    job_id = create_job("learn", request.model_dump())
    asyncio.create_task(resolve_learn_job(job_id, request))
    return JobResponse(job_id=job_id, status="queued", poll=f"/jobs/{job_id}")


@app.post("/jobs/think")
async def queue_think(request: ThinkRequest) -> JobResponse:
    job_id = create_job("think", request.model_dump())
    asyncio.create_task(resolve_think_job(job_id, request))
    return JobResponse(job_id=job_id, status="queued", poll=f"/jobs/{job_id}")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return jobs.get(
        job_id,
        {
            "job_id": job_id,
            "status": "missing",
            "created_at": None,
            "updated_at": time.time(),
        },
    )


def create_job(kind: str, request: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    jobs[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "request": request,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    return job_id


async def resolve_resonate_job(job_id: str, request: ResonateRequest) -> None:
    jobs[job_id]["status"] = "running"
    jobs[job_id]["updated_at"] = time.time()
    try:
        async with runtime_lock:
            run = await asyncio.to_thread(
                runtime.resonate,
                request.text,
                steps=request.steps,
                expected=request.expected,
                learn=request.learn,
            )
            if request.learn and persistence_enabled():
                save_state()
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = run.to_dict()
    except Exception as exc:  # pragma: no cover - service safety net
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
    finally:
        jobs[job_id]["updated_at"] = time.time()


async def resolve_learn_job(job_id: str, request: LearnRequest) -> None:
    jobs[job_id]["status"] = "running"
    jobs[job_id]["updated_at"] = time.time()
    try:
        async with runtime_lock:
            result = await asyncio.to_thread(
                runtime.learn,
                request.text,
                expected=request.expected,
                rounds=request.rounds,
                steps=request.steps,
            )
            if persistence_enabled():
                save_state()
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as exc:  # pragma: no cover - service safety net
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
    finally:
        jobs[job_id]["updated_at"] = time.time()


async def resolve_think_job(job_id: str, request: ThinkRequest) -> None:
    jobs[job_id]["status"] = "running"
    jobs[job_id]["updated_at"] = time.time()
    try:
        async with runtime_lock:
            run = await asyncio.to_thread(
                runtime.think,
                request.text,
                max_budget=request.max_budget,
                min_steps=request.min_steps,
                temperature=request.temperature,
                expected=request.expected,
                learn=request.learn,
                verifier_control=request.verifier_control,
            )
            if request.learn and persistence_enabled():
                save_state()
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = run.to_dict()
    except Exception as exc:  # pragma: no cover - service safety net
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
    finally:
        jobs[job_id]["updated_at"] = time.time()


def stream_runtime_events(local_runtime: CognitiveMeshRuntime, request: ThinkRequest) -> Iterator[str]:
    yield encode_sse({"event": "server", "time": time.time(), "status": "stream-open"})
    try:
        for payload in local_runtime.think_stream(
            request.text,
            max_budget=request.max_budget,
            min_steps=request.min_steps,
            temperature=request.temperature,
            expected=request.expected,
            learn=request.learn,
            verifier_control=request.verifier_control,
            stream_interval=request.stream_interval,
        ):
            yield encode_sse(payload)
    except Exception as exc:
        yield encode_sse({"event": "error", "message": str(exc)})


def encode_sse(payload: dict[str, Any]) -> str:
    event = str(payload.get("event", "message"))
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def save_state() -> dict[str, Any]:
    topology_path, basins_path = state_paths()
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.mesh.save_quantized(topology_path)
    runtime.basin_tracker.save(basins_path)
    return {
        "status": "saved",
        "topology_path": str(topology_path),
        "topology_bytes": topology_path.stat().st_size,
        "basins_path": str(basins_path),
        "basins_bytes": basins_path.stat().st_size,
    }


def install_mesh(mesh: PhaseFieldMesh) -> None:
    runtime.mesh = mesh
    runtime.config = mesh.config
    runtime.encoder = TextPhaseEncoder(mesh.config.width, mesh.config.height)


def clone_runtime() -> CognitiveMeshRuntime:
    local = CognitiveMeshRuntime(runtime.config)
    local.mesh.theta = runtime.mesh.theta.copy()
    local.mesh.velocity = runtime.mesh.velocity.copy()
    local.mesh.omega = runtime.mesh.omega.copy()
    local.mesh.landscape = runtime.mesh.landscape.copy()
    local.mesh.predictor_trace = runtime.mesh.predictor_trace.copy()
    local.mesh.pin_phase = runtime.mesh.pin_phase.copy()
    local.mesh.pin_weights = runtime.mesh.pin_weights.copy()
    local.mesh.step_index = runtime.mesh.step_index
    local.mesh._last_coherence = runtime.mesh._last_coherence
    local.basin_tracker.basins = list(runtime.basin_tracker.basins)
    return local


DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase Mesh Demo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d10;
      --panel: #15191f;
      --panel2: #10141a;
      --line: #2b343f;
      --text: #e8eef5;
      --muted: #91a0ad;
      --mint: #67e8b9;
      --amber: #f0b35a;
      --rose: #f4728c;
      --cyan: #69c6f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main { min-height: 100vh; display: grid; grid-template-columns: 360px 1fr; }
    aside {
      border-right: 1px solid var(--line);
      background: #0e1217;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .brand { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    h1 { font-size: 18px; margin: 0; font-weight: 700; }
    .status { font-size: 12px; color: var(--muted); }
    textarea {
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      background: #0a0d11;
      color: var(--text);
      border-radius: 8px;
      padding: 12px;
      font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.45;
    }
    button, select, input {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 8px;
      min-height: 36px;
    }
    button { cursor: pointer; padding: 0 12px; font-weight: 650; }
    button.primary { background: #12382d; border-color: #236552; color: #d9fff1; }
    button.warn { background: #3b2615; border-color: #775128; color: #ffe0b3; }
    button:disabled { opacity: .55; cursor: wait; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .field { display: grid; gap: 6px; }
    label { color: var(--muted); font-size: 12px; }
    .settings { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .content { padding: 18px; display: grid; grid-template-rows: auto 1fr; gap: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
      min-height: 74px;
    }
    .metric span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 8px; }
    .metric strong { display: block; font-size: 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .grid { display: grid; grid-template-columns: minmax(320px, 1.1fr) minmax(320px, .9fr); gap: 16px; min-height: 0; }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel2);
      border-radius: 8px;
      padding: 12px;
      min-height: 0;
      overflow: hidden;
    }
    .panel h2 { margin: 0 0 10px; font-size: 14px; }
    canvas {
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #050608;
      border: 1px solid var(--line);
      border-radius: 8px;
      image-rendering: pixelated;
    }
    .events {
      height: 100%;
      min-height: 420px;
      overflow: auto;
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.45;
      color: #c9d4df;
      white-space: pre-wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .bars { display: grid; gap: 8px; margin-top: 10px; }
    .bar { height: 12px; background: #0a0d11; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
    .bar > i { display: block; height: 100%; width: 0%; background: var(--mint); }
    .bar.gradient > i { background: var(--amber); }
    .bar.error > i { background: var(--rose); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <div class="brand">
        <h1>Phase Mesh</h1>
        <span class="pill" id="health">booting</span>
      </div>
      <textarea id="prompt">check 17 * 19 = 323
ctx_000 ctx_001 ctx_002 ctx_003 ctx_004 ctx_005 ctx_006 ctx_007
route the correct basin and preserve ctx_005</textarea>
      <div class="settings">
        <div class="field"><label for="budget">Budget</label><input id="budget" type="number" value="180" min="1" max="5000"></div>
        <div class="field"><label for="temp">Temperature</label><input id="temp" type="number" value="0.25" min="0" max="2" step="0.05"></div>
        <div class="field"><label for="interval">Stream Every</label><input id="interval" type="number" value="4" min="1" max="128"></div>
        <div class="field"><label for="expected">Expected</label><input id="expected" value=""></div>
      </div>
      <div class="row">
        <button class="primary" id="run">Run</button>
        <button id="stop">Stop</button>
      </div>
      <div class="row">
        <button class="warn" data-preset="context">8K Context</button>
        <button data-preset="math">Hard Math</button>
      </div>
      <div class="row">
        <button data-preset="code">Code Route</button>
        <button id="save">Save State</button>
      </div>
      <div class="status" id="stateLine">state pending</div>
    </aside>
    <section class="content">
      <div class="metrics">
        <div class="metric"><span>Route</span><strong id="route">-</strong></div>
        <div class="metric"><span>Steps</span><strong id="steps">0</strong></div>
        <div class="metric"><span>Coherence</span><strong id="coherence">0.000</strong></div>
        <div class="metric"><span>Gradient</span><strong id="gradient">0.000</strong></div>
        <div class="metric"><span>Error</span><strong id="error">0.000</strong></div>
        <div class="metric"><span>Latency</span><strong id="latency">0 ms</strong></div>
      </div>
      <div class="grid">
        <div class="panel">
          <h2>Phase Field</h2>
          <canvas id="phase" width="320" height="320"></canvas>
          <div class="bars">
            <div class="bar coherence"><i id="cohBar"></i></div>
            <div class="bar gradient"><i id="gradBar"></i></div>
            <div class="bar error"><i id="errBar"></i></div>
          </div>
        </div>
        <div class="panel">
          <h2>Stream</h2>
          <div class="events" id="events"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let controller = null;
    let started = 0;

    async function refreshHealth() {
      const res = await fetch('/health');
      const data = await res.json();
      $('health').textContent = data.ok ? `${data.shape[0]}x${data.shape[1]} ${data.backend}` : 'state error';
      $('stateLine').textContent = data.state.enabled ? `state ${data.state.topology_exists ? 'loaded' : 'new'}` : 'state disabled';
    }

    function logLine(text) {
      const box = $('events');
      box.textContent += text + '\n';
      box.scrollTop = box.scrollHeight;
    }

    function setMetric(id, value) { $(id).textContent = value; }
    function pct(value, max = 1) { return `${Math.max(0, Math.min(100, value / max * 100))}%`; }

    function drawPhase(sample) {
      if (!sample || !sample.values) return;
      const canvas = $('phase');
      const ctx = canvas.getContext('2d');
      const image = ctx.createImageData(sample.width, sample.height);
      for (let i = 0; i < sample.values.length; i++) {
        const value = sample.values[i];
        image.data[i * 4 + 0] = value < 128 ? 40 : value;
        image.data[i * 4 + 1] = value;
        image.data[i * 4 + 2] = 255 - value;
        image.data[i * 4 + 3] = 255;
      }
      const tmp = document.createElement('canvas');
      tmp.width = sample.width;
      tmp.height = sample.height;
      tmp.getContext('2d').putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }

    function handleEvent(payload) {
      if (payload.event === 'step') {
        const step = payload.step;
        const metrics = step.metrics;
        setMetric('steps', `${step.steps_used}/${$('budget').value}`);
        setMetric('coherence', metrics.coherence.toFixed(3));
        setMetric('gradient', metrics.gradient.toFixed(3));
        setMetric('error', step.prediction_error.toFixed(3));
        setMetric('route', payload.decoded.route);
        $('cohBar').style.width = pct(metrics.coherence);
        $('gradBar').style.width = pct(metrics.gradient, 0.18);
        $('errBar').style.width = pct(step.prediction_error, 0.1);
        drawPhase(payload.phase_sample);
        logLine(`step ${step.steps_used} route=${payload.decoded.route} coh=${metrics.coherence.toFixed(3)} grad=${metrics.gradient.toFixed(3)} err=${step.prediction_error.toFixed(3)}`);
      }
      if (payload.event === 'verifier') {
        logLine(`verifier ${payload.verifier.passed ? 'pass' : 'fail'} ${payload.verifier.checker}`);
      }
      if (payload.event === 'basin') {
        logLine(`basin ${payload.basin.basin_id} route=${payload.basin.route} persistence=${payload.basin.persistence.toFixed(3)}`);
      }
      if (payload.event === 'final') {
        const run = payload.run;
        drawPhase(payload.phase_sample);
        setMetric('latency', `${Math.round(performance.now() - started)} ms`);
        setMetric('route', run.decoded.route);
        setMetric('steps', `${run.steps_used}/${run.max_budget}`);
        logLine(`final passed=${run.verifier.passed} steps=${run.steps_used} signature=${run.decoded.signature}`);
        $('run').disabled = false;
      }
      if (payload.event === 'error') {
        logLine(`error ${payload.message}`);
        $('run').disabled = false;
      }
    }

    function parseSseChunk(text, state) {
      state.buffer += text;
      const parts = state.buffer.split('\n\n');
      state.buffer = parts.pop();
      for (const part of parts) {
        const line = part.split('\n').find((item) => item.startsWith('data: '));
        if (!line) continue;
        handleEvent(JSON.parse(line.slice(6)));
      }
    }

    async function runStream() {
      if (controller) controller.abort();
      controller = new AbortController();
      started = performance.now();
      $('events').textContent = '';
      $('run').disabled = true;
      const body = {
        text: $('prompt').value,
        max_budget: Number($('budget').value),
        temperature: Number($('temp').value),
        expected: $('expected').value || null,
        verifier_control: true,
        stream_interval: Number($('interval').value)
      };
      const res = await fetch('/think/stream', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify(body),
        signal: controller.signal
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const state = {buffer: ''};
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        parseSseChunk(decoder.decode(value, {stream: true}), state);
      }
      $('run').disabled = false;
    }

    function preset(kind) {
      if (kind === 'context') {
        const tokens = Array.from({length: 8192}, (_, i) => `ctx_${String(i).padStart(6, '0')}`);
        $('prompt').value = `${tokens.join(' ')}\nQuestion: preserve ctx_000005`;
        $('budget').value = 180;
      }
      if (kind === 'math') {
        $('prompt').value = 'Calculate: 97 * 89\nReturn only the final number.';
        $('expected').value = '8633';
        $('budget').value = 240;
      }
      if (kind === 'code') {
        $('prompt').value = 'def add(a, b):\n    return a + b\nroute this snippet to the right tool and verify it';
        $('expected').value = '';
        $('budget').value = 120;
      }
    }

    $('run').addEventListener('click', () => runStream().catch((err) => { logLine(`error ${err.message}`); $('run').disabled = false; }));
    $('stop').addEventListener('click', () => { if (controller) controller.abort(); $('run').disabled = false; });
    $('save').addEventListener('click', async () => {
      const res = await fetch('/state/save', {method: 'POST'});
      const data = await res.json();
      $('stateLine').textContent = `saved ${data.topology_bytes} bytes`;
    });
    document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => preset(button.dataset.preset)));
    refreshHealth();
  </script>
</body>
</html>
"""
