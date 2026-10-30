from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .domains import ArithmeticDomain, CodeDomain, DomainAdapter, JsonDomain, MemoryDomain, ToolDomain


DOMAIN_ORDER = ("arithmetic", "code", "json", "memory", "tool")


class PhaseMeshRegistry:
    """Registry and orchestration layer for verified PhaseMesh domains."""

    def __init__(self, domains: dict[str, DomainAdapter] | None = None, artifact_dir: str | Path | None = None) -> None:
        self.domains = domains or self.default_domains()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @staticmethod
    def default_domains() -> dict[str, DomainAdapter]:
        return {
            "arithmetic": ArithmeticDomain(),
            "code": CodeDomain(),
            "json": JsonDomain(),
            "memory": MemoryDomain(),
            "tool": ToolDomain(),
        }

    @classmethod
    def load(cls, artifact_dir: str | Path) -> PhaseMeshRegistry:
        path = Path(artifact_dir)
        domains: dict[str, DomainAdapter] = {
            "arithmetic": ArithmeticDomain.load(path / "arithmetic") if (path / "arithmetic" / "readout.json").exists() else ArithmeticDomain(),
            "code": CodeDomain.load(path / "code") if (path / "code" / "readout.json").exists() else CodeDomain(),
            "json": JsonDomain.load(path / "json") if (path / "json" / "readout.json").exists() else JsonDomain(),
            "memory": MemoryDomain.load(path / "memory"),
            "tool": ToolDomain(),
        }
        return cls(domains=domains, artifact_dir=path)

    def fit(self, out_dir: str | Path, domains: Iterable[str] | None = None) -> dict[str, Any]:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        selected = self._selected(domains)
        results = {}
        for name in selected:
            results[name] = self.domains[name].fit(path / name).to_dict()
        manifest = {
            "type": "phase-mesh-domain-registry",
            "version": 1,
            "domains": {
                name: self.domains[name].manifest()
                for name in selected
            },
            "fit": results,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self.artifact_dir = path
        return manifest

    def probe(self, domains: Iterable[str] | None = None) -> dict[str, Any]:
        selected = self._selected(domains)
        results = {
            name: self.domains[name].probe().to_dict()
            for name in selected
        }
        return {
            "status": "ok",
            "passed": all(bool(result["passed"]) for result in results.values()),
            "domains": results,
        }

    def solve(self, text: str, domain: str = "auto") -> dict[str, Any]:
        selected = self._route(text, domain)
        if selected not in self.domains:
            route = self.domains["tool"].solve(text).to_dict()
            return {
                "status": "routed",
                "domain": selected,
                "answer": route.get("answer", ""),
                "result": route,
                "note": "No domain adapter is registered for this route yet.",
            }
        result = self.domains[selected].solve(text).to_dict()
        return {
            "status": result["status"],
            "domain": selected,
            "answer": result.get("answer", ""),
            "result": result,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "type": "phase-mesh-domain-registry",
            "domains": {
                name: adapter.manifest()
                for name, adapter in sorted(self.domains.items())
            },
        }

    def _route(self, text: str, domain: str) -> str:
        if domain != "auto":
            return domain
        routed = self.domains["tool"].solve(text)
        target = str(routed.data.get("domain", "tool"))
        if target in self.domains:
            return target
        return "tool"

    def _selected(self, domains: Iterable[str] | None) -> list[str]:
        if domains is None:
            return [name for name in DOMAIN_ORDER if name in self.domains]
        selected = []
        for item in domains:
            name = str(item).strip()
            if not name or name == "all":
                continue
            if name not in self.domains:
                raise ValueError(f"unknown domain: {name}")
            selected.append(name)
        return selected or [name for name in DOMAIN_ORDER if name in self.domains]


def render_domain_report(payload: dict[str, Any], *, artifact_dir: str | Path | None = None) -> str:
    """Render a concise auditable report for a domain probe payload."""

    status = "PASS" if payload.get("passed") else "FAIL"
    lines = [
        "# PhaseMesh Domain Report",
        "",
        f"Status: **{status}**",
    ]
    if artifact_dir is not None:
        lines.append(f"Artifact: `{artifact_dir}`")
    lines.extend([
        "",
        "| Domain | Gate | Key Metrics |",
        "| --- | --- | --- |",
    ])
    for name, result in sorted(payload.get("domains", {}).items()):
        gate = "pass" if result.get("passed") else "fail"
        metrics = result.get("metrics", {})
        lines.append(f"| {name} | {gate} | {_format_metrics(metrics)} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Arithmetic uses decoded operation/operand factors plus an exact arithmetic resolver; the direct result probe is kept as a control.",
        "- Code uses exact Python AST facts plus an AST-derived factor readout; it is not code generation.",
        "- JSON uses exact parser facts plus a structural factor readout; it is not schema induction.",
        "- Memory and tool routing are exact starter domains until streaming traces justify learned gates.",
        "- Treat this report as a domain-gate snapshot, not as evidence of a general LLM.",
        "",
    ])
    return "\n".join(lines)


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    preferred = [
        "factorized_result_accuracy",
        "direct_result_accuracy",
        "operation_accuracy",
        "left_accuracy",
        "right_accuracy",
        "factor_mean_accuracy",
        "factor_min_accuracy",
        "passed_factor_gate",
        "exact_ast_accuracy",
        "exact_json_accuracy",
        "exact_recall",
        "accuracy",
        "examples",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            parts.append(f"`{key}`={_format_value(metrics[key])}")
    for key in sorted(metrics):
        if key not in preferred and len(parts) < 6:
            parts.append(f"`{key}`={_format_value(metrics[key])}")
    return ", ".join(parts)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
