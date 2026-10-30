from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoreProbeConfig:
    sequence_length: int = 32
    train_size: int = 2048
    test_size: int = 2048
    epochs: int = 140
    batch_size: int = 256
    oscillators: int = 32
    hidden: int = 64
    seed: int = 1
    learned_lr: float = 3e-3
    frozen_lr: float = 5e-3
    bag_lr: float = 3e-3


def run_learnable_core_probe(
    *,
    out_dir: str | Path = "runs/learnable-core",
    sequence_length: int = 32,
    train_size: int = 2048,
    test_size: int = 2048,
    epochs: int = 140,
    batch_size: int = 256,
    oscillators: int = 32,
    hidden: int = 64,
    seed: int = 1,
) -> dict[str, Any]:
    """Train a differentiable oscillator core against frozen/readout baselines."""

    import torch

    started = time.perf_counter()
    torch.manual_seed(int(seed))
    config = CoreProbeConfig(
        sequence_length=int(sequence_length),
        train_size=int(train_size),
        test_size=int(test_size),
        epochs=int(epochs),
        batch_size=int(batch_size),
        oscillators=int(oscillators),
        hidden=int(hidden),
        seed=int(seed),
    )
    train = make_first_bit_memory_data(config.train_size, config.sequence_length, seed=config.seed)
    test = make_first_bit_memory_data(config.test_size, config.sequence_length, seed=config.seed + 1)

    learned = OscillatorMemoryClassifier(config.oscillators, config.hidden, train_core=True)
    frozen = OscillatorMemoryClassifier(config.oscillators, config.hidden, train_core=False)
    bag = BagOfBitsClassifier(config.hidden)

    results = {
        "learned_phase": train_classifier(learned, train, test, config.epochs, config.batch_size, config.learned_lr),
        "frozen_phase": train_classifier(frozen, train, test, config.epochs, config.batch_size, config.frozen_lr),
        "bag_mlp": train_classifier(bag, train, test, config.epochs, config.batch_size, config.bag_lr),
    }
    learned_acc = float(results["learned_phase"]["test_accuracy"])
    baseline_acc = max(float(results["frozen_phase"]["test_accuracy"]), float(results["bag_mlp"]["test_accuracy"]))
    payload = {
        "type": "phase-mesh-learnable-core-probe",
        "version": 1,
        "status": "pass" if learned_acc >= 0.75 and learned_acc - baseline_acc >= 0.15 else "red",
        "elapsed_s": time.perf_counter() - started,
        "task": {
            "name": "first_bit_memory",
            "description": "Binary sequences use the same symbols for signal and noise; the label is the first bit after a noisy sequence.",
            "chance_accuracy": 0.5,
        },
        "config": config.__dict__,
        "results": results,
        "claim_boundary": [
            "This is not an LLM benchmark.",
            "This tests whether gradients through an oscillator core help versus a frozen reservoir and a bag baseline.",
            "A pass means credit assignment exists and mattered on this toy memory task, not that PhaseMesh solves language.",
        ],
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.md").write_text(render_learnable_core_markdown(payload), encoding="utf-8")
    return payload


def make_first_bit_memory_data(count: int, sequence_length: int, *, seed: int) -> tuple[Any, Any]:
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    x = torch.randint(0, 2, (int(count), int(sequence_length)), generator=generator)
    y = x[:, 0].long()
    return x, y


class OscillatorMemoryClassifier:
    """Small differentiable oscillator sequence classifier.

    The core parameters are the token phase drive, natural frequencies, dense
    oscillator coupling, input gain, and leak. The frozen baseline keeps those
    fixed and trains only the readout.
    """

    def __new__(cls, oscillators: int, hidden: int, *, train_core: bool):  # type: ignore[override]
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.oscillators = int(oscillators)
                self.token_phase = nn.Parameter(
                    torch.randn(2, self.oscillators) * 0.2,
                    requires_grad=bool(train_core),
                )
                self.omega = nn.Parameter(
                    torch.randn(self.oscillators) * 0.05,
                    requires_grad=bool(train_core),
                )
                self.coupling = nn.Parameter(
                    torch.randn(self.oscillators, self.oscillators) * 0.03,
                    requires_grad=bool(train_core),
                )
                self.input_gain = nn.Parameter(torch.tensor(0.6), requires_grad=bool(train_core))
                self.leak = nn.Parameter(torch.tensor(0.12), requires_grad=bool(train_core))
                self.readout = nn.Sequential(
                    nn.Linear(self.oscillators * 2, int(hidden)),
                    nn.ReLU(),
                    nn.Linear(int(hidden), 2),
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                theta = x.new_zeros((x.shape[0], self.oscillators), dtype=torch.float32)
                coupling = torch.tanh(self.coupling)
                input_gain = torch.clamp(self.input_gain, 0.0, 2.0)
                leak = torch.clamp(self.leak, 0.0, 0.8)
                for index in range(x.shape[1]):
                    drive = self.token_phase[x[:, index]]
                    phase_diff = theta.unsqueeze(1) - theta.unsqueeze(2)
                    coupled = (torch.sin(phase_diff) * coupling.unsqueeze(0)).mean(dim=2)
                    theta = (1.0 - leak) * theta + 0.2 * (self.omega + coupled + input_gain * drive)
                    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
                features = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)
                return self.readout(features)

        return _Model()


class BagOfBitsClassifier:
    def __new__(cls, hidden: int):  # type: ignore[override]
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, int(hidden)),
                    nn.ReLU(),
                    nn.Linear(int(hidden), 2),
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                import torch.nn.functional as F

                return self.net(F.one_hot(x, 2).float().mean(dim=1))

        return _Model()


def train_classifier(
    model: Any,
    train: tuple[Any, Any],
    test: tuple[Any, Any],
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    train_x, train_y = train
    test_x, test_y = test
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(lr))
    before = parameter_snapshot(model)
    gradient_probe = compute_gradient_probe(model, train_x[: min(len(train_x), batch_size)], train_y[: min(len(train_y), batch_size)])

    for _epoch in range(int(epochs)):
        order = torch.randperm(len(train_x))
        for start in range(0, len(train_x), int(batch_size)):
            index = order[start:start + int(batch_size)]
            loss = F.cross_entropy(model(train_x[index]), train_y[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

    after = parameter_snapshot(model)
    with torch.no_grad():
        train_logits = model(train_x)
        test_logits = model(test_x)
        train_loss = float(F.cross_entropy(train_logits, train_y).item())
        test_loss = float(F.cross_entropy(test_logits, test_y).item())
        train_accuracy = float((train_logits.argmax(dim=1) == train_y).float().mean().item())
        test_accuracy = float((test_logits.argmax(dim=1) == test_y).float().mean().item())
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "trainable_parameter_count": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "gradient_probe": gradient_probe,
        "parameter_delta_l2": parameter_delta_l2(before, after),
    }


def compute_gradient_probe(model: Any, x: Any, y: Any) -> dict[str, float]:
    import torch.nn.functional as F

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    norms = {}
    for name, param in model.named_parameters():
        norms[name] = float(param.grad.norm().item()) if param.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    return norms


def parameter_snapshot(model: Any) -> dict[str, Any]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def parameter_delta_l2(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        name: float((after[name] - value).norm().item())
        for name, value in before.items()
        if name in after
    }


def render_learnable_core_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Learnable Phase Core Probe",
        "",
        payload["task"]["description"],
        "",
        "| Model | Test Accuracy | Train Accuracy | Trainable Params | Core Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("learned_phase", "frozen_phase", "bag_mlp"):
        result = payload["results"][name]
        core_delta = sum(
            float(value)
            for key, value in result["parameter_delta_l2"].items()
            if not key.startswith("readout") and not key.startswith("net")
        )
        lines.append(
            f"| {name} | {float(result['test_accuracy']):.3f} | {float(result['train_accuracy']):.3f} | "
            f"{int(result['trainable_parameter_count'])} | {core_delta:.4f} |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)
