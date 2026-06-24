"""Generate text from a trained PhaseSSM or matched transformer checkpoint.

The training script saves byte-level LM checkpoints as ``best.pt``. This module
is the small talk-to-it entrypoint for those checkpoints:

    python -m phase_ssm.generate --checkpoint runs/ssm/best.pt "PhaseMesh is"
    python -m phase_ssm.chat --checkpoint runs/ssm/best.pt "PhaseMesh is"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM


def _checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "best.pt"
    return candidate


def load_checkpoint(path: str | Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = _checkpoint_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = payload.get("model_type", "phasessm")
    cfg_data = dict(payload["cfg"])

    if model_type == "phasessm":
        model = PhaseSSMLM(PhaseSSMConfig(**cfg_data))
    elif model_type == "transformer":
        model = TransformerLM(TransformerConfig(**cfg_data))
    else:
        raise ValueError(f"unsupported checkpoint model_type={model_type!r}")

    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def encode_prompt(text: str, device: str) -> torch.Tensor:
    data = list(text.encode("utf-8", errors="replace"))
    if not data:
        data = [10]
    return torch.tensor([data], dtype=torch.long, device=device)


def decode_bytes(ids: torch.Tensor) -> str:
    values = [int(x) & 0xFF for x in ids.detach().cpu().flatten().tolist()]
    return bytes(values).decode("utf-8", errors="replace")


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    prompt: str,
    *,
    device: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
) -> dict[str, Any]:
    ids = encode_prompt(prompt, device)
    out = model.generate(
        ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    prompt_len = ids.shape[1]
    return {
        "prompt": prompt,
        "text": decode_bytes(out[0]),
        "completion": decode_bytes(out[0, prompt_len:]),
        "tokens_generated": int(out.shape[1] - prompt_len),
        "tokens_total": int(out.shape[1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained PhaseSSM checkpoint.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, read prompts interactively.")
    parser.add_argument("--checkpoint", "--model", required=True, help="Path to best.pt or a training run directory.")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true", help="Print the generation payload as JSON.")
    parser.add_argument("--completion-only", action="store_true", help="Print only newly generated text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    top_k = None if args.top_k <= 0 else args.top_k
    model, payload = load_checkpoint(args.checkpoint, args.device)

    def emit(prompt: str) -> None:
        result = generate_text(
            model,
            prompt,
            device=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
        result["checkpoint"] = str(_checkpoint_path(args.checkpoint))
        result["model_type"] = payload.get("model_type", "phasessm")
        result["step"] = payload.get("step")
        result["val_bpc"] = payload.get("val_bpc")
        if args.json:
            print(json.dumps(result, indent=2))
        elif args.completion_only:
            print(result["completion"])
        else:
            print(result["text"])

    if args.prompt:
        emit(" ".join(args.prompt))
        return 0

    print("PhaseSSM chat. Ctrl-D to exit.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        if prompt:
            emit(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
