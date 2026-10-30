"""Premise test: does the additive phase memory actually DEGRADE at high recall load?

DREAM's recall pillar is justified only if the additive memory it replaces fails
at high load while a transformer (positive control) still succeeds. At P=4 the
additive memory is perfect (1.000), so the original 'additive fails MQAR' result
was an underpowering artifact. This sweep pushes the pair count up at fixed,
adequate d_model and watches whether a gap opens between transformer and additive.

Read:
  transformer holds, additive drops  -> a real recall gap exists -> a better memory is justified
  both hold                           -> no gap -> the recall pillar is unmotivated
  both drop together                  -> underpowered at that P (not informative)
"""
from __future__ import annotations
import torch
from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.transformer import TransformerConfig, TransformerLM
from phase_ssm.tasks import train_on_mqar

dev = "cuda" if torch.cuda.is_available() else "cpu"
d_model, n_layers, steps = 128, 2, 4000
print(f"PREMISE TEST | device={dev} d={d_model} L={n_layers} steps={steps}", flush=True)
print(f"{'P':>4} {'NK=NV':>6} {'chance':>7} {'1/P':>6} | {'transformer':>11} {'additive':>9} {'ssm-only':>8}", flush=True)

for P in [16, 32, 64, 128]:
    NK = NV = max(256, 2 * P)
    vocab = 1 + NK + NV
    def mk_xf():
        return TransformerLM(TransformerConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                                               n_heads=max(1, d_model // 32), d_ff_mult=2,
                                               block_size=512, pos_enc="rope"))
    def mk_add():
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                                         state_dim=32, expand=1, d_ff_mult=2, use_phase_memory=True,
                                         mem_heads=2, mem_head_dim=16))
    def mk_ssm():
        return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                                         state_dim=32, expand=1, d_ff_mult=2, use_phase_memory=False))
    row = {}
    for name, mk, lr in [("xf", mk_xf, 3e-4), ("add", mk_add, 1e-3), ("ssm", mk_ssm, 1e-3)]:
        torch.manual_seed(0)
        acc = train_on_mqar(mk(), steps=steps, n_pairs=P, n_queries=8, n_keys=NK, n_values=NV,
                            batch=64, lr=lr, device=dev, log_every=0)
        row[name] = acc
    print(f"{P:>4} {NK:>6} {1/NV:>7.4f} {1/P:>6.3f} | "
          f"{row['xf']:>11.3f} {row['add']:>9.3f} {row['ssm']:>8.3f}", flush=True)
print("PREMISE DONE", flush=True)
