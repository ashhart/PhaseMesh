"""Recall-fix probe: can DREAM's delta memory recall once it gets the local
mixing (short conv) and a clean residual path the additive memory already has?

At P=4/d=128 the additive memory and transformer both hit 1.000; DREAM's settling
block sits at the 0.25 floor. This isolates why:
  +shortconv          -> does local mixing (k,v co-location) fix it inside the settling block?
  mem_only            -> does a clean residual (no oscillator/MLP entanglement) fix it?
  mem_only+shortconv  -> both: is the delta MECHANISM itself fine when set up like the additive memory?
"""
from __future__ import annotations
import torch
from phase_ssm.dream import DreamConfig, DreamLM
from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM
from phase_ssm.transformer import TransformerConfig, TransformerLM
from phase_ssm.tasks import train_on_mqar

dev = "cuda" if torch.cuda.is_available() else "cpu"
P, Q, NK, NV = 4, 4, 32, 32
vocab = 1 + NK + NV
d, L, steps = 128, 2, 3000
print(f"RECALL-FIX | P={P} d={d} steps={steps} chance={1/NV:.3f} floor={1/P:.3f} dev={dev}", flush=True)


def dream(**kw):
    return DreamLM(DreamConfig(vocab_size=vocab, d_model=d, n_layers=L, state_dim=32,
                               mem_heads=2, mem_head_dim=16, **kw))


def additive():
    return PhaseSSMLM(PhaseSSMConfig(vocab_size=vocab, d_model=d, n_layers=L, state_dim=32,
                                     expand=1, d_ff_mult=2, use_phase_memory=True,
                                     mem_heads=2, mem_head_dim=16))


def xf():
    return TransformerLM(TransformerConfig(vocab_size=vocab, d_model=d, n_layers=L,
                                           n_heads=max(1, d // 32), d_ff_mult=2,
                                           block_size=512, pos_enc="rope"))


specs = [
    ("DREAM settling (baseline)", lambda: dream(outer_iters=1, damp_init=2.0), 1e-3),
    ("DREAM +shortconv", lambda: dream(outer_iters=1, damp_init=2.0, mem_short_conv=True), 1e-3),
    ("DREAM mem_only", lambda: dream(mem_only=True), 1e-3),
    ("DREAM mem_only+shortconv", lambda: dream(mem_only=True, mem_short_conv=True), 1e-3),
    ("additive (ref)", additive, 1e-3),
    ("transformer (ref)", xf, 3e-4),
]
for name, mk, lr in specs:
    torch.manual_seed(0)
    acc = train_on_mqar(mk(), steps=steps, n_pairs=P, n_queries=Q, n_keys=NK, n_values=NV,
                        batch=64, lr=lr, device=dev, log_every=0)
    print(f"  {name:<28} recall={acc:.3f}", flush=True)
print("RECALL-FIX DONE", flush=True)
