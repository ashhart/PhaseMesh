# PhaseSSM 130M Scout

Status date: 2026-06-24.

This is the first completed 131.7M-parameter PhaseSSM quality scout on the
text8 byte-level training path.

## Result

| Metric | Value |
| --- | ---: |
| Parameters | 131,688,256 |
| Best validation bpc | 1.2568 |
| Best step | 4750 |
| Final validation bpc | 1.2603 |
| Final step | 5000 |
| Typical interval throughput | ~12.6k tok/s |
| Checkpoint size | ~503 MB |

The run completed cleanly on NVIDIA GB10. The best checkpoint is:

```text
runs/ssm-130m-scout/best.pt
```

## Generation Smoke Test

Command:

```bash
python -m phase_ssm.chat \
  --checkpoint runs/ssm-130m-scout \
  --max-tokens 80 \
  --temperature 0.8 \
  --top-k 40 \
  --json \
  "the "
```

Observed completion:

```text
program reports that some participants object to the condition of the community
```

Measured generation payload:

| Field | Value |
| --- | ---: |
| tokens_generated | 80 |
| elapsed_s | 4.607 |
| tokens_per_s | 17.36 |
| checkpoint_step | 4750 |
| checkpoint_val_bpc | 1.2568 |

## Read

This is a real talkable byte-level checkpoint and a material quality improvement
over the earlier partial scout rows. It is not yet a matched 130M transformer
comparison and it is not an instruction-following assistant. The next quality
gate is to run a matched transformer with the same data/config budget and report
bpc/perplexity side by side.
