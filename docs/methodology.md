# Methodology And Reproduction Limits

## Paper-Reported Settings

Source: VITA, arXiv:2511.19859v2, Table 11 and Sections 7.3, 8.2 and 9.2.
This document describes an independent implementation, not recovered author code.

| Setting | Paper | This LIBERO-only recipe |
| --- | --- | --- |
| Warmup / co-train updates | 200K / 100K | Matched |
| Pretraining batch | 1024 | Matched via gradient accumulation |
| Pretraining AdamW | LR 1e-4, betas .9/.95, decay .01, BF16 | Matched |
| LIBERO fine-tune | 60K updates, batch 192, two cameras | Matched |
| Pretraining data | Mixed human video, robot action/video, simulation | LIBERO training episodes only |
| Foundation model | Pretrained PaliGemma, frozen DINOv2 | Compact random model by default; optional local adapters |
| Image / future frames | 256 / 12 | 32 / 2 default; 256 / 12 optional |
| Codebook | 8192 entries of dimension 1024 | 64 x 32 default; optional paper dimensions |
| Fine-tune scope | Section 7.3/Algorithm 1: action decoder only | Action decoder only |
| Temporal sampling | One-second keyframe intervals | Default stride 1 frame, explicit adaptation |
| Demo count | Section 9.2: 50 demonstrations/task | Uses available eligible data with 80/10/10 episode split |
| Simulator trials | Not unambiguously specified in cited text | 50 initial states/task by explicit convention |

The total is 360,000 optimizer updates, not 360,000 epochs. With effective
batches 1024/192, the schedule draws 307,200,000 pretraining windows and
11,520,000 fine-tuning windows (with reshuffling/reuse). These counts do not
represent distinct demonstrations. Gradient accumulation is not an exact claim
of matching distributed floating-point numerics or data ordering.

## Independent Choices

- Fine-tuning LR `1e-4`, constant LR, clipping at norm 1, seed 17, validation and
  checkpoint cadence are independent choices where the cited settings do not
  fully specify the procedure.
- The model implements shared visual/action quantization, staged freezing,
  progressive textual/internal motion tokens, action reconstruction and DCT
  decoding. Straight-through gradients and numerical losses are explicit
  implementation choices; unpublished source details cannot be recovered from
  prose alone.
- Warmup alternates visual and action objectives by optimizer update on paired
  LIBERO data. Co-training uses paired LIBERO observations/actions. This differs
  from a heterogeneous human-video/robot corpus and is not a substitute for it.
- Section 8.2 contains differing fine-tuning wording. This recipe follows
  Section 7.3 and Algorithm 1 rather than silently claiming both interpretations.
- `configs/libero_paper_model.json` sets LIBERO state dimension 8 and retains
  independent choices for M-Former, token layout, action horizon 12 and
  PaliGemma positional interpolation to 256 pixels. Reported parameter counts
  are not guaranteed to match the paper. Action horizon and latent token count
  are assumptions, not deduced merely from future-frame count.
- Twelve future frames at stride 20 can exclude short LIBERO demonstrations.
  The recipe therefore defaults to stride 1. Setting stride 20 is possible but
  requires reviewing exclusions and task coverage; missing usable tasks fail.
- One model is trained across the requested suites. Per-suite independent
  training, paper data filtering and exact author normalization are not claimed.
- The separate legacy `libero_balanced` configuration includes engineering
  changes (both-modality codebook updates and auxiliary token loss). The new
  recipe's compact default does not silently enable those changes.

## What Validation Means

Synthetic CPU tests verify implementation contracts: exact resume, frozen
modules, split isolation, stage dependency, export equality, metric aggregation,
error propagation and release packaging. Mock rollout receipts are labeled as
synthetic fixtures and are not real simulator evidence. Large foundation model
training and paper-level success rates are not established by those tests.

A successful final receipt means the configured schedule and requested
assessment were completed with consistent artifact identities. It never sets
`benchmark_reproduced` to true. This release contains no historical benchmark
receipts and no performance claim.
