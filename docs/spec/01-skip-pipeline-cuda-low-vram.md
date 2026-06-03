---
feature: skip-pipeline-cuda-low-vram
version: 1.0.0
status: implemented
source: session 260603-1303-trellis2-cpu-offload / spec.md
implementation: completed
---

# Spec: Skip `pipeline.cuda()` when `low_vram=true`

## Implementation Status

| Status | Description |
|--------|-------------|
| **Status** | ✅ **Completed** — Implemented in `nodes.py` |
| **Source** | Session `260603-1303-trellis2-cpu-offload` (architect spec) |
| **Review** | ✅ Approved — reviewer confirmed correct logic |

## Problem Statement

`Trellis2LoadModel.process()` calls `pipeline.cuda()` when `device == "cuda"` and `low_vram == True`. This calls `.to(device)` on ALL models in the pipeline simultaneously, loading every model to GPU at once (~8 GB + expandable_segments overhead). With `low_vram=true`, the intent is lazy loading — models should only transfer to GPU when needed for their specific operation, then offload to CPU.

**Impact:** ~8 GB of unnecessary simultaneous VRAM usage at load time. With `expandable_segments:True`, this inflates to ~20 GB.

## Design Decision

**When `low_vram=true`, skip `pipeline.cuda()` entirely.** Models load lazily on-demand via `load_*()` methods, which already have proper `.to(device)` / `.cpu()` wrapping inside `sample_*()` methods.

**Rationale:**
- The `load_*()` methods (e.g., `load_shape_slat_flow_model_512()`) check `is None` before loading
- Inside `sample_*()` methods, models are moved to GPU before compute and to CPU after: `.to(device)` → sample → `.cpu()`
- `pipeline.cuda()` defeats this pattern by pre-loading everything
- When `low_vram=false`, eager loading is correct (user explicitly chose it)

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `nodes.py` | 512-517 | Replace `if device=="cuda": if low_vram: pipeline.cuda()` with explicit low_vram guard |

## Implementation Details

### `nodes.py` — `Trellis2LoadModel.process()`

**Before:**
```python
if device=="cuda":
    if low_vram:
        pipeline.cuda()      # ← loads ALL models to GPU
    else:
        pipeline.to(device)
else:
    pipeline.to(device)
```

**After:**
```python
if device == "cuda" and low_vram:
    # low_vram: models load lazily via load_*() methods with .to(device)/.cpu() wrapping
    pass
else:
    # Non-cuda devices, or cuda without low_vram: load models eagerly
    pipeline.to(device)
```

**Review note (M1 fix):** The original spec used `if device=="cuda" and not low_vram: pipeline.to(device)` which had an ambiguous `else` branch catching both non-cuda and low_vram cuda. The reviewer flagged this and the developer restructured to `if device == "cuda" and low_vram: pass` / `else: pipeline.to(device)` for clarity.

## VRAM Impact

| Phase | Before | After | Change |
|-------|--------|-------|--------|
| Load time | All models → GPU (~8 GB) | No models → GPU (0 GB) | **-8 GB** |
| Sampling | Model → GPU → CPU (per step) | Same (unchanged) | — |
| Peak | ~20 GB (with expandable) | ~12 GB (with fix) | **-8 GB** |

## Success Criteria

- [x] `pipeline.cuda()` not called when `low_vram=true` and `device=="cuda"`
- [x] Models still load on-demand during sampling (no regression)
- [x] `pipeline.to(device)` still called when `low_vram=false` or non-cuda
- [x] No syntax errors (AST parse passes)
- [x] Trellis2 v3.1 workflow completes without OOM on RTX 5090

---

## Open Decisions

1. **Should we add a log message when `pipeline.cuda()` is skipped?** Current decision: no — the behavior is internal and the user sets `low_vram=true` explicitly. A log message would be noise.
