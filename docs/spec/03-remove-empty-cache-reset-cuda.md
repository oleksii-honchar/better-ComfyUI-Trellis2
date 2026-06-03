---
feature: remove-empty-cache-reset-cuda
version: 1.0.0
status: implemented
source: session 260603-1303-trellis2-cpu-offload / spec.md
implementation: completed
---

# Spec: Remove `empty_cache()` from `reset_cuda()`

## Implementation Status

| Status | Description |
|--------|-------------|
| **Status** | ✅ **Completed** — Implemented in `nodes.py` |
| **Source** | Session `260603-1303-trellis2-cpu-offload` (architect spec) |
| **Review** | ✅ Approved — reviewer confirmed correct removal |

## Problem Statement

`reset_cuda()` calls `torch.cuda.empty_cache()`. This function is called at the start of `Trellis2MeshWithVoxelAdvancedGenerator.process()` (line 1442) and `Trellis2MeshWithVoxelGenerator.process()` (line 553), and at 15 other call sites throughout the codebase.

When cumesh objects exist during mesh post-processing (fill holes, simplify, remove faces), `cudaMallocAsync` inside `empty_cache()` reclaims cumesh-owned GPU memory. This causes **"Tensor without storage"** crashes — cumesh's internal tensors lose their backing storage.

**Impact:** Intermittent crashes during mesh post-processing, especially when `empty_cache()` is called between pipeline phases while cumesh objects are still alive.

## Design Decision

**Remove `torch.cuda.empty_cache()` from `reset_cuda()` entirely.** The pipeline's `_cleanup_cuda()` method (called between phases after `unload_*()`) is the proper place for `empty_cache()` — it only runs when cumesh objects are safely freed.

**Rationale:**
- `reset_cuda()` is called at the start of `process()` methods — at that point, no cumesh objects exist yet (they're created inside `pipeline.run()`)
- However, `reset_cuda()` is also called at 15 other call sites where cumesh objects might exist
- `gc.collect()` + `torch.cuda.synchronize()` is sufficient for clearing Python-level references at the start of `process()`
- The `_cleanup_cuda()` method calls `empty_cache()` when it's safe (after `del cumesh` + `gc.collect()`)

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `nodes.py` | 137-145 | Remove `torch.cuda.empty_cache()` from `reset_cuda()` |

## Implementation Details

### `nodes.py` — `reset_cuda()`

**Before:**
```python
def reset_cuda():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
```

**After:**
```python
def reset_cuda():
    torch.cuda.synchronize()
    gc.collect()
    # Don't call empty_cache() if cumesh might be active — it would reclaim cumesh memory
    # and cause "Tensor without storage" crashes. The pipeline's _cleanup_cuda() handles
    # cleanup between phases when it's safe.
```

**Impact on call sites:** 17 call sites of `reset_cuda()` are automatically fixed. No need to patch each one individually.

### `reset_cuda()` Call Sites (for reference)

| File | Line | Context |
|------|------|---------|
| `nodes.py` | 373 | `Trellis2ImageTo3DGenerator.process()` |
| `nodes.py` | 553 | `Trellis2MeshWithVoxelGenerator.process()` |
| `nodes.py` | 1442 | `Trellis2MeshWithVoxelAdvancedGenerator.process()` |
| ... | ... | 14 more call sites |

## Risks and Mitigations

### Risk: Stale GPU memory from previous runs

**Severity:** Low — the pipeline's `_cleanup_cuda()` handles cleanup between phases.

**Mitigation:**
- `gc.collect()` + `torch.cuda.synchronize()` in `reset_cuda()` is sufficient for clearing Python-level references
- The `_cleanup_cuda()` method (called after each `unload_*()`) calls `empty_cache()` when it's safe
- For inter-run cleanup, ComfyUI's own memory management handles GPU eviction

## Success Criteria

- [x] `torch.cuda.empty_cache()` removed from `reset_cuda()`
- [x] Explanatory comment added (documents the reasoning)
- [x] No syntax errors (AST parse passes)
- [x] No "Tensor without storage" crashes during cumesh operations
- [x] Trellis2 v3.1 workflow completes without mesh processing errors

## Verification

```bash
# Check reset_cuda() implementation
docker exec <container> python -c "
from ComfyUI.custom_nodes.ComfyUI_Trellis2.nodes import reset_cuda
import inspect
print(inspect.getsource(reset_cuda))
"
# Expected: no empty_cache() call, has explanatory comment
```
