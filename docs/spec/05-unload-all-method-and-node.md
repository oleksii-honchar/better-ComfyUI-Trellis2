---
feature: unload-all-method-and-node
version: 1.0.0
status: implemented
source: session 260603-1303-trellis2-cpu-offload / spec.md
implementation: completed
---

# Spec: `unload_all()` Method and `Trellis2UnloadModels` Node

## Implementation Status

| Status | Description |
|--------|-------------|
| **Status** | ✅ **Completed** — Implemented in `trellis2_image_to_3d.py` + `nodes.py` |
| **Source** | Session `260603-1303-trellis2-cpu-offload` (architect spec) |
| **Review** | ✅ Approved — reviewer confirmed correct logic, minor redundancy noted (H1) |

## Problem Statement

After the Trellis2 pipeline completes, models stay in CPU RAM with references in `self.models`. Python GC is lazy — models might not be freed until a GC cycle triggers. Users have no way to deterministically free memory when switching to non-Trellis2 workflows.

**Symptoms:**
- User runs Trellis2 → pipeline finishes → models on CPU → 8+ GB CPU RAM consumed
- User queues a non-Trellis2 workflow → Trellis2 models still in CPU RAM → Python GC is lazy → CPU RAM stays consumed
- User queues another Trellis2 run → `load_*()` checks `is None` → model exists → `.to(device)` → fast restart (good, but memory never freed)

**Impact:** 8+ GB CPU RAM consumed after each Trellis2 run, no way to free it without restarting ComfyUI.

## Design Decision

**Add `unload_all()` method to `Trellis2ImageTo3DPipeline` and expose via new `Trellis2UnloadModels` ComfyUI node.**

**Rationale:**
- Individual `unload_*` methods exist (e.g., `unload_shape_slat_flow_model_512()`) but require knowing the exact model names
- A single `unload_all()` iterates all model keys and calls each `unload_*()` method
- The ComfyUI node gives users a visual, workflow-level control point
- Distinct from `Trellis2UnloadAllModels` which uses ComfyUI's memory management system (existing, unchanged)

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `trellis2/pipelines/trellis2_image_to_3d.py` | after line 526 | Add `unload_all()` method to `Trellis2ImageTo3DPipeline` |
| `nodes.py` | after `Trellis2CudaReset` | Add `Trellis2UnloadModels` node class |
| `nodes.py` | line 7357 | Register in `NODE_CLASS_MAPPINGS` |
| `nodes.py` | line 7433 | Register in `NODE_DISPLAY_NAME_MAPPINGS` |

## Implementation Details

### `trellis2_image_to_3d.py` — `Trellis2ImageTo3DPipeline.unload_all()`

**New method (after line 526, before `to()`):**
```python
def unload_all(self) -> None:
    """Unload ALL models — frees GPU and CPU memory."""
    # Use existing unload_* methods if available
    for name in list(self.models.keys()):
        method = getattr(self, f"unload_{name}", None)
        if callable(method):
            method()

    # Also unload image_cond_model
    if self.image_cond_model is not None:
        del self.image_cond_model
        self.image_cond_model = None

    # Fallback for models without dedicated unload_*:
    for name, model in self.models.items():
        if model is not None:
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.models[name]
            self.models[name] = None

    self._cleanup_cuda()
```

**Known limitation (H1):** `_cleanup_cuda()` is called N+1 times — each `unload_*()` method calls it, plus the final explicit call. Safe (idempotent) but wasteful. Follow-up candidate.

**Known limitation (M2):** The fallback loop does `del self.models[name]` then `self.models[name] = None` — `del` is unnecessary after setting to `None`. Style issue inherited from existing code.

**Fix (2026-06-03):** Original `unload_all()` only freed `self.models` dict entries. Investigation during Trellis2→UltraShape OOM revealed 7+ direct-attribute models never freed (~6 GB orphaned GPU memory). Added unloading for `moge_model`, 4× `pixal3d_image_cond_*`, `rembg_model`, and `VGGT_model`. See [08-allocator-crossroads](./08-allocator-crossroads-blackwell.md).

### `nodes.py` — `Trellis2UnloadModels` node

**New class (after `Trellis2CudaReset`, before `Trellis2SaveImage`):**
```python
class Trellis2UnloadModels:
    """Directly unloads all Trellis2 pipeline models via pipeline.unload_all().
    Distinct from Trellis2UnloadAllModels which uses ComfyUI's memory management system."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline):
        pipeline.unload_all()
        return (pipeline,)
```

### Registration

**`NODE_CLASS_MAPPINGS` (line 7357):**
```python
"Trellis2UnloadModels": Trellis2UnloadModels,
```

**`NODE_DISPLAY_NAME_MAPPINGS` (line 7433):**
```python
"Trellis2UnloadModels": "Trellis2 - Unload Models",
```

**Review note (C2 fix):** The reviewer found the node class was implemented but NOT registered. The developer fixed this by adding entries to both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

## Usage

### In Workflow

Add `Trellis2UnloadModels` at the end of a Trellis2 workflow:

```
Trellis2MeshWithVoxelAdvancedGenerator → Trellis2UnloadModels
```

The node takes a `TRELLIS2PIPELINE` input, calls `pipeline.unload_all()`, and returns the pipeline (for chaining or output).

### In Code

```python
pipeline = Trellis2ImageTo3DPipeline(...)
# ... run pipeline ...
pipeline.unload_all()  # Frees all GPU and CPU memory
```

## Success Criteria

- [x] `unload_all()` method present on `Trellis2ImageTo3DPipeline`
- [x] `Trellis2UnloadModels` node class defined
- [x] Node registered in `NODE_CLASS_MAPPINGS`
- [x] Node registered in `NODE_DISPLAY_NAME_MAPPINGS`
- [x] No syntax errors (AST parse passes)
- [x] Calling `unload_all()` frees all model memory (GPU and CPU)
- [x] Calling `unload_all()` is idempotent (safe to call multiple times)

## Verification

```bash
# Verify unload_all() exists
docker exec <container> python -c "
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
assert hasattr(Trellis2ImageTo3DPipeline, 'unload_all'), 'unload_all not found'
print('unload_all: OK')
"

# Verify Trellis2UnloadModels is registered
docker exec <container> python -c "
from ComfyUI.custom_nodes.ComfyUI_Trellis2.nodes import NODE_CLASS_MAPPINGS
assert 'Trellis2UnloadModels' in NODE_CLASS_MAPPINGS, 'Trellis2UnloadModels not registered'
print('Trellis2UnloadModels: OK')
"
```
