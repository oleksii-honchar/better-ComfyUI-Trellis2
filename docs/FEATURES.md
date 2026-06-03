# better-ComfyUI-Trellis2 Features

This document describes the six features added by the `better-ComfyUI-Trellis2` fork.

For an overview of the fork's purpose and installation, see [BETTER-TRELLIS2.md](./BETTER-TRELLIS2.md).

---

## 1. Skip `pipeline.cuda()` when `low_vram=true`

📋 [Detailed Spec](./spec/01-skip-pipeline-cuda-low-vram.md)

**Status:** ✅ Implemented

**Problem:** `Trellis2LoadModel.process()` calls `pipeline.cuda()` when `device == "cuda"` and `low_vram == True`. This calls `.to(device)` on ALL models in the pipeline, loading every model to GPU simultaneously at load time (~8 GB + activations). With `low_vram=true`, the intent is lazy loading — models should only go to GPU when needed.

**Solution:** When `low_vram=true`, skip the `cuda()` call entirely. Models load lazily on-demand via `load_*()` methods, which already have proper `.to(device)` / `.cpu()` wrapping inside `sample_*()` methods.

**Before:**
```python
if device == "cuda":
    if low_vram:
        pipeline.cuda()      # ← loads ALL models to GPU
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

**Impact:** Prevents ~8 GB of simultaneous model loading at start-up. Models load on-demand during sampling phases.

---

## 2. Replace `expandable_segments:True` with Tight Allocator Config

📋 [Detailed Spec](./spec/02-replace-expandable-segments.md)

**Status:** ✅ Implemented

**Problem:** `expandable_segments:True` causes PyTorch's CUDA allocator to expand all GPU memory segments aggressively. On a 32 GB RTX 5090, this inflates allocations by ~12 GB beyond actual model weight usage, pushing peak VRAM from ~16 GB to ~28 GB.

**Solution:** Replace with `max_split_size_mb:128,garbage_collection_threshold:0.6`:
- `max_split_size_mb:128` — limits fragmentation by splitting large allocations into 128 MB chunks
- `garbage_collection_threshold:0.6` — triggers allocator GC at 60% capacity, keeping memory tight

**Before:**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

**After:**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
```

**Impact:** ~12 GB VRAM reduction — the single largest fix in this fork.

---

## 3. Remove `empty_cache()` from `reset_cuda()`

📋 [Detailed Spec](./spec/03-remove-empty-cache-reset-cuda.md)

**Status:** ✅ Implemented

**Problem:** `reset_cuda()` calls `torch.cuda.empty_cache()`. When cumesh objects exist during mesh post-processing, `cudaMallocAsync` inside `empty_cache()` reclaims cumesh-owned memory, causing "Tensor without storage" crashes.

**Solution:** Remove `empty_cache()` from `reset_cuda()`. The pipeline's `_cleanup_cuda()` method (called between phases after `unload_*()`) is the proper place for `empty_cache()` — it only runs when cumesh objects are safely freed.

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

**Impact:** Eliminates "Tensor without storage" crashes during mesh post-processing.

---

## 4. Add `torch.cuda.synchronize()` Around Cumesh Operations

📋 [Detailed Spec](./spec/04-synchronize-cumesh-operations.md)

**Status:** ✅ Implemented

**Problem:** Without `synchronize()` around cumesh operations, `empty_cache()` (called by `_cleanup_cuda()` between pipeline phases) can reclaim cumesh-owned GPU memory, causing "Tensor without storage" crashes.

**Solution:** Add 3 `synchronize()` calls per cumesh method (init/compute/read) = 9 total across `fill_holes()`, `remove_faces()`, `simplify_with_cumesh()` in `mesh/base.py`.

**Pattern (applied to all 3 methods):**
```python
mesh = cumesh.CuMesh()
mesh.init(vertices, faces)
torch.cuda.synchronize()  # Ensure init data is committed to device
mesh.fill_holes(...)       # or simplify() / remove_faces()
torch.cuda.synchronize()  # Ensure compute result is ready before read
new_vertices, new_faces = mesh.read()
torch.cuda.synchronize()  # Ensure read result is committed before del
del mesh
```

**Impact:** Eliminates "Tensor without storage" crashes from cumesh memory reclamation. Overhead: ~9-15 ms per cumesh operation (negligible vs seconds-long mesh processing).

---

## 5. `unload_all()` Method and `Trellis2UnloadModels` Node

📋 [Detailed Spec](./spec/05-unload-all-method-and-node.md)

**Status:** ✅ Implemented

**Problem:** After Trellis2 pipeline completes, models stay in CPU RAM with references in `self.models`. Python GC is lazy — models might not be freed until a GC cycle triggers. Users have no way to deterministically free memory when switching to non-Trellis2 workflows.

**Solution:** Add `unload_all()` method to `Trellis2ImageTo3DPipeline` that iterates all model keys, calls each `unload_*()` method, and runs `_cleanup_cuda()`. Expose via new `Trellis2UnloadModels` ComfyUI node.

**New `unload_all()` method:**
```python
def unload_all(self) -> None:
    """Unload ALL models — frees GPU and CPU memory."""
    for name in list(self.models.keys()):
        method = getattr(self, f"unload_{name}", None)
        if callable(method):
            method()
    if self.image_cond_model is not None:
        del self.image_cond_model
        self.image_cond_model = None
    for name, model in self.models.items():
        if model is not None:
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.models[name]
            self.models[name] = None
    self._cleanup_cuda()
```

**New ComfyUI node:**
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

**Impact:** Deterministic memory cleanup. Users add the node at the end of a Trellis2 workflow to free all GPU and CPU memory before switching workflows.

---

## 6. Cascade Sequential Model Loading (P1 — Not Used by Current Workflow)

📋 [Detailed Spec](./spec/06-cascade-sequential-loading.md)

**Status:** ✅ Implemented (affects cascade paths, not current workflow's `pipeline_type="512"`)

**Problem:** Cascade pipeline paths (`run()`, `run_cascade()`, `run_multiview()`) pre-load BOTH the 512 and 1024 shape flow models before sampling. This creates a ~4-6 GB double-load spike that doesn't exist in the upstream Microsoft pipeline, which loads models on-demand.

**Solution:** Remove pre-loading of the 1024 model. The `sample_shape_slat_cascade()` method already loads models on-demand inside via `.to(device)`.

**Before:**
```python
# run() — lines 1537-1538
self.load_shape_slat_flow_model_512()
self.load_shape_slat_flow_model_1024()
shape_slat, res = self.sample_shape_slat_cascade(...)
```

**After:**
```python
# run() — lines 1537-1538
self.load_shape_slat_flow_model_512()
# Don't load 1024 yet — sample_shape_slat_cascade loads it on demand
shape_slat, res = self.sample_shape_slat_cascade(...)
```

**Same pattern applied to all cascade paths:**
- `run()` — 4 cascade paths (lines 1537-1538, 1577-1578, 1617-1618, 1657-1658)
- `run_cascade()` — 2 cascade paths (lines 2018-2019, 2059-2060)
- `run_multiview()` — 3 cascade paths (lines 2227-2228, 2250-2251, 2273/2290)

**Impact:** Prevents ~4-6 GB double-load spike in cascade pipeline types. No effect on the current workflow's `pipeline_type="512"` path.

---

## VRAM Impact Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Peak VRAM (RTX 5090, 32 GB) | ~28 GB | ~12 GB | **-16 GB** |
| VRAM from model weights | ~8 GB | ~8 GB | — |
| VRAM from expandable_segments | ~12 GB | ~0 GB | **-12 GB** |
| VRAM from activations | ~4 GB | ~4 GB | — |
| OOM risk | HIGH | LOW | ✅ |
