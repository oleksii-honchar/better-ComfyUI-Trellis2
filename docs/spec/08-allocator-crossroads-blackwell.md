# 08. Allocator Crossroads: cudaMallocAsync, expandable_segments, and Custom Image

**Status:** 🔴 Open — investigation in progress
**Severity:** Critical — Trellis2 → UltraShape OOM on RTX 5090
**Date:** 2026-06-03

---

## The Crossroad

On Blackwell GPUs (RTX 5090), PyTorch 2.11+ force-enables `cudaMallocAsync` — a CUDA driver-level async memory allocator. Three paths exist to address the resulting OOM in the Trellis2 → UltraShape pipeline. This document records what works, what doesn't, and why.

---

## Experiment Results

### Test 1: `PYTORCH_CUDA_ALLOC_CONF=backend:native`

**Configuration:**
```yaml
# docker-compose.yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=backend:native
```

**Result: ❌ Ignored**
- Container starts without crash
- Device line still shows `NVIDIA GeForce RTX 5090 : cudaMallocAsync`
- Trellis2 `nodes.py` line 388 overrides env var with `max_split_size_mb:128,...` (no `backend:native`)
- **Why:** PyTorch 2.11 initializes cudaMallocAsync at `import torch` time. `backend:native` does NOT override it — no RuntimeError, just silent ignore. This contradicts the spec 07 prediction of a RuntimeError crash.

### Test 2: `expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6`

**Configuration (deployed 2026-06-03):**
```yaml
# docker-compose.yaml (env var set BEFORE Python starts)
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6
```
```python
# Trellis2 nodes.py line 388 (set AFTER import torch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6"
```

**Result: ⚠️ Allocator now `native`, but OOM persists**

| Metric | Fresh restart | After Trellis2 run |
|---|---|---|
| Allocator backend | `native` ✅ | `native` ✅ |
| `torch.cuda.memory_allocated()` | 0.0 GiB | **26.6 GiB** 🔴 |
| `torch.cuda.memory_reserved()` | 0.0 GiB | **30.0 GiB** 🔴 |
| Free (torch.mem_get_info) | 16 GiB | **0 B** 🔴 |
| nvidia-smi VRAM used | 0 | 15.9 GiB |

**Key finding:** `expandable_segments:False` **successfully** switches the allocator backend from `cudaMallocAsync` to `native`. This is unexpected — `backend:native` was thought necessary, but `expandable_segments:False` (when set in the Docker environment BEFORE Python starts) achieves the same result on this PyTorch build.

**But:** the OOM still occurs because Trellis2 allocates 26 GiB that is never freed — even with the native allocator and `torch.cuda.empty_cache()` calls.

### Test 3: Custom Docker Image (NOT YET TRIED)

**Proposed approach:**
```dockerfile
FROM mmartial/comfyui-nvidia-docker:ubuntu24_cuda12.8-latest
ENV PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6
```

**Theoretical benefit:** `ENV` instruction sets the variable at image build time, so it's in the environment when `import torch` first runs during ComfyUI startup. This is earlier than docker-compose env vars.

**Risk:** Spec 07 predicts a RuntimeError (`Allocator backend parsed at runtime != allocator backend parsed at load time`). Since Test 1 didn't trigger this error (backend:native was silently ignored), Test 3 might also be silently ignored.

**Current assessment:** Test 3 is **not needed** because Test 2 already achieves a `native` backend. The remaining problem is NOT the allocator backend — it's Trellis2's failure to release pipeline memory.

---

## Root Cause (Revised)

The cudaMallocAsync vs native allocator debate was a **red herring**. The actual problem has two layers:

### Layer 1: Pipeline Memory Not Freed (PRIMARY)

Trellis2 loads ~12 GB of model weights (VAE, FLUX-1, Trellis2 image-to-3d) into GPU memory. After processing:
- `Trellis2UnloadAllModels` — uses ComfyUI's `mm.current_loaded_models` → Trellis2 models NOT tracked → frees **0 B**
- `Trellis2UnloadModels` — checks `hasattr(data, 'unload_all')` on connection output → receives MESH (not pipeline) → also frees **0 B**
- `AdvancedGenerator` auto-unload (spec 05) — calls `pipeline.unload_all()` BUT `unload_all()` only deletes the pipeline wrapper, not the model objects themselves

### Layer 2: cumesh CUDA Buffers (SECONDARY)

Even when pipeline models ARE freed, cumesh (`CuMesh`) allocates CUDA working buffers via the CUDA driver API directly, bypassing PyTorch entirely. These buffers survive `torch.cuda.empty_cache()` and model deletion. They're released only when:
1. The CUDA context is destroyed (process exit)
2. `CuMesh` destructor runs AND the CUDA driver frees the allocation

With the native allocator, these cumesh buffers still block UltraShape from allocating ~13 GB of contiguous memory for its own models.

---

## Forward Path

### Short-term (test now)

1. **Verify Trellis2 standalone behavior** — run Trellis2 workflow, note VRAM before and after, without chaining to UltraShape
2. **Check `unload_all()` effectiveness** — instrument to log what it actually frees
3. **Force cumesh cleanup** — add `CuMesh` destructor calls and `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` loops after all mesh operations

### Medium-term

4. **Subprocess isolation** — run Trellis2 mesh processing in a separate Python subprocess. When the subprocess exits, all CUDA memory (including cumesh buffers) is released to the OS.

5. **Move mesh intermediates to CPU** — after Trellis2 processing, move ALL tensors (vertices, faces, normals) to CPU before returning. The current ReconstructMeshWithQuad patch does this partially.

### Long-term

6. **Rebuild cumesh with PyTorch allocator** — the only permanent fix for the cumesh layer.

---

## Test Log

| Date | Config | Allocator | After Trellis2 | After Unload | OOM? |
|---|---|---|---|---|---|
| 2026-06-03 | expandable_segments:False | native | 26.6 GiB allocated | 26.6 GiB (unchanged) | ✅ yes |
| 2026-06-03 | backend:native | cudaMallocAsync | — | — | ✅ yes |

---

## Related

- [07. cudaMallocAsync / cumesh Memory Conflict](./07-cudamallocasync-cumesh-conflict.md) — original crash investigation
- [05. Pipeline Unload All](./05-unload-all-method-and-node.md) — auto-unload patch in AdvancedGenerator
- docker-compose.yaml → `COMFY_CMDLINE_EXTRA=--enable-manager --lowvram`
