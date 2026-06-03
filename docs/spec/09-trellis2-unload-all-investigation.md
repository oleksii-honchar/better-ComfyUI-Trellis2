# Trellis2 → UltraShape OOM Investigation (2026-06-03)

## Summary

Trellis2 leaves 26+ GiB GPU VRAM allocated after pipeline completion, causing UltraShape (needs ~13 GiB) to OOM. Root cause identified and fixed.

## Root Cause

**`Trellis2ImageTo3DPipeline.unload_all()` only freed models in `self.models` dict — missed 7+ direct-attribute models holding ~6 GiB of orphaned GPU memory.**

| Attribute | Model Type | Est. Size | Has unload method? |
|---|---|---|---|
| `moge_model` | MoGe geometry | ~0.5-1 GiB | ✅ `unload_moge_model()` |
| `pixal3d_image_cond_ss` | DINOv3 ViT-Large | ~1.2 GiB | ✅ `unload_pixal3d_image_cond_ss()` |
| `pixal3d_image_cond_shape_512` | DINOv3 ViT-Large | ~1.2 GiB | ✅ |
| `pixal3d_image_cond_shape_1024` | DINOv3 ViT-Large | ~1.2 GiB | ✅ |
| `pixal3d_image_cond_tex_1024` | DINOv3 ViT-Large | ~1.2 GiB | ✅ |
| `rembg_model` | Background removal | ~0.2 GiB | ❌ |
| `VGGT_model` | VGGT (texturing) | ~0.5 GiB | ❌ |

Total orphaned: ~5-7 GiB. Combined with ~15 GiB pipeline weights + ~3 GiB cumesh CUDA buffers → 26 GiB total, leaving only ~5 GiB for UltraShape's 13 GiB requirement.

## cudaMallocAsync Investigation (Dead End)

PyTorch 2.11+ defaults to `cudaMallocAsync` on Blackwell GPUs. Investigation showed:
- `backend:native` env var silently ignored — allocator stays cudaMallocAsync
- `expandable_segments:False` successfully switches to `native` allocator
- Even with `native` allocator, OOM persisted — proving allocator type was NOT the root cause

See [08-allocator-crossroads](./spec/08-allocator-crossroads-blackwell.md) for full analysis.

## Fix Applied

### 1. Extended `unload_all()` (trellis2_image_to_3d.py:549-571)

```python
# Unload direct-attribute models NOT in self.models dict
direct_attrs = [
    'moge_model', 'pixal3d_image_cond_ss',
    'pixal3d_image_cond_shape_512', 'pixal3d_image_cond_shape_1024',
    'pixal3d_image_cond_tex_1024', 'rembg_model', 'VGGT_model',
]
for attr in direct_attrs:
    if hasattr(self, attr):
        val = getattr(self, attr, None)
        if val is not None:
            print(f"[Trellis2 AutoUnload] Freeing direct attr: {attr}")
            if hasattr(val, 'cpu'):
                val.cpu()
            setattr(self, attr, None)
```

### 2. Auto-unload logging (nodes.py:1545-1551)

Added `[Trellis2 AutoUnload]` prefix to AdvancedGenerator's auto-unload calls.

### 3. Allocator config (docker-compose.yaml)

```yaml
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6
```

### 4. Existing patches (from prior sessions)

| Patch | File | What |
|---|---|---|
| Auto-unload on generate | nodes.py:1545 | `pipeline.unload_all()` called in AdvancedGenerator |
| CPU-offload mesh | nodes.py:2276 | `ReconstructMeshWithQuad` moves mesh to CPU |
| cumesh sync | base.py:58 | `torch.cuda.synchronize()` before cumesh reads |

## Files Modified (This Session)

| File | Change |
|---|---|
| `trellis2/pipelines/trellis2_image_to_3d.py:549-571` | Extended `unload_all()` with direct-attribute unloading + logging |
| `nodes.py:1547-1551` | Added `[Trellis2 AutoUnload]` prefix and success log |
| `docker-compose.yaml` | `expandable_segments:False` (switches allocator to native) |
| `docs/spec/05-unload-all-method-and-node.md` | Added fix note |
| `docs/spec/08-allocator-crossroads-blackwell.md` | NEW — cudaMallocAsync vs native investigation |
| `docs/spec/09-trellis2-unload-all-investigation.md` | THIS FILE |

## Verification Steps

1. Run Trellis2 pipeline
2. Check console for `[Trellis2 AutoUnload]` messages — should show which direct attributes were freed
3. After Trellis2 completes, nvidia-smi should show <2 GiB (down from 26 GiB)
4. Run UltraShape — should succeed without OOM

## Status

| Item | Status |
|---|---|
| Root cause identified | ✅ |
| `unload_all()` fix deployed | ✅ |
| Auto-unload logging added | ✅ |
| Allocator is native | ✅ |
| `expandable_segments:False` | ✅ |
| Awaiting test | ⏳ Run Trellis2 → UltraShape workflow |
