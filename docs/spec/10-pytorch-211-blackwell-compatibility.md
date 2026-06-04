# PyTorch 2.11 + Blackwell (RTX 5090) Compatibility Fixes

**Status:** Fixed (5 layers of patches applied)
**Environment:** PyTorch 2.11.0+cu128, NVIDIA RTX 5090, ComfyUI 0.23.0, cu128, xformers 0.0.35

## Overview

PyTorch 2.11 on Blackwell GPUs introduces stricter tensor metadata checks that break three dependencies:
cuMesh, o_voxel, and themselves (cudaMallocAsync). This document catalogs all failures
and the applied fixes.

---

## Layer 1: cudaMallocAsync + cuMesh Race Condition (FIXED)

**Symptom:** `torch.OutOfMemoryError` or CUDA kernel crashes after multiple pipeline runs.

**Root Cause:** PyTorch 2.11 force-enables `cudaMallocAsync` on Blackwell. cuMesh's
CUDA memory allocations interleave with PyTorch's async allocator, causing
use-after-free races.

**Fix:** `PYTORCH_NO_CUDA_MEMORY_CACHING=1` in docker-compose.yaml.
Disables PyTorch's caching allocator entirely — all allocations use direct `cudaMalloc`.
Trade-off: `torch.cuda.empty_cache()` becomes a no-op.

**Attempted alternatives:**
- `PYTORCH_CUDA_ALLOC_CONF=backend:native` — silently ignored on torch ≥ 2.9 on Blackwell
- Downgrade torch to 2.9 — xformers requires torch ≥ 2.10, plus 2.9 also force-enables async
- Custom Docker image with torch 2.9 + backend:native — dead end

---

## Layer 2: cuMesh `fill_holes()` Internal Crashes (FIXED)

**Symptom:** `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`
in `cumesh.CuMesh.read()` called from `Mesh.fill_holes()` inside pipeline's `decode_latent`.

**Root Cause:** cuMesh's internal CUDA tensor management is incompatible with both
`cudaMallocAsync` and the fallback `NO_CUDA_MEMORY_CACHING` mode on Blackwell.

**Fix:** Replaced all cuMesh operations in `trellis2/representations/mesh/base.py` with trimesh:
- `fill_holes()` → `trimesh.Trimesh.fill_holes()` (CPU, GPU→CPU→GPU roundtrip)
- `simplify_with_cumesh()` → `trimesh.Trimesh.simplify_quadratic_decimation()` (CPU)
- `remove_faces()` → torch boolean indexing (GPU, no cuMesh)

**Files changed:** `trellis2/representations/mesh/base.py`

---

## Layer 3: cuMesh `CuMesh.cuBVH()` Crashes (FIXED)

**Symptom:** `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`
when building BVH after mesh generation.

**Root Cause:** `CuMesh.cuBVH()` creates a CUDA-accelerated BVH structure. The CUDA
tensor operations inside cuBVH() fail on Blackwell/PyTorch 2.11.

**Fix:** Patched `import cumesh as CuMesh` with a `_safe_cuBVH()` wrapper that catches
`RuntimeError` and returns a `_SafeBVH` placeholder with `.vertices` and `.faces` attributes.
The BVH is only used by texturing nodes (UnwrapAndRasterizer), not by our Generator→Refiner pipeline.

**Files changed:** `nodes.py` (import block)

---

## Layer 4: Workflow cuMesh Nodes Eliminated (FIXED)

**Symptom:** `cumesh.read()` crashes in various Trellis2 nodes (PostProcessMesh,
SimplifyMesh, ReconstructMeshWithQuad, FillHolesWithCuMesh).

**Root Cause:** These nodes create `CuMesh.CuMesh()` instances directly and call
methods on them — bypassing the base.py fixes.

**Fix:** Replaced all cuMesh-dependent ComfyUI nodes with alternatives:
- `Trellis2ReconstructMeshWithQuad` → **REMOVED** (replaced by direct PostProcess2 path)
- `Trellis2PostProcessMesh` → `Trellis2PostProcess2` (CPU-only trimesh operations)
- `Trellis2SimplifyMesh` → **REMOVED** (consolidated into PostProcess2)
- `Trellis2FillHolesWithMeshlib` → **REMOVED** (consolidated into PostProcess2)
- UltraShape nodes → `Trellis2MeshRefiner` (reuses pipeline SLAT models, no cuMesh)

**Workflow:** `Trellis2 v3.3 (1).json`

---

## Layer 5: o_voxel `set_stride()` Crash (FIXED)

**Symptom:**
```
RuntimeError: set_stride is not allowed on a Tensor created from .data or .detach().
```
in `o_voxel.convert.flexible_dual_grid._C.mesh_to_flexible_dual_grid_cpu()`,
called from `encode_shape_slat()` in the MeshRefiner pipeline.

**Root Cause:** PyTorch 2.11 added a `allow_tensor_metadata_change()` check in ALL
`TensorImpl` metadata mutators (`set_stride`, `set_size`, `set_storage_offset`,
`set_sizes_contiguous`, and both overloads of `set_sizes_and_strides`).
The o_voxel C++ extension (precompiled `.so`, no source available) was compiled
against PyTorch 2.7 where these checks didn't exist. When run on PyTorch 2.11,
its internal C++ tensors have `allow_tensor_metadata_change_ = false` by default,
and any call to `set_stride()` on them triggers:

```cpp
// c10/core/TensorImpl.h:1935 (IntArrayRef overload)
TORCH_CHECK(
    allow_tensor_metadata_change(),
    "set_sizes_and_strides ",
    err_msg_tensor_metadata_change_not_allowed);
```

The o_voxel C++ completes all 4 QEF phases successfully (Intersect QEF, Face QEF,
Boundary QEF, Dual vertices) then crashes during result tensor creation/reshaping.

**Fix:** `torch._C._set_tensor_metadata(t, {"allow_tensor_metadata_change": True})`.
This internal PyTorch API sets the `allow_tensor_metadata_change_` flag on a tensor.
**Critically, this flag propagates through ALL PyTorch operations** — clone, view,
as_strided, arithmetic, empty_like. Setting it on the input tensors means all
C++-internal tensors derived from them inherit `allow_tensor_metadata_change=True`,
bypassing the check.

1. In `trellis2/pipelines/trellis2_image_to_3d.py` (`encode_shape_slat` and `get_coords_from_trimesh`):
   ```python
   vertices = torch.from_numpy(mesh.vertices).float().clone()
   faces = torch.from_numpy(mesh.faces).long().clone()
   torch._C._set_tensor_metadata(vertices, {"allow_tensor_metadata_change": True})
   torch._C._set_tensor_metadata(faces, {"allow_tensor_metadata_change": True})
   ```

2. In `o_voxel/convert/flexible_dual_grid.py` (in-container, as defense-in-depth):
   ```python
   def _ensure_grad_meta(t):
       if not isinstance(t, torch.Tensor): return t
       if t.is_floating_point():
           return torch.empty_like(t).copy_(t).requires_grad_(True)
       return torch.empty_like(t).copy_(t)
   vertices = _ensure_grad_meta(vertices)
   voxel_size = _ensure_grad_meta(voxel_size)
   grid_range = _ensure_grad_meta(grid_range)
   ```

**Attempted alternatives that DID NOT work (9 attempts across two sessions):**

| # | Approach | Why it failed |
|---|----------|---------------|
| 1 | `.clone()` to own storage | `.clone()` preserves null `autograd_meta` from `from_numpy()` source |
| 2 | `torch.no_grad()` wrapper | C++ `allow_tensor_metadata_change()` check ignores `GradMode` |
| 3 | `sitecustomize.py` monkey-patch | C++ methods (`set_()`) can't be patched from Python |
| 4 | `faces.requires_grad_(True)` | `RuntimeError: only Tensors of floating point dtype can require gradients` |
| 5 | `.requires_grad_(True)` on `vertices` only | C++ still hits internal tensors without the flag |
| 6 | `torch.empty_like().copy_().requires_grad_(True)` on ALL float inputs | Same — only fixes inputs, not C++-internal tensors |
| 7 | Swap to Torch 2.9.1 o_voxel `.so` | `set_stride` check is in PyTorch 2.11 runtime, not in the `.so` |
| 8 | `LD_PRELOAD` intercept | Can't intercept C++ class methods with `LD_PRELOAD` |
| 9 | Remove `@torch.no_grad()` | Check is on `allow_tensor_metadata_change`, not `GradMode` |

**Files changed:**
- `trellis2/pipelines/trellis2_image_to_3d.py` (both `encode_shape_slat` and `get_coords_from_trimesh`)
- `/comfy/mnt/venv/lib/python3.12/site-packages/o_voxel/convert/flexible_dual_grid.py` (in-container, defense-in-depth via `_ensure_grad_meta`)

---

## Complete List of Modified Files

| File | Changes |
|---|---|
| `nodes.py` | `_SafeBVH` fallback for `CuMesh.cuBVH()`, auto-unload in AdvancedGenerator, mesh CPU-offload, nuclear GC sweep |
| `trellis2/representations/mesh/base.py` | `fill_holes()` → trimesh, `simplify_with_cumesh()` → trimesh, `remove_faces()` → torch |
| `trellis2/pipelines/trellis2_image_to_3d.py` | `torch._C._set_tensor_metadata(... allow_tensor_metadata_change: True)` on vertices+faces before o_voxel calls, extended `unload_all()` with direct-attr unloading |
| `o_voxel/convert/flexible_dual_grid.py` | `_ensure_grad_meta()` helper before C++ call — defense-in-depth (in-container patch) |
| `Trellis2 v3.3 (1).json` | Workflow: PostProcess2 instead of PostProcessMesh, no Reconstruct/Simplify/UltraShape |

## Known Limitations

- **BVH operations are dummy:** `CuMesh.cuBVH()` fallback returns vertices/faces container only. Texturing/unwrap nodes that use BVH ray casting will fail.
- **`PYTORCH_NO_CUDA_MEMORY_CACHING=1`** remains set in docker-compose. Remove only if cuMesh is completely eliminated from all workflows.
- **o_voxel patches are in-container only:** Both `flexible_dual_grid.py` patches are applied directly to the venv. If the container is rebuilt from scratch, they must be reapplied.
- **`allow_tensor_metadata_change` is a private API:** `torch._C._set_tensor_metadata()` is an internal PyTorch API. It may change or be removed in future PyTorch versions, requiring a different approach.
- **trimesh mesh operations are CPU-bound:** Base.py's `fill_holes()` and `simplify_with_cumesh()` now do GPU→CPU→GPU roundtrips. Acceptable for mesh sizes up to ~15M faces.

## VRAM Profile (v3.3 workflow)

| Stage | Peak VRAM | Notes |
|---|---|---|
| Generator | ~6 GB | Auto-unloaded after completion |
| PostProcess2 | 0 GB | CPU trimesh only |
| UnloadAllModels | 3-5 GB | Residual mesh data from MVTT conversion |
| MeshRefiner | 10-15 GB | Loads/unloads SLAT models internally |
| After cleanup | 0 GB | Pipeline unloaded, mesh offloaded to CPU |
