# 07. cudaMallocAsync / cumesh Memory Conflict

**Status:** ⚠️ Mitigated but not eliminated — **allocator is `native`** (see [08-allocator-crossroads](./08-allocator-crossroads-blackwell.md))  
**Severity:** Critical — causes `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`  
**Platform:** Blackwell GPUs (RTX 50xx) with PyTorch ≥ 2.11  

---

## Problem

PyTorch 2.11+ defaults to `cudaMallocAsync` on Blackwell GPUs. This is a CUDA driver-level async memory allocator that manages GPU memory independently of PyTorch's tensor lifetime tracking.

**cumesh** (`cumesh.CuMesh`) also manages its own CUDA memory internally, outside of PyTorch's control. When `cudaMallocAsync` re-frees memory that cumesh still holds references to, cumesh operations crash with:

```
RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
```

The crash is **non-deterministic** — first run often succeeds, second run fails. This is because `cudaMallocAsync` keeps freed memory in a pool and re-uses it, but doesn't coordinate with cumesh's internal state.

### Crash sites observed

All crashes occur inside cumesh methods called from `base.py`:

| Crash location | Base class method | When |
|---|---|---|
| `cumesh.read()` | `fill_holes()`, `remove_faces()`, `simplify_with_cumesh()` | After `del mesh` frees cumesh storage |
| `cumesh.read_manifold_boundary_adjacency()` | `fill_holes()` | During mesh processing (second run) |

### Why first run succeeds but second run fails

1. **First run:** cumesh allocates fresh GPU memory, does work, `del mesh` releases it.
2. `cudaMallocAsync` keeps the freed memory in its pool.
3. **Between runs:** ComfyUI's `--lowvram` triggers `torch.cuda.empty_cache()`, which tells `cudaMallocAsync` the memory is free for reuse.
4. **Second run:** cumesh allocates GPU memory — `cudaMallocAsync` gives it the pooled memory from step 2.
5. PyTorch (or another component) still thinks it owns that memory and re-frees it during cumesh operations → crash.

---

## Attempted fixes

### ❌ `PYTORCH_CUDA_ALLOC_CONF=backend:native` (REJECTED)

Adding `backend:native` to the env var disables `cudaMallocAsync` entirely, reverting to PyTorch's synchronous allocator.

**Why rejected:** PyTorch 2.11 initializes the allocator backend at library load time, before environment variables are processed. Setting `backend:native` in `docker-compose.yaml` causes:

```
RuntimeError: Allocator backend parsed at runtime != allocator backend parsed at load time,
cudaMallocAsync != native
```

The mmartial/comfyui-nvidia-docker image bakes in `cudaMallocAsync` during ComfyUI's early imports.

### ✅ `torch.cuda.synchronize()` before cumesh operations (DEPLOYED)

Added at the top of `fill_holes()`, `remove_faces()`, and `simplify_with_cumesh()` in `trellis2/representations/mesh/base.py`:

```python
def fill_holes(self, max_hole_perimeter=3e-2):
    torch.cuda.synchronize()  # Flush pending async frees before cumesh allocates
    ...
```

**Why this helps:** `synchronize()` blocks until all pending CUDA operations complete, including `cudaMallocAsync` frees. This ensures cumesh allocates from a clean pool.

**Limitation:** `synchronize()` only helps at method boundaries. If `cudaMallocAsync` frees memory **during** a cumesh operation (while cumesh holds active references), the crash still occurs.

### ✅ `.clone()` after `cumesh.read()` (DEPLOYED)

Added in the same methods:

```python
new_vertices, new_faces = mesh.read()
torch.cuda.synchronize()
new_vertices = new_vertices.clone()  # Detach from cumesh storage
new_faces = new_faces.clone()
del mesh  # Safe — tensors now own their storage
```

**Why this helps:** `cumesh.read()` returns tensors that share storage with the cumesh object. When `del mesh` triggers the cumesh destructor, that storage is freed. `.clone()` creates independently-owned tensors before the destructor runs.

---

## Remaining risk

The two deployed mitigations cover the known crash sites, but `cudaMallocAsync` / cumesh conflicts can occur at **any** cumesh CUDA operation if PyTorch concurrently frees memory. This is a fundamental architectural conflict between two independent GPU memory managers.

### Potential future fixes

1. **Custom Docker image:** Build a ComfyUI image that sets `PYTORCH_CUDA_ALLOC_CONF=backend:native` **before** any PyTorch imports. This requires modifying the `mmartial/comfyui-nvidia-docker` entrypoint or creating a wrapper image.

2. **Environment variable at Docker runtime:** Pass `--env PYTORCH_CUDA_ALLOC_CONF=backend:native,...` directly to `docker run` (not in compose env). The timing might be different.

3. **PyTorch downgrade:** PyTorch 2.10 does not default to `cudaMallocAsync` on Blackwell. But this may break other dependencies (xformers, CUDA 12.8).

4. **cumesh rebuild:** If cumesh could be rebuilt to use PyTorch's allocator instead of its own, the conflict would disappear.

---

## Verification

After each deployment, run the pipeline **twice** in the same ComfyUI session without restart:

1. Queue the Trellis2 v3.1 workflow → should complete successfully.
2. Queue again → should also complete successfully.

If the second run crashes at any cumesh method, the `cudaMallocAsync` conflict is still present and requires deeper investigation.

---

## Related

- [04. Synchronize cumesh operations](./04-synchronize-cumesh-operations.md) — initial fix attempt
- PyTorch issue: `cudaMallocAsync` is enabled by default on Blackwell in PyTorch 2.11
- ComfyUI `--lowvram` flag: triggers `torch.cuda.empty_cache()` between queue runs, accelerating the conflict
