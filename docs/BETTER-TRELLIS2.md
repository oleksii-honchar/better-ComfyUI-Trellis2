# better-ComfyUI-Trellis2

## Purpose

A maintained fork of [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) that patches critical VRAM OOM issues on high-end GPUs (RTX 5090, 32 GB) when running the TRELLIS.2 3D generation pipeline:

1. **VRAM OOM at ~28 GB peak** — The PyTorch CUDA allocator `expandable_segments:True` inflates all GPU allocations by ~12 GB beyond actual model weight usage, pushing the peak from ~16 GB to ~28 GB on a 32 GB GPU.
2. **"Tensor without storage" crashes** — `torch.cuda.empty_cache()` called during cumesh mesh operations reclaims GPU memory owned by cumesh, causing silent GPU crashes.
3. **No deterministic memory cleanup** — Pipeline models stay in CPU RAM after generation with no way for users to explicitly free them, causing resource contention when switching workflows.

This fork applies CPU offload fixes, allocator tuning, and cumesh synchronization that bring peak VRAM from ~28 GB to ~12 GB — a **~57% reduction** — enabling reliable Trellis2 generation on 32 GB GPUs.

---

## Features

The fork adds several fixes to address VRAM OOM, cumesh crashes, and memory management:

- **Skip `pipeline.cuda()` when `low_vram=true`** — Prevents eager loading of all models to GPU at load time. Models load lazily on-demand via `load_*()` methods with proper `.to(device)` / `.cpu()` wrapping. [See detailed spec](./spec/01-skip-pipeline-cuda-low-vram.md)
- **Replace `expandable_segments:True`** — Eliminates ~12 GB VRAM inflation from PyTorch CUDA allocator segment expansion. Replaced with `max_split_size_mb:128,garbage_collection_threshold:0.6` for tight allocation. [See detailed spec](./spec/02-replace-expandable-segments.md)
- **Remove `empty_cache()` from `reset_cuda()`** — Prevents cumesh memory reclamation crashes ("Tensor without storage") during mesh post-processing. [See detailed spec](./spec/03-remove-empty-cache-reset-cuda.md)
- **Add `torch.cuda.synchronize()` around cumesh ops** — 9 synchronization points across 3 cumesh methods (init/compute/read) prevent "Tensor without storage" errors from `empty_cache()` reclaiming cumesh-owned memory. [See detailed spec](./spec/04-synchronize-cumesh-operations.md)
- **`unload_all()` method** — Deterministic model unloading from GPU and CPU memory, called via the new `Trellis2UnloadModels` node. [See detailed spec](./spec/05-unload-all-method-and-node.md)
- **Cascade sequential model loading (P1)** — Cascade pipeline paths load shape flow models on-demand instead of pre-loading both, preventing ~4-6 GB double-load spikes. [See detailed spec](./spec/06-cascade-sequential-loading.md)
- **cudaMallocAsync vs cumesh conflict (P0)** — Blackwell GPUs force-enable `cudaMallocAsync`, causing use-after-free races with cuMesh. Resolved via `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. [See detailed spec](./spec/07-cudamallocasync-cumesh-conflict.md)
- **Allocator crossroads on Blackwell (P0)** — Investigation of `backend:native` vs `NO_CUDA_MEMORY_CACHING` approaches. [See detailed spec](./spec/08-allocator-crossroads-blackwell.md)
- **Trellis2 `unload_all()` VRAM leak fix (P0)** — Extended model unloading to catch accelerate-managed models bypassing ComfyUI tracking. [See detailed spec](./spec/09-trellis2-unload-all-investigation.md)
- **PyTorch 2.11 + Blackwell full compatibility (P0)** — 5-layer fix: cudaMallocAsync, cuMesh replacements, o_voxel stride patch, SafeBVH fallback, cuMesh-free workflow. [See detailed spec](./spec/10-pytorch-211-blackwell-compatibility.md)

See **[FEATURES.md](./FEATURES.md)** for detailed descriptions and configuration.

---

## Installation

### For ComfyUI Custom Node (Docker)

This fork replaces the `ComfyUI-Trellis2` custom node in your ComfyUI installation. If using Docker (e.g., `mmartial/comfyui-nvidia-docker`):

```bash
# 1. Clone the fork into ComfyUI custom_nodes
cd /path/to/comfyui/custom_nodes
git clone https://github.com/oleksii-honchar/better-ComfyUI-Trellis2.git

# 2. Replace the existing node symlink/copy
#    (or remove the old one and use this)

# 3. Install wheels (Torch version-dependent)
python -m pip install better-ComfyUI-Trellis2/wheels/Linux/Torch280/*.whl

# 4. Install requirements
python -m pip install -r better-ComfyUI-Trellis2/requirements.txt

# 5. Restart ComfyUI container
docker restart <container>
```

### For ComfyUI Portable (Windows)

```bash
# 1. Clone the fork
cd ComfyUI\custom_nodes
git clone https://github.com/oleksii-honchar/better-ComfyUI-Trellis2.git

# 2. Install wheels
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\better-ComfyUI-Trellis2\wheels\Windows\Torch280\cumesh-1.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\better-ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\better-ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\better-ComfyUI-Trellis2\wheels\Windows\Torch280\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\better-ComfyUI-Trellis2\wheels\Windows\Torch280\o_voxel-0.0.1-cp311-cp311-win_amd64.whl

# 3. Install requirements
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\better-ComfyUI-Trellis2\requirements.txt

# 4. Restart ComfyUI
```

### From Source (Custom Build)

See the [upstream README](https://github.com/visualbruno/ComfyUI-Trellis2) for custom build instructions (o_voxel, Cumesh, FlexGEMM).

---

## Usage

### Workflow Configuration

The fork is **drop-in compatible** — no workflow changes required. Existing workflows work without modification. Key settings:

- **`low_vram=true`** — Required for the P0.1 fix to take effect (models load lazily)
- **`keep_models_loaded=false`** — Recommended for maximum VRAM savings (models `del`-ed after each phase)
- **`pipeline_type="512"`** — Simple path; cascade paths also benefit from P1 sequential loading

### New Node: `Trellis2UnloadModels`

Add this node at the end of a Trellis2 workflow to explicitly free all memory:

```
Trellis2MeshWithVoxelAdvancedGenerator → Trellis2UnloadModels
```

This node is optional — the pipeline cleans up between phases automatically. Use it when switching to non-Trellis2 workflows or when memory is constrained.

### VRAM Monitoring

Verify the VRAM reduction with `nvidia-smi`:

```bash
# Terminal 1: monitor VRAM
watch -n 1 nvidia-smi

# Terminal 2: run a Trellis2 generation
# Expected: ~12 GB peak (vs ~28 GB before)
```

---

## Runbook: Making Changes and Syncing with Source

For detailed governance procedures — fork structure, syncing with upstream, making changes, and pushing — see **[GOVERNANCE.md](./GOVERNANCE.md)**.

The governance document covers:
- Fork structure and branch conventions
- Upstream sync procedures (with conflict resolution)
- Making VRAM/stability fixes in the fork
- Push procedures and verification

---

## Compatibility

This fork is **backward-compatible** with ComfyUI-Trellis2. All patches use existing code paths — no new dependencies or breaking changes:

- **Node class names:** Unchanged. Existing workflows reference the same class names.
- **Workflow JSON:** No changes needed. All existing workflows load and run without modification.
- **Dependencies:** Same as upstream. No new pip packages required.
- **API compatibility:** `pipeline.cuda()`, `reset_cuda()`, and `unload_*()` methods retain their signatures.

The only new addition is the `Trellis2UnloadModels` node (optional) and `pipeline.unload_all()` method (new, backward-compatible).

---

## Known Limitations (Follow-up)

| Issue | Severity | Description |
|-------|----------|-------------|
| H1 | LOW | `unload_all()` calls `_cleanup_cuda()` redundantly (N+1 times). Safe but inefficient. |
| M2 | LOW | `unload_all()` fallback loop does `del` then `= None` — `del` is unnecessary. Style issue. |

---

## Deferred Features

- **Activation offload during denoising** — The upstream Microsoft pipeline achieves ~2-3 GB VRAM by offloading model activations during diffusion steps. Our fork's per-operation offload reaches ~12 GB. Further reduction would require hooking into the sampler's step loop to offload activations between steps. (Deferred — complex, requires sampler instrumentation.)
- **Docker image** — Pre-built Docker image with the fork as a custom node. (Deferred — user manages their own Docker setup.)

---

## License

MIT — Same as upstream ComfyUI-Trellis2.
