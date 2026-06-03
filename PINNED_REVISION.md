# PINNED_REVISION

This fork is based on `ComfyUI-Trellis2` at commit **`438fe4e`** ("Added node").

## Upstream

- **Original repo:** https://github.com/visualbruno/ComfyUI-Trellis2
- **Fork:** https://github.com/oleksii-honchar/better-ComfyUI-Trellis2
- **Pinned upstream commit:** `438fe4e`
- **Fork HEAD (before this refactor):** `438fe4e`

## Changes in this fork

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes made in this fork.

### Quick Summary

| Change | File | Impact |
|--------|------|--------|
| P0.1: Skip `pipeline.cuda()` when `low_vram=true` | `nodes.py` | Enforces lazy loading |
| P0.2: Replace `expandable_segments:True` | `nodes.py` | ~12 GB VRAM reduction |
| P0.3: Remove `empty_cache()` from `reset_cuda()` | `nodes.py` | Prevents cumesh crashes |
| P0.4: Add `synchronize()` around cumesh ops | `mesh/base.py` | Prevents "Tensor without storage" |
| P2.1: Add `unload_all()` to pipeline | `trellis2_image_to_3d.py` | Deterministic unloading |
| P2.2: Add `Trellis2UnloadModels` node | `nodes.py` | Explicit memory cleanup |

## Syncing with upstream

To sync with upstream changes:

```bash
# Add upstream remote (if not already added)
git remote add upstream https://github.com/visualbruno/ComfyUI-Trellis2.git

# Fetch and rebase
git fetch upstream
git rebase upstream/main
```

**Warning:** Rebase may introduce conflicts if upstream modifies the same files (`nodes.py`, `mesh/base.py`). Resolve conflicts carefully, preserving the fork's changes.
