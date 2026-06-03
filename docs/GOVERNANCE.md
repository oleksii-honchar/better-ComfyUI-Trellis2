# better-ComfyUI-Trellis2 Governance

This document defines the governance procedures for maintaining the `better-ComfyUI-Trellis2` fork — how to make changes, sync with upstream, and push.

For an overview of the fork's purpose and features, see [BETTER-TRELLIS2.md](./BETTER-TRELLIS2.md).

---

## Fork Structure

```
              upstream/visualbruno/ComfyUI-Trellis2
              ┌────────────────────────────────────┐
              │  main (upstream)                   │◀── upstream target, changes over time
              └────────────────────────────────────┘
                            │
                            │ fork
                            ▼
        oleksii-honchar/better-ComfyUI-Trellis2 (origin)
        ┌────────────────────────────────────────────┐
        │  main (working branch)                     │◀── VRAM fixes + synced with upstream
        │  260603-feat-cpu-offload...                │◀── feature branches off main
        └────────────────────────────────────────────┘
```

**Key rules:**
- **`main`** — Working branch. Contains VRAM/stability fixes + synced with upstream.
- **Feature branches** — Branch off `main`. Rebased onto `main` before merge.
- **`origin`** — Your fork (push target). NOT the original repo.
- **`upstream`** — Original repo (read-only, never push).

---

## Core Principle: Preserve Our Fixes

**Our VRAM fixes are the highest priority.** When resolving any merge or rebase conflict:

1. **Always preserve our fix code** — if a conflict exists between our changes and upstream changes, our fix logic wins.
2. **Adapt to upstream structural changes** — if upstream changed APIs or patterns, adapt our fix code to the new upstream patterns while preserving its behavior.
3. **Never discard our fix changes** — do NOT use `-X theirs` or blindly accept upstream's version when our fix code is involved.
4. **If unsure, keep both** — include both our code and upstream's code, then clean up manually.

This principle applies to:
- Rebase conflicts on `main` onto upstream/main
- Rebase conflicts on feature branches onto `main`

---

## Syncing with Upstream (Staying Current)

**Before making any changes, always sync first:**

```bash
cd ~/www/misc/better-ComfyUI-Trellis2

# 1. Fetch latest from both remotes
git fetch upstream main --quiet
git fetch origin --quiet

# 2. Check divergence
git log --oneline main..upstream/main | wc -l  # commits behind
git log --oneline upstream/main..main | wc -l  # commits ahead

# 3. Rebase main onto upstream/main
git checkout main
git rebase upstream/main

# 4. If conflicts occur, resolve them (preserve our fixes!):
#    - Edit conflicted files — keep our fix code, adapt to upstream changes
#    - git add <resolved-file>
#    - git rebase --continue      (after resolving each conflict)
#    - git rebase --abort         (to cancel)

# 5. Push rebased branch to your fork
git push origin main --force-with-lease
```

---

## Rebase Workflow for Feature Branches

Feature branches are created from `main` and must be rebased onto `main` before they are merged back. This ensures a linear history and that our fixes stay on top of the latest upstream + patches.

### Step 1: Create a feature branch from `main`

```bash
cd ~/www/misc/better-ComfyUI-Trellis2

# Ensure main is up to date (sync with upstream first if needed)
git checkout main
git pull origin main

# Create feature branch
git checkout -b 260603-feat-cpu-offload

# Make your changes...
# ...

# Commit and push
git add .
git commit -m "feat: describe your change"
git push origin 260603-feat-cpu-offload
```

### Step 2: Rebase feature branch onto `main` (before merge)

```bash
cd ~/www/misc/better-ComfyUI-Trellis2

# 1. Make sure main is current
git checkout main
git pull origin main

# 2. Switch to feature branch
git checkout 260603-feat-cpu-offload

# 3. Rebase onto main
git rebase main

# 4. Resolve any conflicts:
#    - Our fix code is the priority — preserve it
#    - Adapt to upstream structural changes if needed
#    - git add <resolved-file>
#    - git rebase --continue

# 5. Force-push the rebased branch
git push origin 260603-feat-cpu-offload --force-with-lease
```

### Step 3: Merge feature branch into `main`

```bash
git checkout main
git merge 260603-feat-cpu-offload --no-ff
git push origin main
```

Use `--no-ff` to preserve the feature branch as a distinct merge commit in history.

---

## Rebase Workflow — Agent Instructions

When an agent (or human) is performing a rebase, follow these steps:

### 1. Read this governance file first

Before resolving any conflicts, read `docs/GOVERNANCE.md` and `docs/FEATURES.md` to understand:
- What fixes exist and their implementation status
- The core principle: **preserve our fixes**

### 2. Understand the rebase direction

- **`main` onto `upstream/main`** — upstream changed, adapt our fixes to new upstream
- **`feature branch` onto `main`** — main moved forward, update feature branch

### 3. Resolve conflicts with fix preservation priority

Apply this decision tree:

1. **Is our code in the conflict?** → Keep our code, adapt to upstream patterns if needed.
2. **Is this a structural change (API, type, pattern)?** → Adapt our code to the new upstream pattern while preserving its behavior.
3. **Is this a doc/spec file for our fix?** → Keep our version.
4. **Is this a shared file where both sides added different things?** → Keep both, merge carefully.
5. **When in doubt** → Keep our code. It's safer to have a conflict to fix later than to lose fix code.

### 4. Verify after rebase

After rebase completes:

```bash
# Check for syntax errors
python -c "import ast; ast.parse(open('nodes.py').read())"
python -c "import ast; ast.parse(open('trellis2/representations/mesh/base.py').read())"
python -c "import ast; ast.parse(open('trellis2/pipelines/trellis2_image_to_3d.py').read())"
```

---

## Pushing Changes to GitHub

**After making local changes:**
```bash
cd ~/www/misc/better-ComfyUI-Trellis2
git push origin main
```

**If you need to force push (after rebase):**
```bash
git push origin main --force-with-lease
```

---

## Recovery Scenarios

| Problem | Solution |
|---------|----------|
| Rebase fails with conflicts | Resolve conflicts (preserve our fixes!), `git rebase --continue` |
| Rebase is too messy to continue | `git rebase --abort` to reset, then resolve manually |
| Accidentally lost our fix code in conflict | Check `git reflog` to recover, or re-apply from feature branch |
| Fork is far behind upstream | Run sync steps above, resolve conflicts iteratively |

---

## Common Mistakes to Avoid

- **Don't push to `upstream`** — `upstream` is read-only. Always push to `origin`.
- **Don't use `-X theirs` with local fixes** — It will discard your changes on conflict.
- **Don't forget to fetch `origin/main`** — Your fork's remote may have updates you haven't seen.
- **Don't merge feature branches without rebasing first** — Always rebase onto `main` before merging.
- **Don't skip syntax check after rebase** — Upstream changes may have broken our fix code silently.

---

## ComfyUI Integration

### Production: Using the Fork as a Custom Node

After cloning the fork into ComfyUI's custom_nodes directory:

```bash
# 1. Clone the fork
cd ComfyUI/custom_nodes
git clone https://github.com/oleksii-honchar/better-ComfyUI-Trellis2.git

# 2. Install wheels (Torch version-dependent)
python -m pip install better-ComfyUI-Trellis2/wheels/Linux/Torch280/*.whl

# 3. Install requirements
python -m pip install -r better-ComfyUI-Trellis2/requirements.txt

# 4. Restart ComfyUI
```

### Docker (mammoth-lan setup)

When using `mmartial/comfyui-nvidia-docker` or similar:

```bash
# 1. Replace the custom node in the container
docker exec <container> bash -c "cd /opt/ComfyUI/custom_nodes && rm -rf ComfyUI-Trellis2 && ln -s /path/to/better-ComfyUI-Trellis2 ComfyUI-Trellis2"

# 2. Reinstall wheels inside container
docker exec <container> bash -c "pip install -f /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch280/*.whl"

# 3. Restart container
docker restart <container>

# 4. Verify VRAM improvement
watch -n 1 nvidia-smi  # Run a Trellis2 generation and verify ~12 GB peak
```

---

## Verification Commands

```bash
# Verify remotes are correct
git remote -v
# origin → oleksii-honchar/better-ComfyUI-Trellis2.git (your fork)
# upstream → visualbruno/ComfyUI-Trellis2.git (original)

# Verify current branch
git branch --show-current
# Should show: main

# Verify divergence
git log --oneline origin/main..main | wc -l  # commits ahead
git log --oneline main..origin/main | wc -l  # commits behind

# Verify Python syntax
python -c "import ast; ast.parse(open('nodes.py').read())" && echo "nodes.py: OK"
python -c "import ast; ast.parse(open('trellis2/representations/mesh/base.py').read())" && echo "base.py: OK"

# Verify VRAM reduction (in ComfyUI running a Trellis2 workflow)
watch -n 1 nvidia-smi  # Peak should be ~12 GB (vs ~28 GB before)
```
