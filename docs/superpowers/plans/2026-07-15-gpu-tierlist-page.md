# GPU Tier List Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one locally usable, sortable web page for comparing GPU suitability and Mamba3 MIMO compatibility for Murmur training.

**Architecture:** A single HTML file embeds a typed JavaScript data array, CSS and table rendering logic. It is accompanied by a Python source-level test that verifies the page contains all supplied priced GPUs, MIMO labels and client-side sorting hooks; no web framework or build system is introduced.

**Tech Stack:** HTML5, CSS, browser JavaScript, Python `pytest`.

## Global Constraints

- Create exactly one deployable web-page file: `docs/gpu_tierlist.html`.
- Use no CDN, package manager, build step, external fonts or runtime network requests.
- Treat user-supplied price and instance fields as provided data, not independently verified market quotes.
- MIMO status must be only `Verified upstream`, `Gate required` or `Blocked here`.
- A row may be `Verified upstream` only when the upstream documentation explicitly tests the exact card or a source explicitly documents that exact Mamba3 MIMO configuration.
- The T4 row must remain `Blocked here` and state that the failure happened during TileLang/TVM-FFI import on Kaggle.

---

### Task 1: Create the self-contained sortable comparison page

**Files:**
- Create: `docs/gpu_tierlist.html`
- Test: `tests/integration/test_gpu_tierlist_page.py`

**Interfaces:**
- Consumes: User-provided GPU rental records and the MIMO status definitions in `docs/superpowers/specs/2026-07-15-gpu-tierlist-design.md`.
- Produces: A file opened directly in a browser, exposing `GPU_ROWS`, `renderRows(rows)` and `sortRows(key)` in its inline JavaScript.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


PAGE = Path(__file__).parents[2] / "docs" / "gpu_tierlist.html"


def test_gpu_tierlist_contains_sortable_data_and_required_rows() -> None:
    content = PAGE.read_text(encoding="utf-8")
    assert "const GPU_ROWS =" in content
    assert "function sortRows(key)" in content
    assert "RTX 6000 Ada" in content
    assert "A40" in content
    assert "B300" in content
    assert "$0.77/hr" in content
    assert "$0.44/hr" in content
    assert "Blocked here" in content
    assert "Gate required" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/integration/test_gpu_tierlist_page.py -v`

Expected: `FAIL` because `docs/gpu_tierlist.html` does not exist.

- [ ] **Step 3: Implement the page**

Create `docs/gpu_tierlist.html` with:

```html
<script>
const GPU_ROWS = [/* supplied records with numeric price and VRAM fields */];
function sortRows(key) { /* toggle numeric or locale string sort, then rerender */ }
function renderRows(rows) { /* create tbody rows and status badges */ }
</script>
```

Include a sticky sortable table header for GPU, generation, hourly price, VRAM, max GPUs, system RAM, vCPU, MIMO status and recommended Murmur role. Include a text search, MIMO-status filter buttons, source links and an explicit data provenance note.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/integration/test_gpu_tierlist_page.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add docs/gpu_tierlist.html tests/integration/test_gpu_tierlist_page.py
git commit -m "Add sortable GPU training tier list"
```
