# GPU tier list — design

## Purpose

Provide one self-contained web page for selecting a GPU to train Murmur. It must make the distinction between verified Mamba3 MIMO support, a hardware-dependent hypothesis, and a known failure explicit.

## Deliverable

`docs/gpu_tierlist.html` containing all HTML, CSS, JavaScript and source links. It must open locally in a browser without a build step or network dependency.

## Data and scope

Rows cover: Tesla T4, RTX A5000, RTX 4000 Ada, RTX A6000, RTX 6000 Ada, RTX 4090, RTX 5090, RTX PRO 4500, RTX PRO 6000, RTX PRO 6000 WK, A40, L40, L40S, A100 PCIe 80 GB, A100 SXM 80 GB, H100 PCIe 80 GB, H100 SXM 80 GB, H100 NVL, H200 SXM, H200 NVL, B200, B300 and B300 MIG 34 GB.

The table records the user-provided hourly price, VRAM, maximum GPU count, system RAM and vCPU count where supplied. Unknown values are shown as an em dash and sort after known values.

MIMO labels use exactly these meanings:

- **Verified upstream**: the source explicitly names the hardware or its Mamba3 MIMO path.
- **Gate required**: architecture is compatible in principle, but neither the Mamba3 MIMO project nor this project has a passing result on the exact GPU.
- **Blocked here**: this project observed a failure. This is not a claim that the underlying GPU can never work.

The T4 row is labelled `Blocked here`: the Kaggle execution failed while importing TileLang/TVM-FFI before the MIMO kernel ran.

## Interaction

- Clicking any column header toggles ascending/descending sorting.
- Top buttons filter rows by MIMO label.
- A search field filters by GPU name, architecture and recommendation.
- The page displays an explanatory note that MIMO status is a compatibility claim, not a throughput benchmark.

## Presentation

The visual design is a compact dark research dashboard with a high-contrast table. A recommendation column makes the preferred role clear: smoke-only, 350M GQA/SISO, MIMO gate, or full MIMO training.

## Validation

- HTML parses successfully.
- Sorting works for textual, numeric and missing-price fields.
- Each source link opens to an authoritative NVIDIA, Mamba or TileLang page.
- The status wording does not claim an untested GPU is confirmed to run Mamba3 MIMO.
