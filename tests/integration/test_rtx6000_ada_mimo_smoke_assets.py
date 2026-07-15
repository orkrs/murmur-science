import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "rtx6000_ada_mamba3_mimo_40m_smoke.ipynb"
CONFIG = ROOT / "configs" / "rtx6000_ada_mimo_40m_smoke.toml"


def test_rtx6000_ada_mimo_smoke_assets_are_self_contained() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'mixer = "mamba3_mimo"' in config
    assert "max_tokens = 2000000" in config
    assert "fp16 = false" in config
    assert "bf16 = true" in config
    assert "codex/rtx6000-ada-mimo-smoke" in source
    assert "english_smoke" in source
    assert "HuggingFaceFW/fineweb-edu" not in source
    assert "tilelang==0.1.9" in source
    assert "download.pytorch.org/whl/cu130" in source
    assert "MAMBA_FORCE_BUILD" in source
    assert "mamba3_mimo" in source
    assert "Add Data" not in source
