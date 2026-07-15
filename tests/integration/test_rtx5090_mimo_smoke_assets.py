import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "rtx5090_mamba3_mimo_40m_smoke.ipynb"
CONFIG = ROOT / "configs" / "rtx5090_mimo_40m_smoke.toml"


def test_rtx5090_mimo_notebook_pins_blackwell_training_stack_and_own_branch():
    config = CONFIG.read_text(encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'mixer = "mamba3_mimo"' in config
    assert "fp16 = false" in config and "bf16 = true" in config
    assert "codex/rtx5090-mimo-smoke" in source
    assert "tilelang==0.1.9" in source
    assert "2.9.0+cu130" in source
    assert "d_model=896" in source
    assert "d_state=128" in source
    assert "mimo_rank=4" in source
    assert "chunk_size=16" in source
    assert "d_model=1792" in source
    assert "mimo_rank=2" in source
    assert "english_smoke" in source
    assert "Add Data" not in source
