import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_smoke_notebooks_share_english_profile_but_keep_mixers_separate():
    gqa = _source("kaggle_t4_smoke.ipynb")
    siso = _source("kaggle_t4_mamba3_siso_40m_smoke.ipynb")
    mimo = _source("rtx6000_ada_mamba3_mimo_40m_smoke.ipynb")
    for source in (gqa, siso, mimo):
        assert "--profile" in source and "english_smoke" in source
        assert "Add Data" not in source
    assert "config.model.mixer=='gqa'" in gqa
    assert "config.model.mixer=='mamba3'" in siso
    assert "mamba3_mimo" in mimo


def test_350m_notebook_uses_mixed_profile_and_gqa_config():
    source = _source("kaggle_t4_350m_training.ipynb")
    assert "--profile" in source and "mixed_350m" in source
    assert "7_500_000_000" in source
    assert "config.model.mixer == 'gqa'" in source
    assert "Add Data" not in source
