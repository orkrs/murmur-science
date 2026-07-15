"""Contract tests for autonomous RTX 4090 MIMO launchers."""

from pathlib import Path


ROOT = Path(__file__).parents[2]
SMOKE = ROOT / "scripts" / "run_rtx4090_mimo_smoke.sh"
TRAINING = ROOT / "scripts" / "run_rtx4090_mimo_350m.sh"
SETUP = ROOT / "scripts" / "setup_rtx4090_mamba_env.sh"
GATE = ROOT / "scripts" / "mamba_gate.py"
MEMORY_GATE = ROOT / "scripts" / "model_memory_gate.py"
SMOKE_CONFIG = ROOT / "configs" / "rtx4090_mimo_40m_smoke.toml"
TRAINING_CONFIG = ROOT / "configs" / "rtx4090_mimo_350m.toml"


def test_rtx4090_smoke_launcher_is_autonomous_and_gated():
    """Smoke installs its isolated stack, selects a valid MIMO kernel, then trains."""
    source = SMOKE.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    config = SMOKE_CONFIG.read_text(encoding="utf-8")

    assert 'mixer = "mamba3_mimo"' in config
    assert "mamba_chunk_size = 16" in config
    assert "setup_rtx4090_mamba_env.sh" in source
    assert "require_free_space_gb 40" in source
    assert "mamba_gate.py" in source and "--profile rtx4090_smoke" in source
    assert "selected_chunk_size" in source
    assert "scripts/train.py" in source
    assert "torch==2.9.0" in setup
    assert "cu128" in setup
    assert "cloudpickle" in setup
    assert "tilelang==0.1.10" in setup
    assert "apache-tvm-ffi==0.1.10" in setup


def test_rtx4090_350m_launcher_runs_memory_gate_before_data_download():
    """The costly corpus build occurs only after model memory has passed."""
    source = TRAINING.read_text(encoding="utf-8")
    config = TRAINING_CONFIG.read_text(encoding="utf-8")

    assert 'mixer = "mamba3_mimo"' in config
    assert "max_tokens = 7500000000" in config
    assert "mamba_chunk_size = 16" in config
    assert "mamba_gate.py" in source and "--profile rtx4090_350m" in source
    assert "require_free_space_gb 350" in source
    assert "model_memory_gate.py" in source
    assert source.index("model_memory_gate.py") < source.index("build_hf_mix.py")
    assert "--profile mixed_350m" in source
    assert "--max-tokens 7500000000" in source
    assert "--resume" in source


def test_mimo_gate_and_full_model_memory_gate_are_present():
    """Both hardware gates produce machine-readable results for launchers."""
    gate = GATE.read_text(encoding="utf-8")
    memory = MEMORY_GATE.read_text(encoding="utf-8")

    assert "selected_chunk_size" in gate
    assert "loss.backward()" in gate
    assert "peak_vram_gb" in gate
    assert '"status": "passed"' in memory
    assert "optimizer.step()" in memory
