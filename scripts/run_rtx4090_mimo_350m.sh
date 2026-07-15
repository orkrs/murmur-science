#!/usr/bin/env bash
# Install, prove real 350M MIMO training memory, then build data and train/resume.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
require_free_space_gb() {
  local required="$1"
  local available
  available="$(df --output=avail -BG "$ROOT" | tail -n 1 | tr -dc '0-9')"
  if (( available < required )); then
    echo "Need at least ${required} GB free; only ${available} GB is available at $ROOT." >&2
    exit 3
  fi
}
require_free_space_gb 350
./scripts/setup_rtx4090_mamba_env.sh
PY="${VENV_DIR:-$ROOT/.venv-rtx4090}/bin/python"
OUT="artifacts/rtx4090_mimo_350m"
GATE="$OUT/mimo_gate.json"
CONFIG="$OUT/session.toml"
RUN="artifacts/runs/rtx4090_mimo_350m"
mkdir -p "$OUT"
"$PY" scripts/hardware_probe.py --output "$OUT/hardware.json"
"$PY" scripts/mamba_gate.py --profile rtx4090_350m --output "$GATE"
selected_chunk_size="$($PY - "$GATE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "passed", payload
print(payload["selected_chunk_size"])
PY
)"
cp configs/rtx4090_mimo_350m.toml "$CONFIG"
sed -i "s/^mamba_chunk_size = .*/mamba_chunk_size = $selected_chunk_size/" "$CONFIG"
"$PY" scripts/model_memory_gate.py --config "$CONFIG" --output "$OUT/model_memory_gate.json"
if [[ ! -f artifacts/rtx4090_mixed_350m_corpus/train.jsonl ]]; then
  "$PY" scripts/build_hf_mix.py --profile mixed_350m --output artifacts/rtx4090_mixed_350m_corpus --max-tokens 7500000000
fi
if [[ ! -f artifacts/rtx4090_mixed_350m_tokenizer.model ]]; then
  "$PY" scripts/train_tokenizer.py --corpus artifacts/rtx4090_mixed_350m_corpus/corpus.txt --output artifacts/rtx4090_mixed_350m_tokenizer.model --vocab-size 48000
fi
if [[ ! -e artifacts/rtx4090_mixed_350m_data/train.bin ]]; then
  "$PY" scripts/prepare_data.py --config "$CONFIG" --tokenizer artifacts/rtx4090_mixed_350m_tokenizer.model --train-input artifacts/rtx4090_mixed_350m_corpus/train.jsonl --val-input artifacts/rtx4090_mixed_350m_corpus/val.jsonl --output artifacts/rtx4090_mixed_350m_data
fi
"$PY" scripts/param_count.py --config "$CONFIG"
resume_args=()
if [[ -d "$RUN/checkpoints/last" ]]; then
  resume_args=(--resume "$RUN/checkpoints/last")
fi
"$PY" scripts/train.py --config "$CONFIG" --run-dir "$RUN" --device cuda "${resume_args[@]}"
