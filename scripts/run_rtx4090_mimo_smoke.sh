#!/usr/bin/env bash
# Install, prove MIMO backward on this RTX 4090, then run the 40M smoke training.
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
require_free_space_gb 40
./scripts/setup_rtx4090_mamba_env.sh
PY="${VENV_DIR:-$ROOT/.venv-rtx4090}/bin/python"
OUT="artifacts/rtx4090_mimo_smoke"
GATE="$OUT/mimo_gate.json"
CONFIG="$OUT/session.toml"
mkdir -p "$OUT"
"$PY" scripts/hardware_probe.py --output "$OUT/hardware.json"
"$PY" scripts/mamba_gate.py --profile rtx4090_smoke --output "$GATE"
selected_chunk_size="$($PY - "$GATE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "passed", payload
print(payload["selected_chunk_size"])
PY
)"
cp configs/rtx4090_mimo_40m_smoke.toml "$CONFIG"
sed -i "s/^mamba_chunk_size = .*/mamba_chunk_size = $selected_chunk_size/" "$CONFIG"
if [[ ! -f artifacts/rtx4090_english_smoke_corpus/train.jsonl ]]; then
  "$PY" scripts/build_hf_mix.py --profile english_smoke --output artifacts/rtx4090_english_smoke_corpus --max-tokens 2400000
fi
if [[ ! -f artifacts/rtx4090_english_smoke_tokenizer.model ]]; then
  "$PY" scripts/train_tokenizer.py --corpus artifacts/rtx4090_english_smoke_corpus/corpus.txt --output artifacts/rtx4090_english_smoke_tokenizer.model --vocab-size 32000
fi
if [[ ! -e artifacts/rtx4090_english_smoke_data/train.bin ]]; then
  "$PY" scripts/prepare_data.py --config "$CONFIG" --tokenizer artifacts/rtx4090_english_smoke_tokenizer.model --train-input artifacts/rtx4090_english_smoke_corpus/train.jsonl --val-input artifacts/rtx4090_english_smoke_corpus/val.jsonl --output artifacts/rtx4090_english_smoke_data
fi
"$PY" scripts/param_count.py --config "$CONFIG"
"$PY" scripts/train.py --config "$CONFIG" --run-dir artifacts/runs/rtx4090_mimo_40m_smoke --device cuda
