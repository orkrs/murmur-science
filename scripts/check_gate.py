"""Apply objective checks before allowing a prototype run."""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    required = ("cache_parity", "resume", "finite", "manifest")
    reasons = [f"missing or failed: {key}" for key in required if not evidence.get(key, False)]
    decision = {"allowed": not reasons, "reasons": reasons, "evidence": evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
