"""Resolve and record the official Mamba repository revision."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default="https://github.com/state-spaces/mamba.git")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "ls-remote", args.repository, "refs/heads/main"], text=True).split()[0]
    if len(commit) != 40:
        raise RuntimeError("upstream did not return a full commit SHA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"repository": args.repository, "commit": commit, "checked_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(commit)
