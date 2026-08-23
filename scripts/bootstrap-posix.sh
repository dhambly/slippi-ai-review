#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  printf '%s\n' "git is required." >&2
  exit 1
fi

uv python install 3.12
uv venv --python 3.12 --seed "$ROOT/.venv"
"$ROOT/.venv/bin/python" "$ROOT/scripts/bootstrap.py" "$@"
