# Slippi AI Review

Slippi AI Review finds attacker advantage opportunities in a `.slp`, samples
counterfactual continuations with Phillip and melee-sim-light, and publishes an
interactive replay-versus-agent report. Selected routes can also be exported to
Training Mode Community Edition for hands-on practice.

This repository contains the application. Models, game images, replays,
generated reports, upstream source checkouts, and local emulator data are not
vendored.

## Supported Setup

- Windows 10/11 with WSL2 and an NVIDIA GPU
- Python 3.11+
- Node.js 20+
- local checkouts of `slippi-ai` and `melee-sim-light`
- a Phillip-compatible model such as `gm-v2`
- a legally obtained NTSC Melee 1.02 ISO
- Fox, Falco, Marth, and Sheik replays

## Install

```powershell
git clone <private-repository-url> slippi-ai-review
cd slippi-ai-review
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
npm install
Copy-Item config.example.toml "$HOME\.slippi-ai-review.toml"
```

Edit `~/.slippi-ai-review.toml`, then verify the machine:

```powershell
slippi-review doctor
```

The current replay bridge expects the legacy-Slippi compatibility changes in
[`patches/melee-sim-light-slippi-2-compat.patch`](patches/melee-sim-light-slippi-2-compat.patch).
Apply it from the configured melee-sim-light checkout when that checkout does
not already contain the changes:

```powershell
cd D:\code\melee-sim-light
git apply --unidiff-zero C:\path\slippi-ai-review\patches\melee-sim-light-slippi-2-compat.patch
```

## Run

```powershell
slippi-review serve --host 0.0.0.0 --port 8877 --allow-remote
```

The server starts its persistent worker automatically. Open
`http://127.0.0.1:8877`, upload a replay, choose the analyzed player, and choose
Quick, Standard, or Deep sampling.

For localhost-only use, omit `--host` and `--allow-remote`. Public binding is
disabled unless `--allow-public` is combined with an access token. Tailscale is
the recommended remote-access path.

Install the dashboard at Windows login:

```powershell
slippi-review startup install --port 8877
```

## Direct Analysis

```powershell
slippi-review analyze `
  --replay "C:\path\game.slp" `
  --controlled-port 1 `
  --job-dir ".\data\manual-review"
```

Each job is resumable at stage boundaries. Its `artifacts/advantage_review.html`
is the stable final report.

## Data And Cleanup

Runtime data lives under `data/` by default and is ignored by Git. Review
metadata, logs, simulation batches, traces, and final artifacts are isolated by
UUID. Cleanup is dry-run by default:

```powershell
slippi-review cleanup --days 30
slippi-review cleanup --days 30 --apply
```

Only pipeline intermediates from archived, failed, or cancelled reviews are
eligible. Source replays and final reports are retained.

## Development

```powershell
python -m pytest
python -m ruff check src tests
```

See [Architecture](docs/architecture.md) for package boundaries and the job
state model.

Third-party viewer assets and their provenance are documented in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
