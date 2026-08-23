# Slippi AI Review

Slippi AI Review finds attacker advantage opportunities and recorded neutral
losses in a `.slp`, samples counterfactual continuations with Phillip and
melee-sim-light, and publishes linked interactive reports for both phases.
Selected advantage routes can also be exported to Training Mode Community
Edition for hands-on practice.

This repository contains the application plus checksum-pinned MSL runtime
wheels for Linux x86_64 and macOS on Apple Silicon and Intel. Models, game
images, replays, generated reports, upstream source checkouts, and extracted
emulator data are not vendored.

## Distribution

[`dependencies.lock.json`](dependencies.lock.json) pins the exact tested MSL
and `slippi-ai` revisions. Supported Linux and macOS hosts install both MSL
backends from the verified wheels already in this repository. Unsupported
architectures can build the same pinned revisions automatically into
`.runtime/`; users do not manage MSL checkouts.

The automated native setup supports:

- Linux x86_64 with bundled runtimes; Linux arm64 uses source fallback
- macOS 12+ on Apple Silicon or Intel with bundled runtimes
- Python 3.12 and Node.js 20, installed locally by the bootstrap
- CPU inference on both operating systems; Linux NVIDIA acceleration is optional
- a Phillip-compatible model such as `gm-v2`
- a legally obtained NTSC Melee 1.02 ISO
- Fox, Falco, Marth, Sheik, Captain Falcon, Ganondorf, Jigglypuff, Pikachu,
  Samus, Peach, and Yoshi replays

Models, game images, replay uploads, generated reviews, and extracted game data
are never vendored.

## Linux And macOS

After installing `git` and `uv`:

```bash
git clone <repository-url> slippi-ai-review
cd slippi-ai-review
sh ./scripts/bootstrap-posix.sh \
  --model /absolute/path/to/gm-v2 \
  --iso /absolute/path/to/GALE01.iso
source .venv/bin/activate
slippi-review doctor
slippi-review serve --host 127.0.0.1 --port 8877
```

The bootstrap verifies and unpacks the bundled MSL runtimes when available,
installs Phillip and Node, extracts MSL data, writes
`~/.slippi-ai-review.toml`, and runs a smoke test. Full host preparation, GPU
notes, source fallback behavior, and copy-paste commands are in
[Portable Installation](docs/portable-install.md).

## Windows

Windows remains supported through the existing WSL/micromamba runtime. Set
`runtime.mode = "wsl"` and fill in the `[wsl]` section of
[`config.example.toml`](config.example.toml). The portable bootstrap is
deliberately native-POSIX only.

## Run

```bash
slippi-review serve --host 0.0.0.0 --port 8877 --allow-remote
```

The server starts its persistent worker automatically. Open
`http://127.0.0.1:8877`, upload a replay, choose the analyzed player, and choose
Quick, Standard, or Deep sampling.

For localhost-only use, omit `--host` and `--allow-remote`. Public binding is
disabled unless `--allow-public` is combined with an access token. Tailscale is
the recommended remote-access path.

Install the dashboard at Windows login:

```bash
slippi-review startup install --port 8877
```

`startup` currently manages Windows login startup only. On Linux and macOS,
launch the server manually or use the host's service manager.

## Direct Analysis

```bash
slippi-review analyze \
  --replay "/path/to/game.slp" \
  --controlled-port 1 \
  --job-dir "./data/manual-review"
```

Each job is resumable at stage boundaries. Its `artifacts/advantage_review.html`
is the stable entry point, with a linked `artifacts/neutral_review.html` for
reliable neutral-loss avoidance options.

Neutral simulation is CPU-only by default so it can coexist with other MSL
workloads. Pass `--neutral-enable-gpu` to `analyze` or `worker` on a dedicated
machine to accelerate it.

## Data And Cleanup

Runtime data lives under `data/` by default and is ignored by Git. Review
metadata, logs, simulation batches, traces, and final artifacts are isolated by
UUID. Cleanup is dry-run by default:

```bash
slippi-review cleanup --days 30
slippi-review cleanup --days 30 --apply
```

Only pipeline intermediates from archived, failed, or cancelled reviews are
eligible. Source replays and final reports are retained.

## Development

```bash
python -m pytest
python -m ruff check src tests
```

See [Architecture](docs/architecture.md) for package boundaries and the job
state model. See [Decomp simulation backend](docs/decomp-backend.md) for the
opt-in backend and parity-audit workflow.

Third-party viewer assets and their provenance are documented in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
