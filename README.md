# Slippi AI Review

Slippi AI Review finds attacker advantage opportunities and recorded neutral
losses in a `.slp`, samples counterfactual continuations with Phillip and
melee-sim-light, and publishes linked interactive reports for both phases.
Selected advantage routes can also be exported to Training Mode Community
Edition for hands-on practice.

This repository contains the application, the checksum-pinned `gm-v2` model,
and MSL runtime wheels for Linux x86_64 and macOS on Apple Silicon and Intel.
Game images, replays, generated reports, upstream source checkouts, and
extracted emulator data are not vendored.

## Distribution

[`dependencies.lock.json`](dependencies.lock.json) pins the model checksum and
the exact tested MSL and `slippi-ai` revisions. Supported Linux and macOS hosts
install both MSL backends from the verified wheels already in this repository.
Unsupported architectures can build the same pinned revisions automatically
into `.runtime/`; users do not manage MSL checkouts.

The automated native setup supports:

- Linux x86_64 with bundled runtimes; Linux arm64 uses source fallback
- macOS 12+ on Apple Silicon or Intel with bundled runtimes
- Python 3.12 and Node.js 20, installed locally by the bootstrap
- CPU inference on both operating systems; Linux NVIDIA acceleration is optional
- the bundled `gm-v2` model, with an optional `--model` override
- a legally obtained NTSC Melee 1.02 ISO
- Fox, Falco, Marth, Sheik, Captain Falcon, Ganondorf, Jigglypuff, Pikachu,
  Samus, Peach, and Yoshi replays

Game images, replay uploads, generated reviews, and extracted game data are
never committed.

## Linux And macOS

After installing `git` and `uv`:

```bash
git clone <repository-url> slippi-ai-review
cd slippi-ai-review
sh ./scripts/bootstrap-posix.sh \
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

### Complete Private Bundle

The private complete archive contains the application, `gm-v2`, native MSL
runtimes, and a local legal ISO. After extracting it, setup needs no asset
arguments:

On Linux and macOS, run these commands in the normal terminal. On Windows, run
them inside WSL, not PowerShell. A fresh Windows installation must already have
WSL, `git`, and `uv` installed.

```bash
cd slippi-ai-review
sh ./scripts/bootstrap-posix.sh
source .venv/bin/activate
slippi-review serve --host 127.0.0.1 --port 8877
```

Create or refresh that local-only archive with:

```bash
python scripts/build_source_bundle.py \
  --iso /absolute/path/to/GALE01.iso \
  --out dist/slippi-ai-review-complete.tar
```

The complete archive and `local-assets/` are ignored by Git. Never push or
publish an archive containing the ISO.

## Windows

The portable bootstrap does not run directly in PowerShell or `cmd.exe`.
Windows uses WSL: place or extract the repository in the WSL filesystem, open a
WSL terminal, and run the same POSIX commands shown above. Keeping the checkout
under the WSL home directory is strongly recommended because installation on a
mounted Windows drive is much slower.

After setup, start the server from WSL:

```bash
source .venv/bin/activate
slippi-review serve --host 127.0.0.1 --port 8877
```

Open `http://127.0.0.1:8877` in the Windows browser. This server command also
starts the persistent analysis worker. A true fresh-Windows one-command
PowerShell installer is not currently included.

Existing development installations may instead use the older WSL/micromamba
configuration by setting `runtime.mode = "wsl"` and filling in the `[wsl]`
section of [`config.example.toml`](config.example.toml).

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
