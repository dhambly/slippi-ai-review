# Portable Installation

The review stack is operated from this repository. `dependencies.lock.json`
pins Phillip and both MSL backends to exact commits. Linux x86_64 clones already
contain checksum-pinned MSL wheels under `vendor/runtimes/`, so setup does not
need another MSL checkout or compiler. Bootstrap unpacks each backend into
`.runtime/`, fetches Phillip at its pinned public commit, installs Node,
extracts legal simulator data, writes the local configuration, and runs an
import smoke test.

macOS and Linux arm64 use the same command and automatically build the locked
MSL revisions from source. Those managed source trees live under `.deps/` and
do not require manual coordination.

The model, ISO, uploaded replays, and generated reviews remain outside Git.
The source fallback uses the private MSL fork. Authenticate Git before setup on
macOS, Linux arm64, or when explicitly passing `--from-source`:

```bash
gh auth login
gh auth setup-git
git ls-remote https://github.com/dhambly/melee-sim-light.git HEAD
```

The final command must return a commit. The bootstrap disables interactive Git
prompts so an unauthenticated service install fails immediately instead of
hanging.

## Supported Hosts

- Linux x86_64. A C11 compiler is only needed on arm64 or with `--from-source`.
- macOS 12 or newer with Xcode Command Line Tools. Apple Silicon uses TensorFlow 2.21; Intel macOS uses the
  final Intel-compatible TensorFlow 2.16.2 runtime.
- Python 3.12, installed by `uv`.
- CPU inference is the portable baseline.

The dashboard and interactive MSL viewer work natively on both platforms.
Training Mode CE export/playback and Dolphin rendering remain optional
Windows-only integrations.

## Linux

Install host tools on Debian or Ubuntu (`build-essential` is optional on
x86_64 and required for source fallback):

```bash
sudo apt update
sudo apt install -y build-essential curl git gh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Clone or unpack the application, then bootstrap:

```bash
git clone <slippi-ai-review-repository-url> slippi-ai-review
cd slippi-ai-review
sh ./scripts/bootstrap-posix.sh \
  --model /absolute/path/to/gm-v2 \
  --iso /absolute/path/to/GALE01.iso
source .venv/bin/activate
slippi-review doctor
slippi-review serve --host 127.0.0.1 --port 8877
```

For NVIDIA acceleration, first establish a working CPU installation. Then
follow TensorFlow's current Linux driver requirements and install the CUDA
extra into this environment:

```bash
source .venv/bin/activate
python -m pip install "tensorflow[and-cuda]==2.21.0"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## macOS

Install Apple's compiler tools and `uv`:

```bash
xcode-select --install
brew install gh
gh auth login
gh auth setup-git
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Clone or unpack the application, then bootstrap:

```bash
git clone <slippi-ai-review-repository-url> slippi-ai-review
cd slippi-ai-review
sh ./scripts/bootstrap-posix.sh \
  --model /absolute/path/to/gm-v2 \
  --iso /absolute/path/to/GALE01.iso
source .venv/bin/activate
slippi-review doctor
slippi-review serve --host 127.0.0.1 --port 8877
```

CPU inference is the default on macOS. `tensorflow-metal` is optional and
should only be enabled after comparing its outputs against the CPU baseline.

## Remote Access

Bind to a private network interface:

```bash
slippi-review serve --host 0.0.0.0 --port 8877 --allow-remote --no-open
```

Public binding requires both `--allow-public` and an access token. Tailscale or
an SSH tunnel is preferred to exposing the service directly.

## Update Or Repair

The bootstrap is idempotent for pinned runtime artifacts and dependency
checkouts:

```bash
sh ./scripts/bootstrap-posix.sh \
  --model /absolute/path/to/gm-v2 \
  --iso /absolute/path/to/GALE01.iso
```

It refuses to reset unexpected or hand-edited runtime/dependency directories.
Move the affected `.runtime/<name>` or `.deps/<name>` directory aside and rerun
when a clean rebuild is intended. To audit the source path on any host, add
`--from-source` to the bootstrap command.

To validate code packaging without private game assets:

```bash
mkdir -p /tmp/empty-model
touch /tmp/empty.iso
sh ./scripts/bootstrap-posix.sh \
  --model /tmp/empty-model \
  --iso /tmp/empty.iso \
  --skip-data \
  --dev
python -m pytest
```

To create a transferable source archive from the current worktree:

```bash
python scripts/build_source_bundle.py
```

The archive is written to `dist/slippi-ai-review-source.tar.gz`.
