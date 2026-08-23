#!/usr/bin/env python3
"""Create the pinned native Linux/macOS Slippi AI Review environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "dependencies.lock.json"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=env, check=True)


def output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd or ROOT, text=True).strip()


def verify_host() -> None:
    if sys.platform not in {"linux", "darwin"}:
        raise SystemExit("This bootstrap is for native Linux and macOS. Windows uses it through WSL.")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 is required; bootstrap is running under {platform.python_version()}.")
    if shutil.which("git") is None:
        raise SystemExit("Missing required executable: git")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_status(checkout: Path, patch: Path) -> str:
    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=checkout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if forward.returncode == 0:
        return "pending"
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=checkout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return "applied" if reverse.returncode == 0 else "conflict"


def prepare_dependency(name: str, spec: dict[str, Any], deps_dir: Path) -> Path:
    checkout = deps_dir / str(spec["directory"])
    commit = str(spec["commit"])
    git_environment = os.environ.copy()
    git_environment["GIT_TERMINAL_PROMPT"] = "0"
    if not checkout.exists():
        checkout.mkdir(parents=True)
        run(["git", "init"], cwd=checkout)
        run(["git", "remote", "add", "origin", str(spec["url"])], cwd=checkout)
    elif not (checkout / ".git").exists():
        raise SystemExit(f"{checkout} exists but is not a Git checkout")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    actual = head.stdout.strip() if head.returncode == 0 else ""
    if not actual:
        try:
            run(
                ["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", commit],
                cwd=checkout,
                env=git_environment,
            )
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Could not fetch dependency {name} from {spec['url']}. "
                "For a private remote, run `gh auth login` and `gh auth setup-git`, then retry."
            ) from exc
        actual = output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != commit:
        raise SystemExit(
            f"{name} is at {actual}, expected {commit}. Move {checkout} aside and rerun bootstrap; "
            "existing checkouts are never reset."
        )

    patch_value = spec.get("patch")
    if patch_value:
        patch = ROOT / str(patch_value)
        expected_hash = str(spec["patchSha256"])
        actual_hash = sha256(patch)
        if actual_hash != expected_hash:
            raise SystemExit(f"Patch checksum mismatch for {patch}: {actual_hash} != {expected_hash}")
        status = patch_status(checkout, patch)
        if status == "pending":
            run(["git", "apply", "--whitespace=error-all", str(patch)], cwd=checkout)
        elif status == "conflict":
            raise SystemExit(f"Cannot safely apply or identify {patch} in {checkout}")
        print(f"{name}: pinned at {commit}; patch {status}", flush=True)
    else:
        print(f"{name}: pinned at {commit}", flush=True)
    return checkout


def runtime_platform_key() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        machine = "arm64"
    system = "macos" if sys.platform == "darwin" else "linux"
    return f"{system}-{machine}-cp{sys.version_info.major}{sys.version_info.minor}"


def _safe_extract_wheel(wheel: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise SystemExit(f"Unsafe path in runtime artifact {wheel}: {member.filename}")
        archive.extractall(destination)


def install_runtime_artifact(
    *,
    name: str,
    artifact: Path,
    expected_hash: str,
    destination: Path,
    identity: dict[str, Any],
) -> Path:
    actual_hash = sha256(artifact)
    if actual_hash != expected_hash:
        raise SystemExit(f"Runtime checksum mismatch for {artifact}: {actual_hash} != {expected_hash}")
    marker_payload = {"name": name, "sha256": expected_hash, **identity}
    marker = destination / ".slippi-review-runtime.json"
    if destination.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing != marker_payload:
            raise SystemExit(
                f"Runtime directory {destination} is not the locked {name} artifact. "
                "Move it aside and rerun bootstrap; existing runtimes are never reset."
            )
        print(f"{name}: using verified runtime {destination}", flush=True)
        return destination

    temporary = destination.with_name(f".{destination.name}.extract-{os.getpid()}")
    if temporary.exists():
        raise SystemExit(f"Temporary runtime directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        _safe_extract_wheel(artifact, temporary)
        (temporary / ".slippi-review-runtime.json").write_text(
            json.dumps(marker_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"{name}: installed verified runtime {destination}", flush=True)
    return destination


def prepare_bundled_runtimes(
    specs: dict[str, dict[str, Any]],
    runtime_dir: Path,
    asset_dir: Path | None,
) -> dict[str, Path]:
    runtimes: dict[str, Path] = {}
    for name in ("legacy", "decomp"):
        spec = specs[name]
        locked_artifact = ROOT / str(spec["path"])
        artifact = (asset_dir / locked_artifact.name) if asset_dir else locked_artifact
        runtimes[name] = install_runtime_artifact(
            name=f"msl-{name}",
            artifact=artifact,
            expected_hash=str(spec["sha256"]),
            destination=runtime_dir / str(spec["directory"]),
            identity={"sourceCommit": str(spec["sourceCommit"]), "platform": runtime_platform_key()},
        )
    return runtimes


def prepare_source_runtimes(
    specs: dict[str, dict[str, Any]],
    deps_dir: Path,
    runtime_dir: Path,
) -> dict[str, Path]:
    if shutil.which("cc") is None:
        raise SystemExit("A C compiler is required for source fallback (build-essential or Xcode tools).")
    runtimes: dict[str, Path] = {}
    for name in ("legacy", "decomp"):
        dependency_name = f"msl-{name}"
        source = prepare_dependency(dependency_name, specs[dependency_name], deps_dir)
        destination = runtime_dir / dependency_name
        marker = destination / ".slippi-review-runtime.json"
        identity = {
            "name": dependency_name,
            "sourceCommit": str(specs[dependency_name]["commit"]),
            "platform": runtime_platform_key(),
            "build": "source",
        }
        if destination.exists():
            try:
                existing = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing != identity:
                raise SystemExit(
                    f"Runtime directory {destination} is not the locked source build. Move it aside and retry."
                )
            runtimes[name] = destination
            continue
        with tempfile.TemporaryDirectory(prefix=f"{dependency_name}-", dir=runtime_dir) as build_directory:
            wheel_dir = Path(build_directory)
            run([
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(source),
            ])
            wheels = list(wheel_dir.glob("*.whl"))
            if len(wheels) != 1:
                raise SystemExit(f"Expected one wheel for {dependency_name}, found {len(wheels)}")
            wheel = wheels[0]
            artifact_hash = sha256(wheel)
            runtimes[name] = install_runtime_artifact(
                name=dependency_name,
                artifact=wheel,
                expected_hash=artifact_hash,
                destination=destination,
                identity={
                    "sourceCommit": str(specs[dependency_name]["commit"]),
                    "platform": runtime_platform_key(),
                    "build": "source",
                },
            )
            # Source builds use a stable identity rather than the nondeterministic wheel hash.
            (destination / ".slippi-review-runtime.json").write_text(
                json.dumps(identity, indent=2) + "\n",
                encoding="utf-8",
            )
        runtimes[name] = destination
    return runtimes


def venv_bin_dir() -> Path:
    return Path(sys.prefix) / "bin"


def venv_executable(name: str) -> Path:
    return venv_bin_dir() / name


def install_node(version: str) -> tuple[Path, Path]:
    node = venv_executable("node")
    npm = venv_executable("npm")
    if not node.is_file() or not npm.is_file():
        run([sys.executable, "-m", "nodeenv", "--python-virtualenv", "--node", version])
    installed = output([str(node), "--version"]).lstrip("v")
    if installed != version:
        raise SystemExit(f"Node version mismatch: installed {installed}, expected {version}")
    return node, npm


def toml_string(value: Path | str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def write_config(
    path: Path,
    *,
    data_dir: Path,
    model: Path,
    msl_root: Path,
    msl_decomp_root: Path,
    slippi_ai_root: Path,
    iso: Path,
) -> None:
    content = (
        "[paths]\n"
        f"data_dir = {toml_string(data_dir)}\n"
        f"model = {toml_string(model)}\n"
        f"msl_root = {toml_string(msl_root)}\n"
        f"msl_decomp_root = {toml_string(msl_decomp_root)}\n"
        f"slippi_ai_root = {toml_string(slippi_ai_root)}\n"
        f"melee_iso = {toml_string(iso)}\n\n"
        "[runtime]\n"
        'mode = "native"\n'
        f"python = {toml_string(Path(sys.executable).absolute())}\n"
        'simulation_backend = "legacy"\n'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def runtime_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(root)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["MSL_DATA_DIR"] = str(root / "data")
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", type=Path, default=os.environ.get("SLIPPI_REVIEW_MELEE_ISO"))
    parser.add_argument("--model", type=Path, default=os.environ.get("SLIPPI_REVIEW_MODEL"))
    parser.add_argument("--deps-dir", type=Path, default=ROOT / ".deps")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".runtime")
    parser.add_argument("--runtime-asset-dir", type=Path, help="Use matching local artifacts for packaging tests.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=Path.home() / ".slippi-ai-review.toml")
    parser.add_argument("--msl-url", help="Override both locked MSL source remotes.")
    parser.add_argument("--slippi-ai-url", help="Override the locked slippi-ai remote.")
    parser.add_argument("--from-source", action="store_true", help="Build both MSL runtimes from pinned source.")
    parser.add_argument("--skip-data", action="store_true", help="Skip ISO extraction for packaging/CI checks.")
    parser.add_argument("--skip-node", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Install test and lint dependencies.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_host()
    if args.model is None:
        raise SystemExit("--model is required (or set SLIPPI_REVIEW_MODEL)")
    if args.iso is None:
        raise SystemExit("--iso is required (or set SLIPPI_REVIEW_MELEE_ISO)")
    model = args.model.expanduser().resolve()
    iso = args.iso.expanduser().resolve()
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")
    if not iso.is_file():
        raise SystemExit(f"ISO not found: {iso}")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source_specs = lock["sourceDependencies"]
    if args.msl_url:
        source_specs["msl-legacy"]["url"] = args.msl_url
        source_specs["msl-decomp"]["url"] = args.msl_url
    if args.slippi_ai_url:
        source_specs["slippi-ai"]["url"] = args.slippi_ai_url

    deps_dir = args.deps_dir.expanduser().resolve()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    deps_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    slippi_ai_root = prepare_dependency("slippi-ai", source_specs["slippi-ai"], deps_dir)
    run([sys.executable, "-m", "pip", "install", "-e", str(slippi_ai_root)])
    run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-inference.txt")])
    app_requirement = f"{ROOT}[dev]" if args.dev else str(ROOT)
    run([sys.executable, "-m", "pip", "install", "-e", app_requirement])

    bundle_specs = lock.get("runtimeBundles", {}).get(runtime_platform_key())
    if bundle_specs and not args.from_source:
        asset_dir = args.runtime_asset_dir.expanduser().resolve() if args.runtime_asset_dir else None
        runtimes = prepare_bundled_runtimes(bundle_specs, runtime_dir, asset_dir)
    else:
        reason = "requested" if args.from_source else f"no bundle for {runtime_platform_key()}"
        print(f"MSL source build fallback: {reason}", flush=True)
        runtimes = prepare_source_runtimes(source_specs, deps_dir, runtime_dir)

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(venv_bin_dir()), env.get("PATH", "")))
    if not args.skip_node:
        _, npm = install_node(str(lock["node"]))
        run([str(npm), "ci"], cwd=ROOT, env=env)

    if not args.skip_data:
        for name, root in runtimes.items():
            print(f"Extracting legal game data for MSL {name}", flush=True)
            extract_command = [
                sys.executable,
                "-m",
                "melee_sim.extract_data",
                "--iso",
                str(iso),
                "--out-dir",
                str(root / "data"),
            ]
            if name == "legacy":
                extract_command.append("--timings")
            run(extract_command, env=runtime_environment(root))

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    config = args.config.expanduser().resolve()
    write_config(
        config,
        data_dir=data_dir,
        model=model,
        msl_root=runtimes["legacy"],
        msl_decomp_root=runtimes["decomp"],
        slippi_ai_root=slippi_ai_root,
        iso=iso,
    )
    smoke_env = env.copy()
    smoke_env["SLIPPI_REVIEW_CONFIG"] = str(config)
    smoke_command = [sys.executable, str(ROOT / "scripts" / "portable_smoke.py")]
    if args.skip_data:
        smoke_command.append("--code-only")
    if args.skip_node:
        smoke_command.append("--skip-node")
    run(smoke_command, env=smoke_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
