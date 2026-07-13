from __future__ import annotations

import argparse
import sys


COMMANDS = {
    "serve": "server",
    "worker": "worker",
    "analyze": "pipeline",
    "candidates": "candidates",
    "simulate": "simulation",
    "select": "selection",
    "practice": "training",
    "doctor": "operations",
    "cleanup": "operations",
    "startup": "operations",
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="slippi-review", description="Analyze Slippi replays with Phillip and melee-sim-light.")
    parser.add_argument("command", choices=COMMANDS)
    args, remaining = parser.parse_known_args()
    module = __import__(f"slippi_ai_review.{COMMANDS[args.command]}", fromlist=["main"])
    sys.argv = [f"slippi-review {args.command}", *remaining]
    if args.command in {"doctor", "cleanup", "startup"}:
        return int(module.main([args.command, *remaining]) or 0)
    return int(module.main() or 0)
