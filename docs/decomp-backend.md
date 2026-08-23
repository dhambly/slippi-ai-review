# Decomp Simulation Backend

The advantage and phase-sweep pipelines can run against the newer decomp-based
`melee-sim-light` fork. The legacy backend remains the default while parity is
measured across a larger replay corpus.

Bootstrap installs this backend beside legacy. Select it in the generated
configuration:

```toml
[paths]
msl_decomp_root = "/path/to/slippi-ai-review/.runtime/msl-decomp"

[runtime]
simulation_backend = "decomp"
```

On Windows, a WSL UNC path is accepted for `msl_decomp_root`, for example
`//wsl.localhost/Ubuntu/home/user/slippi-ai-review/.runtime/msl-decomp`.

You can override the configured backend for one command:

```bash
slippi-review analyze --simulation-backend decomp --replay game.slp --controlled-port 1
```

The selected backend is recorded in queue files, trace manifests, and the final
pipeline summary. Neutral and disadvantage experiments currently remain on the
legacy simulator; the summary's `phaseBackends` field makes this explicit.

## Parity Audit

Compare equivalent run directories containing `lanes.jsonl`:

```bash
python -m slippi_ai_review.backend_compare \
  --legacy-run data/legacy-run \
  --decomp-run data/decomp-run \
  --out data/backend-comparison.json
```

The hard gate requires identical replay-derived branch state and takeover frame.
Phillip's sampled option distribution is compared using total variation distance.
Outcome differences are reported but are not treated as an injection failure,
because the two simulators may legitimately resolve the same inputs differently.
