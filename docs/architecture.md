# Architecture

## Product Boundary

The application owns one workflow:

1. validate and store a Slippi replay;
2. choose the analyzed player and sampling preset;
3. enumerate advantage opportunities from Slippi events;
4. sample Phillip/MSL continuations in a native or WSL runtime;
5. select robust option families against the replay baseline;
6. export synchronized replay and counterfactual traces;
7. publish an interactive HTML report;
8. optionally export a selected route to Training Mode CE.

Historical Dolphin takeover probes, video-render experiments, lesson/wiki
generation, and generated outputs are intentionally outside this repository.

## Modules

- `server`: HTTP API, dashboard, uploads, job lifecycle, artifact serving.
- `worker`: durable single-GPU queue consumer and crash recovery.
- `pipeline`: resumable orchestration and artifact publication.
- `candidates`: exhaustive Slippi conversion/opportunity extraction.
- `simulation`: vectorized Phillip/MSL rollout backend.
- `selection`: robust option-family ranking and replay comparison.
- `render_target`: one selected route to synchronized traces.
- `artifacts`: portable trace consolidation and report assembly.
- `report`: interactive game-review document.
- `training`: selected-route export and CE launch integration.
- `config`: the only machine-specific configuration boundary.

## Runtime Data

`data/reviews/<uuid>/review.json` is the durable job record. Pipeline stages
write beneath that review's `pipeline/` directory and publish final output by
atomically replacing `artifacts/`. A killed worker returns an interrupted job to
the queue and reuses completed stage outputs.

## External Dependencies

`slippi-ai` remains a pinned upstream checkout because Phillip uses its model
runtime. Bundled platforms unpack both `melee-sim-light` backends from verified
native wheels; source fallback checkouts are managed automatically. The app
passes every runtime location explicitly, and no production module searches a
developer workspace.

Punish simulations use recorded-contact anchoring: for an opening already
present in the replay, the branch begins on the first recorded physics frame
after hitlag. This holds the opening contact and defensive DI fixed instead of
asking approximate simulation physics to recreate that contact.
