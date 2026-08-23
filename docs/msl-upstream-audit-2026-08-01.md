# MSL Upstream Audit - 2026-08-01

## Repositories and refs

- Fork: `dhambly/melee-sim-light`
- Original upstream: `kyhavlov/melee-sim-light` (private)
- Current analyzer backend: `fork-ready/peach-parity` at `c7a41171`
- Most complete upstream candidate: `decomp-port-arm64-ppc-polish` at `8eb59d09`
- The checkout now has an `upstream` remote. Fetches use LFS smudging disabled unless replay data is deliberately requested.

The two active lines have no Git merge base. Our fork line starts at `dcba3966`; the decomp-port
line starts at upstream's separate `e88dc831` root. This is a replacement architecture, not a
branch that can be fast-forwarded or conventionally merged.

## Upstream branch map

`decomp-port-arm64-ppc-polish` contains the substantive stacked line:

1. `experiment/decomp-port`
2. `decomp-port-luigi`
3. `decomp-port-marios`
4. `decomp-port-samus`
5. `decomp-port-gh-actions`
6. `decomp-port-icies`
7. `decomp-port-pikachu`
8. `decomp-port-dk`
9. `decomp-port-ganon`
10. `decomp-port-arm64-ppc`
11. `decomp-port-arm64-ppc-polish`

The latest tip is 1,477 commits beyond the unrelated common history counted by Git's symmetric
comparison, while our current line has 116 commits absent from it. Separate heads such as
`gh-actions-badge`, `gh-actions-cache-split`, `harden-viewer-data-check`, and
`experiment/decomp-port-interface-cleanup` are small PR/helper branches whose substantive changes
are already represented on the stacked decomp-port line. The old `dev`, `newchar-*`, exact-trig,
and legacy portability branches are not the migration target.

## What the new core adds

- Source-shaped C runtime built from the Melee decomp rather than the current hand-maintained
  simulator owners.
- Public batch operations for masked stepping, arbitrary batch copy, complete save/restore, and
  observation without stepping.
- Sixteen character selections: Fox, Falco, Marth, Sheik, Zelda, Captain Falcon, Jigglypuff,
  Peach, Luigi, Mario, Dr. Mario, Samus, Ice Climbers, Pikachu, Donkey Kong, and Ganondorf.
- Native, PPC, and Wasm validation lanes plus a live/replay web viewer.
- Raw ISO extraction with a deterministic manifest and stale-data validation.
- PPC-exact fused arithmetic, collision, matrix, quaternion, dynamics, and source-order work.
- Linux and macOS host support, including arm64/Rosetta build work.

The final polish report records:

- 366 replay cases and 3,501,461 frames.
- 286 exact passes, 80 reviewed classifications, zero failures, and zero errors.
- 16-character, six-stage, two/four-player runtime census.
- About 102k resident simulation frames/sec at batch 256 and 96k at batch 512 on its retained
  Ryzen 9950X3D workload.
- About 650k aggregate replay-validation frames/sec when validation is parallelized.
- All 45 Python tests and the native/PPC/Wasm/viewer gates passing on the reported host.

## What it does not contain

- Yoshi runtime support. The bundled viewer can display Yoshi, but the simulator API does not
  admit character 14.
- Our `tools/slippi` replay-state builder.
- Our `tools/modelplay` Phillip bridge and state adapter.
- Parser-compatible policy-controller history and teacher-forced recurrent-state injection.
- A public API for constructing an arbitrary internal match from one Slippi post-frame row.
- Native Windows/MSVC support. The Python wrapper loads `libmelee_core.so`, and the source/build
  layout assumes a case-sensitive filesystem. Windows development should use WSL on its Linux
  filesystem or a Linux host.

The upstream validator starts from match initialization and replays controller inputs in order.
That is slower than arbitrary row seeding for one scenario, but it preserves hidden state. For a
whole game, we can replay once, save at candidate frames, copy each saved match into a large batch,
then run thousands of counterfactual lanes. This is a good fit for the public API.

## Analyzer impact

Do not merge `8eb59d09` into `c7a41171`. The practical integration is a second backend or a future
replacement branch:

1. Clone/check out the polished upstream line inside WSL ext4.
2. Extract its raw data from the local ISO and run its native smoke plus one replay validation.
3. Build a small analyzer-owned adapter from `MslObservation` to Phillip's observation schema.
4. Teacher-force Phillip's recurrent/controller history from the original SLP independently of
   simulator state.
5. Replay each game once and save states at all candidate frames.
6. Restore/copy those states into batched rollout lanes, then apply Phillip controllers.
7. Export trace data through the existing MSL trace/viewer path.
8. Compare identical Fox/Falco/Marth/Sheik/Samus scenarios against the current backend before
   changing the website default.

Keep the existing backend for current Windows production and Yoshi. If the decomp backend proves
better for takeover consistency and throughput, port Yoshi as a source packet in that architecture
rather than trying to transplant the decomp character code into the legacy simulator.

## Recommended decision

Treat `decomp-port-arm64-ppc-polish` as the upstream candidate worth prototyping. Do not cherry-pick
its character commits into the legacy branch: the source, data, state, binding, and validation
boundaries are different. Maintain both backends during evaluation, with a single analyzer-facing
interface for replay pre-roll, snapshots, batch copies, stepping, observations, and trace export.

Upstream's default branch is still old, the polished tip has not been merged to it, and the open PR
stack currently stops earlier in the chain. Pin the exact candidate commit during prototyping and
expect upstream history to move.
