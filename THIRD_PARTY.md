# Third-Party Components

## Project Slippi JavaScript SDK

Installed from npm as `@slippi/slippi-js` and used to parse `.slp` files.
Package license: LGPL-3.0-or-later.

## slippi-viewer

The packaged browser viewer under `src/slippi_ai_review/web/msl_static` comes
from melee-sim-light's `tools/viewer` integration of
`@gcpreston/slippi-viewer`. Character animation archives are runtime visual
assets used by that viewer. Upstream provenance is preserved here because the
interactive reports must work without a separate web build.

## melee-sim-light and slippi-ai

The Linux x86_64 and macOS MSL runtime wheels are vendored under
`vendor/runtimes/` from the exact fork commits recorded in
`dependencies.lock.json`. They contain the native simulator and the
replay/Phillip analysis bridges, but no copyrighted game data. Bootstrap
extracts that data locally from the user's ISO.

Other platforms build those same commits into local `.runtime/` directories.
`slippi-ai` is fetched at its locked public commit, with the compatibility
patch under `patches/` applied during setup.

## gm-v2 model

The user-supplied Phillip `gm-v2` checkpoint is vendored under `vendor/models/`
and checksum-pinned in `dependencies.lock.json`. Its redistribution provenance
has not been established, so this private repository and any archive containing
the model must not be made public until that is resolved.
