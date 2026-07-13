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

These are external runtime checkouts and are not vendored. Their paths are
supplied through the local configuration file. The compatibility patch under
`patches/` is maintained by this project for replay ingestion.
