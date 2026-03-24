# Rubin DP0.2 Access Setup

LensForge includes `output/jupyter-notebook/rubin-dp02-access.ipynb` as a prepared live-access artifact for the LSST/Rubin side of the GSoC 2026 DeepLense evaluation.

## Intended use

- local TAP discovery against `https://data.lsst.cloud/` when Rubin client packages and a valid access token are available
- Butler discovery in a Rubin notebook environment where `DAF_BUTLER_REPOSITORY_INDEX` is already configured

## Local prerequisites

- a Rubin access token in `ACCESS_TOKEN` or `LensForge/.secrets/rubin_access_token.txt`
- network access to Rubin services
- Rubin Python clients available in the active environment

## Environment note

The TAP and Butler packages are not part of the base LensForge requirements because they are environment-specific and may be installed differently depending on whether you are using a local machine or the Rubin Science Platform.

The notebook now fails with explicit, user-friendly messages when:

- Rubin client packages are missing
- the access token is missing
- Butler repository metadata is unavailable

## Why this exists

This notebook is the bridge between the current mock LSST-style pipeline in LensForge and a real Rubin-backed implementation. It documents the exact TAP query shape and Butler discovery path the project would use once the proper external access environment is available. It should be read as a prepared live-access path, not as a guarantee that a default local environment can execute Rubin services end to end.
