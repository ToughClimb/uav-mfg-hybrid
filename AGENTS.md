# Project rules

- Implement independently from the final ESWA-D-26-05581R1 revised manuscript.
  Its provenance and reconstruction limits are documented in `docs/REIMPLEMENTATION.md`.
  Private manuscript files, when available locally, live under ignored `reference/`.
- `archive/` is historical material. Do not import, execute, copy, or translate
  its numerical, model, training, or experiment code into the new implementation.
  Only colors, colormaps, colorbar conventions, and figure layouts may be referenced.
- Keep all dependencies in the project-local `.venv` and run through its Python
  or `uv run`. Do not install system packages, NVIDIA drivers, or a system CUDA toolkit.
- Distinguish paper settings from reduced demonstration settings in saved metadata.
- Audit mass with the actual shared face fluxes, source, target absorption, and
  any density projection. Never label an unconverged numerical solve as converged.
- Save commands, seeds, configuration, environment versions, source hashes,
  checkpoints, and diagnostics with experiment outputs.
- Preserve the pre-reconstruction Git history. Do not treat historical code or
  outputs as the current implementation or as verified revised-paper results.
- Publish reproducible code and explicitly scoped validation evidence; keep
  submission correspondence, private reference files and local archives out of Git.
