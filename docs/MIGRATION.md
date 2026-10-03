# Current implementation and historical code

On 2026-10-04 the active repository was replaced with an independent reconstruction
of the final ESWA-D-26-05581R1 revised manuscript. The scientific source is
identified by SHA256 in `configs/reconstruction.yaml` and
[REIMPLEMENTATION.md](REIMPLEMENTATION.md). This is a code and documentation
correction, with explicitly disclosed reconstruction limits.

The last historical `main` commit was
[`b7207f1463bba936d3a018470f05cf712c645f08`](https://github.com/ToughClimb/uav-mfg-hybrid/commit/b7207f1463bba936d3a018470f05cf712c645f08).
It remains reachable at
[`legacy/pre-eswa-reconstruction`](https://github.com/ToughClimb/uav-mfg-hybrid/tree/legacy/pre-eswa-reconstruction).
The current commit extends that history; no historical commits were rewritten.

| Earlier checkout | Current checkout |
| --- | --- |
| `uavpinn/` | Independent package `src/uav_mfg/` |
| `scripts/train.py`, routes A/B | `python -m uav_mfg.cli run` |
| `paperconfig/` | `configs/`, with explicit manuscript values and reconstruction choices |
| Historical `runs/` and checkpoints | Newly generated `results/`, ignored by Git |
| Earlier paper reproduction descriptions | Scoped validation, failed-run disclosure, and reproducible commands |

Historical numerical code, training code, YAML parameters, checkpoints, and
output fields were not reused, translated, or executed in the reconstruction.
Only visual conventions such as colors, colorbar styling, and layouts could
be consulted. Historical figures and earlier reproduction claims are not
validation evidence for the current code or the final revised experiments.

There is no checkpoint or configuration conversion between the implementations.
Old command lines belong to the historical branch. Start current experiments
from the new configs and fresh result directories. For continuations within
the current implementation, physical parameters and grid must match the
checkpoint, and cumulative projection accounting is retained.

The current result set deliberately reports unresolved differences: omitted
original coefficients, the source-side versus target-side density hotspot,
empty outer 3D slices, and the limited monolithic baseline budget. See
[REPRODUCTION.md](REPRODUCTION.md) and [../validation/README.md](../validation/README.md).

Private submission correspondence, the local reference manuscript and original
figures, local environments, and the large HTTP archive are not distributed
with this code. The existing Apache-2.0 `LICENSE` and attribution `NOTICE` are
retained.
