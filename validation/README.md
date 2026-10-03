# Measured reconstruction evidence

`summary.json` records fresh shared-face flux audits for nine accepted
increased-demand cases, their original field hashes, resolved physical configs,
and the actual generating source hashes and package versions. It also records
the earlier unsuccessful high-load trials and the failed demonstration
monolithic PINN. These results were independently computed from the final
ESWA-D-26-05581R1 model under disclosed reconstruction choices.

`configs/` contains resolved configs for the accepted scenes. `figures/`
contains two numeric plots from those runs, with SHA256 recorded in the summary.
The main density panel covers the true range. The before/after comparison uses
the same linear 0–0.05 range, marking over-range values with a colorbar arrow.
No original manuscript image is included.

Large numeric fields, checkpoints, the private manuscript and correspondence,
and local environment dumps are excluded from Git. Field hashes identify the
local artifacts underlying these compact published records; regenerate fields
using [../docs/REPRODUCTION.md](../docs/REPRODUCTION.md). Hashes and iteration
counts are measurements, not promises of byte-identical results on other
hardware/software stacks.

The publication additionally makes explicit CPU selection apply to transport,
allows rendering without private reference files, and retains projection
history across checkpoint continuations. Those portability/accounting changes
do not retroactively change the source hashes of the previously computed fields.
The numerical source identifiers in the summary refer to the generating runs,
while each new run records its own current identifiers.

The reported mass and continuity checks do not establish exact replication of
the original experiments or exact PDE satisfaction at every continuous point.
The source/target density morphology differs from the manuscript, outer 3D
slices are empty, and the monolithic baseline has a reduced training budget.
