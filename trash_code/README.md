# Trash Code Archive

This folder keeps inactive code and historical outputs out of the active 24-block
LIF SNN path while preserving them for reference.

## Code

```text
code/
|-- ann2snn/          # Direct ANN-to-SNN conversion utilities and notes.
|-- exp_ann2snn/      # Old ANN-to-SNN launcher scripts.
|-- exp_lif_staged/   # Earlier 3/6/12-block LIF staged launchers.
|-- exp_sweeps/       # Validation-only and scope sweep launchers.
|-- legacy_models/    # Old standalone SpikMamba baseline.
|-- sync_scripts/     # Previous sync/server helper scripts.
`-- tools/            # Miscellaneous helper scripts.
```

## Outputs

```text
outputs/
|-- ann2snn/                 # Direct conversion outputs.
|-- ann_experiments/         # Old ANN, scratch, and local runs.
|-- custom_trainable_spike/  # Signed custom TrainableSpike3dSeq experiments.
|-- lif_snn/                 # LIF sweeps and early staged outputs.
`-- reports/                 # Monthly report artifacts.
```

The active outputs that remain outside this archive are:

- clean ANN teacher output
- 12-block unsigned LIF checkpoint source
- 24-block unsigned LIF current run

If an archived experiment needs to be restored, move only the specific script or
output folder back to its original location and check its paths before running.
