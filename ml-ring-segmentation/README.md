# ML ring segmentation in Coral

Events with several particles produce several overlapping Cherenkov rings. Coral's own
multi-ring fit adds rings one at a time, and each new ring means another full likelihood fit.
This module makes a different route optional. WatChMaL's sparse 3D U-Net (the segmentation
network is M. El Baz's) assigns each hit to a ring. Coral then fits each ring's hits as a
single ring, masking the PMTs that belong to the other rings, so that their light is not
read as this ring's absence.

| Path | Role |
|---|---|
| `ring_segmentation.py` | Loads the segmenter from Coral's `[ml]` config section and returns per-ring hit sets. Handles the network's raw-time convention (Coral subtracts the trigger offset, the network does not) and PMTs shared between rings |
| `coral_integration_excerpt.py` | The two places in Coral's event loop that call it |

First measurement (Sep 2026): the network finds 2+ rings in 30 % of π⁰ events, splits 19 % of
single muons, and runs about 1.7× faster than the full multi-ring fit.

M. El Baz later built a hybrid on top of it: network rings → a joint fit with a shared
vertex → a per-ring momentum refit. Against Coral's full-likelihood multi-ring fit on the
same events, his campaign measured:

| | full-likelihood MR | hybrid |
|---|---|---|
| ring count correct | 79.4 % | 89.0 % |
| all rings found + PID correct | 83.9 % | 91.5 % |
| summed kinetic energy, 68 % width | 9.5 % | 5.7 % |
| median time per event | 1× | 4.5× faster |
