# FAST: Agrobot functions

- source_pdf: `pdf_docs/agrobot-functions.pdf`
- date: 2026-10-01
- goal: Identify fruits and vegetables, check ripeness, choose and follow the
  best picking paths, and put the harvest in a bin.

## Semantics

- Edge `A -> B`: B is HOW A is done; A is WHY B is needed.
- `AND`: all children of the same parent are jointly required.
- `in_scope: no`: outside the study boundary (left of study start).
- Every in-scope function is a required function.

## Nodes

- `speed_up_harvesting` | role: why_build_it | in_scope: no | Speed up harvesting | through automatic picking
- `harvest_produce` | role: main_job | in_scope: yes | Harvest produce | fruits and vegetables; work automatically
- `choose_produce` | in_scope: yes | Choose produce | by crop type and ripeness
- `identify_crops` | in_scope: yes | Identify crops | computer vision
- `check_ripeness` | in_scope: yes | Check ripeness | ripe or unripe
- `find_produce` | in_scope: yes | Find produce | find where to pick
- `pick_produce` | in_scope: yes | Pick produce | remove chosen produce
- `separate_produce` | in_scope: yes | Separate produce | remove from plant
- `hold_produce` | in_scope: yes | Hold produce | keep it from falling
- `bin_harvest` | in_scope: yes | Bin harvest | place harvest in the bin
- `move_produce` | in_scope: yes | Move produce | carry harvest to the bin
- `drop_off_produce` | in_scope: yes | Drop off produce | put the harvest in the bin

## Edges

- `speed_up_harvesting -> harvest_produce`
- `harvest_produce -> choose_produce` | AND
- `harvest_produce -> pick_produce` | AND
- `harvest_produce -> bin_harvest` | AND
- `choose_produce -> identify_crops` | AND
- `choose_produce -> check_ripeness` | AND
- `choose_produce -> find_produce` | AND
- `pick_produce -> separate_produce` | AND
- `pick_produce -> hold_produce` | AND
- `bin_harvest -> move_produce` | AND
- `bin_harvest -> drop_off_produce` | AND
