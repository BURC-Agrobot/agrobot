# FAST: Agrobot functions

- source_pdf: `pdf_docs/agrobot-functions.pdf`
- date: 2026-10-01
- goal:
  1. Identify fruits and vegetables.
  2. Check ripeness.
  3. Choose the best picking paths.
  4. Follow these paths.
  5. Put the harvest in a bin.

## Semantics

- Edge `A -> B`: Function B describes HOW to do function A.
  Function A explains WHY the robot needs function B.
- `AND`: The parent function requires all its child functions together.
- `in_scope: no`: The function is outside the study boundary, to the left of the
  study start.
- The robot must perform every function inside the study boundary.

## Nodes

- `speed_up_harvesting` | role: why_build_it | in_scope: no | Increase harvesting speed | Use automatic picking
- `harvest_produce` | role: main_job | in_scope: yes | Harvest produce | Harvest fruits and vegetables automatically
- `choose_produce` | in_scope: yes | Choose produce | Use crop type and ripeness to choose produce
- `identify_crops` | in_scope: yes | Identify crops | Use computer vision
- `check_ripeness` | in_scope: yes | Check ripeness | Determine whether produce is ripe or unripe
- `find_produce` | in_scope: yes | Find produce | Find where to pick
- `pick_produce` | in_scope: yes | Pick produce | Remove the chosen produce
- `separate_produce` | in_scope: yes | Separate produce | Remove produce from the plant
- `hold_produce` | in_scope: yes | Hold produce | Prevent produce from falling
- `bin_harvest` | in_scope: yes | Place harvest | Put the harvest in the bin
- `move_produce` | in_scope: yes | Move produce | Carry the harvest to the bin
- `drop_off_produce` | in_scope: yes | Deposit produce | Put the harvest in the bin

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
