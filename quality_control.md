# Quality Control Checklist

Before any published result:

- [ ] Record event matches Guinness: 142.2 dBA, Sept. 29, 2014.
- [ ] 2014 record-setting game is not accidentally treated as a continuous 142.2 dBA exposure.
- [ ] Every play has a valid game_id and play_id.
- [ ] No duplicate game_id/play_id combinations.
- [ ] Home/away teams agree with game table.
- [ ] Penalty team is valid when penalty == 1.
- [ ] EPA/WPA missingness is documented.
- [ ] 2026 cutoff date is frozen and recorded.
- [ ] Postseason is explicitly included/excluded.
- [ ] Team relocations/stadium renames are harmonized.
- [ ] Roof/surface metadata is time-aligned where possible.
- [ ] Historical noise observations are stored separately from modeled exposure.
- [ ] Main results include uncertainty intervals.
- [ ] At least three counterfactual intensity scenarios are reported.
- [ ] Record-setting 2014 game is excluded in a robustness check.
- [ ] Placebo tests are performed.
- [ ] Sequential simulation updates score, clock, field position, and possession.
- [ ] Simulation seed is recorded.
- [ ] Number of simulations is reported.
- [ ] No graph labels a modeled quantity as "measured dBA."
