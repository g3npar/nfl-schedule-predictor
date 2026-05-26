# g3npar's NFL Schedule Predictor

A full 18-week NFL schedule built with integer linear programming. Every game is assigned a week and kickoff time, covering primetime slots (SNF, MNF, TNF), international games, Thanksgiving, Black Friday, and Christmas.

> **Note:** The official 2026 NFL schedule has been released. A fairness comparison between the predicted and official schedules is now live on the site. A second feature for this website is in development.

🔗 **Live site:** https://g3npar.github.io/nfl-schedule-maker/

---

## How it works

**Opponent assignment** (`src/opponents.py`) follows the NFL's rotation rules: every team plays each division opponent twice, plus cross-division and cross-conference matchups. This produces 272 total games, stored in `data/games.txt`.

**Week assignment** uses [PuLP](https://coin-or.github.io/pulp/) to solve an ILP. Each (game, week) pair becomes a binary variable, and the solver finds an assignment that satisfies all scheduling constraints. Confirmed games are pinned to their exact week and timeslot before solving begins.

**Kickoff time assignment** happens after weeks are set. Primetime slots go to the highest-priority matchups based on per-team weights in `src/primetime_weights.py`. Special slots (Thanksgiving, Black Friday, Christmas, international) are handled separately before the remaining games are distributed.

**HTML generation** (`src/generate_schedules.py`) produces one page per team plus a primetime breakdown page, with light and dark mode support.

---

## Fairness comparison

Once the official schedule was released, a comparison page was built to measure how evenly primetime slots were distributed across all 32 teams in both schedules.

The official schedule was fetched from ESPN's public API and classified by timeslot. Primetime is defined as any game kicking off at 7 PM ET or later, including named slots (SNF, MNF, TNF), night games on Thanksgiving and Christmas, and nighttime international games (Melbourne, Mexico City, etc.). Morning international games and afternoon Thanksgiving/Christmas games are not counted as primetime.

Fairness is measured using the **Gini coefficient**, a value between 0 (perfectly equal) and 1 (maximally concentrated). It's calculated independently for each slot type (TNF, SNF, MNF, overall primetime, 1 PM, 4 PM) and compared side by side.

The ILP solver explicitly constrained how many primetime and afternoon games each team could receive, so the predicted schedule generally achieves a lower Gini score than the official one, which skews heavily toward high-profile markets.

The comparison page is linked from the homepage.

---

## Confirmed games

As the NFL announced official matchups (international games, primetime slots, holiday games, etc.), they were added to `data/confirmed_games.json`. The solver locked those in and filled the rest of the schedule around them.

---

## Project structure

```
index.html                        <- team grid homepage
logos/                            <- team logo images
pages/
  primetime.html                  <- primetime games breakdown
  comparison.html                 <- predicted vs. official fairness comparison
  editor.html                     <- schedule editor (WIP)
  schedules/                      <- one page per team (32 total)

schedule-predictor/
  data/
    games.txt                     <- all 272 matchups
    confirmed_games.json          <- locked-in games (week + timeslot) used during solving
    predicted_schedule.txt        <- final ILP output (the prediction)
    schedule_with_times.txt       <- official 2026 schedule (updated after release)
    real_schedule.json            <- official schedule fetched from ESPN API
    previous_schedule.txt         <- prior year schedule used for constraints
    previous_records.txt          <- prior year win/loss records for primetime weights
    opponents.py                  <- opponent assignment logic (run once to generate games.txt)
  src/
    nfl_schedule_generator.py     <- ILP solver
    generate_schedules.py         <- HTML page generator
    schedule_analysis.py          <- chart generation for the predicted schedule
    scrape_real_schedule.py       <- fetches official schedule from ESPN API
    compare_schedules.py          <- generates fairness comparison charts and page
    editor_server.py              <- local dev server for the schedule editor
    games.py                      <- game data helpers
```
