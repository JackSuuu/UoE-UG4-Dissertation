# Experiment record

Running log of experiment progress against
[`notes/Experiment_plan_1year.md`](../notes/Experiment_plan_1year.md),
one file per week. Inside a file, the newest entry is at the top.

| week | dates | file | main events |
|---|---|---|---|
| 1 | to 29 Sep 2026 | [week1.md](week1.md) | Genesis bring-up and demotion; Task A → box peg, two-fingertip pusher; open-loop chunk protocol; three pipeline defects; v5; repair audit; RQ2b shift ladder |
| 2 | from 30 Sep 2026 | [week2.md](week2.md) | CheckVLA's gradient branch and hard prefix are inert; v6/v7; rate-matched τ (physics beats vision at matched cost); OpenVLA-7B on transformers 5.x; demo + feature pipeline for the VLA actor |

## Conventions

- **English**, one `### Update <date> — <one-line finding>` per entry. The title is
  the result, not the activity.
- Record **what was measured and the number**, then what it means. Say which
  earlier reading it overturns, and edit that entry with a pointer forward
  rather than deleting it.
- A new week starts a new file `weekN.md`; add a row to the table above.
- Commands for long runs are recorded with `setsid nohup ... < /dev/null`.
  Anything started with a plain `&` dies with the launching shell.
