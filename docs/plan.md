# Project plan

## What the course asks for

From the lecturer's organisation slides (Z0) and the course rules, 2026/27:

- Teams of 2–3 people. The project is defended individually.
- Five tasks, 100 points in total, a pass from 51 points:
  - **Z1** analysis of the decision problem
  - **Z2** system architecture
  - **Z3** choice of methods and tools
  - **Z4** prototype
  - **Z5** presentation, given in two parts
- Grades: 51–60 → 3.0, 61–70 → 3.5, 71–80 → 4.0, 81–90 → 4.5, 91–100 → 5.0. Activity in class can raise the grade.
- Deadlines for the tasks are not published yet.

What the introduction slides (Z1) stress:

- **A DSS is part of a decision process, not just an algorithm:** problem → data, preferences, limits → model → analysis / prediction / optimisation → recommendation → human decision → feedback.
- **Z1 describes the problem with a fixed checklist:** decision-maker, goal, options, criteria, limits, environment, uncertainty, and what kind of problem it is.
- **Prediction is not a decision.** A *prediction module* estimates a probability. A *decision module* recommends what to do, using costs, preferences, limits and risk.
- **Hybrid systems dominate:** ML prediction → optimisation → explanation → human decision.
- **Methods the lecturer often sees:** scenario analysis, GIS, AHP, ML in Python, scoring.

## Plan per task

| Task | What we hand in | Where we are |
|---|---|---|
| **Z1** Decision problem | The checklist filled in for Chicago's inspection planning: [z1-problem.md](z1-problem.md) | First draft |
| **Z2** Architecture | Diagram and description: data → features → risk model → scheduler → app → inspector → feedback (new results flow back into the data) | Matches the code layout; diagram to draw |
| **Z3** Methods and tools | Gradient boosting vs logistic regression for the risk; ranking under daily capacity for the schedule; AHP for preference weights; the replay as the measure; Python, scikit-learn, Streamlit | Risk model and replay done |
| **Z4** Prototype | The Streamlit app, extended with the next steps below | Base version works |
| **Z5** Presentation | Part 1: the problem and the city's case. Part 2: our system, the replay results, the limits | — |

## What the prototype already does

- Downloads and cleans 316k inspections (2010–2026).
- Predicts whether a routine inspection fails, from what was known before that day.
- Ranks the places due and fills each day's capacity.
- Replays past 60-day windows. The risk-first order finds failing places 4–8 days sooner than the order the city actually used.
- App with three tabs: daily list and map, replay, model quality.

## Next steps, roughly in order

1. **Mandatory visits first.** Complaints, licence applications and re-inspections must happen. They should use up capacity before the routine list. This is a real constraint for Z1.
2. **Preferences.** A failing school or daycare matters more than a failing bar. Weight places by type and risk category, and derive the weights with **AHP**. This is the decision-maker's preference, separate from the prediction.
3. **What-if scenarios.** What if there were 20% fewer inspectors, or a summer surge? Add a capacity multiplier to the replay.
4. **Inspector areas.** Split the city into areas so each inspector's list is nearby. This is the GIS part.
5. **Better prediction.** Add the features the city used: weather, nearby sanitation complaints from 311, garbage and burglary reports.
6. **Fairness check.** Are some neighbourhoods always pushed to the back of the queue?

## Team split, for three people

| Person | Owns | Writes |
|---|---|---|
| A | Data, the city's original model and evaluation | Z1, the data part of Z2 |
| B | Risk model, features, model quality | Z3 |
| C | Scheduler, constraints, preferences (AHP), app | Z2, Z4 |

Everyone presents their own part, since the defence is individual.

## How we work

- `uv run pytest` before pushing.
- Data and models are never committed; `insp data` and `insp train` rebuild them.
- Small pull requests, one topic each.

## Open questions

- Task deadlines, and the dates of the two presentation parts.
- Whether Z1–Z3 are written reports or are discussed in class.
- The third team member.
