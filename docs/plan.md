# Project plan

## What the course asks for (LeON, checked 12.11.2026)

| Task | Points | What we hand in | Due |
|---|---|---|---|
| **Z1** | 30 | Analysis of the decision problem + assumptions of the system architecture | **14.12** (LeON: 15.12, 00:00) |
| **Z2** | 30 | Choice of methods and tools + implementation of the prototype | **20.12** (LeON: 21.12, 00:00) |
| **Z3** | 40 | Presentation and defence of the project | **20.12** (LeON: 21.12, 00:00) |

- Pass from 51 points. 51–60 → 3.0, 61–70 → 3.5, 71–80 → 4.0, 81–90 → 4.5, 91–100 → 5.0.
- Teams of 2–3. The defence is individual, so each person must be able to explain their own part.
- The Z1 and Z2 reports are written in Polish, as PDFs built from the course LaTeX template. Code and README stay in English.

## What the product is (pitch v2, 12.11)

A decision support system for a city's food-inspection service, built only from the city's open data:

1. **Prediction module.** Our own risk model gives the chance that a routine inspection fails.
   - Gradient boosting, calibrated (Platt) on 2024, tested on 2025–26.
   - It does **not** use coordinates, because they can stand in for neighbourhood income (see Fairness).
2. **Decision module.**
   - The manager's preferences come from AHP: risk vs who eats there (schools, hospitals) vs time since the last visit.
   - Constraints: inspectors, mandatory visits after complaints, travel.
   - It produces daily routes with OR-Tools (a prize-collecting vehicle-routing problem).
   - Neighbourhood balancing keeps poorer areas from being over-inspected or under-protected.
3. **Interface.** Two local web apps:
   - **The product** (`app.py`). One click switches between roles; there are no passwords.
     - The **manager** plans and approves the day, skips or pins places and sets the AHP priorities. A neighbourhood check flags an unbalanced plan.
     - The **inspector** sees their route, each place's history and why it is on the list, and records results. Those results feed back into the data.
   - **The technical report** (`report.py`): replay results, team-size scenarios, fairness audit and model quality. It is kept out of the product.

Evaluation replays 60-day windows of 2025–26 against the city's actual order and a random order.

Everything city-specific sits in one `CityConfig` (`src/inspections/cities/chicago.py`), so another city can be plugged in.

## Status

| Part | State |
|---|---|
| Data: canonical schema, neighbourhoods, socioeconomic indicators | Done |
| Risk model with calibration and fairness audit | Done |
| AHP preferences, priority, reasons in Polish | Done |
| Live planning (due places, mandatory queue, routes) | Done |
| Replay: 6 policies × 4 team sizes × 5 windows | Done |
| Product app: role switch, manager pages, inspector page, results | Done |
| Technical report app | Done |
| Product: named team and absences, place card from map/lists, frozen plans, re-inspections after a fail, demo seed (`insp demo`) | Done |
| Technical report, LaTeX + pgfplots (`docs/tech-report`, PL, title + 7 pages, numbers generated from the results) | Draft; base for Z1 and Z2 |
| Tests (city-agnostic pipeline, AHP, routing, fairness, store) | Done |
| Z1 report (PL) | To write; due 14.12 |
| Z2 report (PL) | To write; due 20.12 |
| Z3 deck and defence | To prepare; due 20.12 |

## Who owns what (for the defence)

| | Piotrek | Wojtek |
|---|---|---|
| Owns | Prediction (features, calibration, Model page), evaluation and fairness | Decision module (AHP, constraints, routing), app, architecture |
| Writes | Z1: problem analysis; Z2: prediction and evaluation methods | Z1: architecture; Z2: decision methods and prototype |
| Presents | Problem, data, model, fairness | System, live demo, results |

## Timeline

| Dates | Work |
|---|---|
| 12–18.11 | Code complete (this list). Read lectures 2–4 from LeON. Z1 outline. |
| 19–30.11 | App polish; Piotrek reviews the prediction and fairness parts |
| 1–14.12 | Write Z1, review, **submit by 14.12** |
| 14–20.12 | Write Z2, deck, rehearse the demo, **submit Z2 and Z3 by 20.12** |

## Open questions

- Inspection frequency by risk category (we assume 182 / 365 / 730 days). Check against the city's food code.
- How many sanitarians the city really has (we default to 13 in the field, 6 visits each, which matches the ~76 visits a day in the data).
- The socioeconomic indicators are from 2008–2012, the newest the portal offers per community area.
- The date of the presentation in class.
