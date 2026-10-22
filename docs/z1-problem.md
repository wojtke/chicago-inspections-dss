# Z1: the decision problem (first draft)

This follows the checklist from the course's introduction slides. Items marked *(to check)* still
need a source.

## Decision-maker

The Food Protection programme of the Chicago Department of Public Health. In practice, the
supervisors who assign each day's inspections to the inspectors (the city calls them
*sanitarians*).

## Goal

Find food-safety hazards as early as possible with the inspectors the city has. The sooner a
failing place is visited, the fewer people are exposed to it.

## Options

- **What is chosen:** which establishments to inspect on each day, from the list of places due for a routine inspection.
- **Options we compare:**
  - the order the city actually used
  - a random order
  - highest predicted risk first
  - later, risk with constraints and preference weights (see the [plan](plan.md))

## Criteria

- How many failing places are found in the first half of the planning window.
- How many days sooner, on average, a failing place is reached.
- Inspector workload: the number of visits per day stays within capacity.
- Later: fairness across neighbourhoods, and travel between visits.

## Limits

- **Capacity:** inspectors × visits per inspector per day. In 2025–26 the city did a median of 31 routine inspections per working day; the quarterly average ranged from 17 to 43 *(to check how many inspectors that is)*.
- **Mandatory visits come first:** complaints, licence applications, suspected food poisoning and re-inspections after a failure.
- **Required frequency:** each place has a risk category from 1 (high) to 3 (low), and higher-risk places must be inspected more often *(to check the exact frequencies)*.

## Environment

- The public, restaurant owners, and the city budget for inspectors.
- The rules changed in July 2018, when the way violations are written changed *(to check: the new food code behind it)*.
- Open data and public scrutiny. The city published its model, and it was independently analysed in 2019.

## Uncertainty

- We only learn whether a place fails by visiting it. The model gives a probability, not a certainty, and it is moderately accurate (AUC about 0.62 on 2025–26).
- We only have results for places that were inspected, so we cannot know what visits that never happened would have found.
- Places open and close. About 26,000 visits in the data ended with "Out of Business".

## What kind of problem it is

- **Operational and repeated:** a decision every day, with quick feedback.
- **Semi-structured:** the ranking can be computed, but supervisors add local knowledge.
- **Decision under risk:** outcomes are uncertain, but their probabilities can be estimated from data.
- **Hybrid DSS:** data-driven (an ML risk model) combined with model-driven (scheduling under capacity, and preference weights from AHP).

## Data

[Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5),
City of Chicago data portal. It has 317,495 inspections from 04.01.2010 to 21.10.2026 (checked
22.10.2026), covering 49,271 licences.

Each row holds the establishment, its type, risk category, address and coordinates, the date and
type of the inspection, the result and the violations, written as free text.

## How we judge a decision

We replay past inspections. We take the routine visits actually done in a 60-day window, keep the
daily capacity, change only the order, and count how much sooner the failing places are found.
This is the method of the city's own 2014 evaluation, which found critical violations 7.5 days
sooner.
