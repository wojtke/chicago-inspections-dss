"""Daily routes for the inspectors: a prize-collecting vehicle routing problem (OR-Tools).

Each candidate place carries a prize, its AHP priority. Each inspector is a vehicle that can make
a limited number of visits within a working day. The solver picks which places to visit and in
what order, trading priority against driving time: skipping a place costs its prize, driving
costs minutes. Routes are open (an inspector starts at the first visit and ends at the last), so
we need no assumption about where inspectors live or which office they leave from.

Assumptions, all in `RoutingConfig`: Manhattan distance (Chicago is a grid), 20 km/h average
city speed, 45 minutes per inspection, an 8-hour day.
"""
import math
from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

KM_PER_DEG_LAT = 110.57


@dataclass
class RoutingConfig:
    speed_kmh: float = 20.0
    service_min: int = 45
    day_min: int = 480
    visits_per_inspector: int = 6
    candidates: int = 100          # top places by priority offered to the solver each day
    time_limit_s: float = 1.0
    minutes_per_priority: float = 240.0  # how many minutes of driving one unit of priority is worth


def xy_km(lon: np.ndarray, lat: np.ndarray, center_lat: float) -> np.ndarray:
    """Local planar coordinates in km (good enough across one city)."""
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    return np.column_stack([np.asarray(lon) * km_per_deg_lon, np.asarray(lat) * KM_PER_DEG_LAT])


def manhattan_km(xy: np.ndarray) -> np.ndarray:
    return np.abs(xy[:, None, 0] - xy[None, :, 0]) + np.abs(xy[:, None, 1] - xy[None, :, 1])


def vehicle_caps(n_visits: int, per_inspector: int) -> list[int]:
    """Split a day's visits over the fewest inspectors, as evenly as possible."""
    if n_visits <= 0:
        return []
    k = math.ceil(n_visits / per_inspector)
    base, extra = divmod(n_visits, k)
    return [base + (i < extra) for i in range(k)]


def solve(xy: np.ndarray, caps: list[int], cfg: RoutingConfig, prize: np.ndarray | None = None,
          optimise: bool = True) -> list[list[int]]:
    """Routes as lists of indices into `xy`, one list per inspector.

    prize=None means every place must be visited (used to measure the km of a given day's list);
    otherwise places may be skipped at the cost of their prize.
    """
    n = len(xy)
    if n == 0 or not caps:
        return [[] for _ in caps]
    minutes = np.zeros((n + 1, n + 1), dtype=int)  # node 0 is a free dummy depot: open routes
    minutes[1:, 1:] = np.rint(manhattan_km(xy) / cfg.speed_kmh * 60).astype(int)
    manager = pywrapcp.RoutingIndexManager(n + 1, len(caps), 0)
    model = pywrapcp.RoutingModel(manager)

    def travel(i, j):
        return int(minutes[manager.IndexToNode(i), manager.IndexToNode(j)])

    def visits(i):
        return 0 if manager.IndexToNode(i) == 0 else 1

    travel_cb = model.RegisterTransitCallback(travel)
    model.SetArcCostEvaluatorOfAllVehicles(travel_cb)
    model.AddDimensionWithVehicleCapacity(model.RegisterUnaryTransitCallback(visits), 0, caps, True, "visits")
    if prize is not None:
        def work(i, j):
            return travel(i, j) + (cfg.service_min if manager.IndexToNode(i) != 0 else 0)
        model.AddDimension(model.RegisterTransitCallback(work), 0, cfg.day_min, True, "time")
        for node in range(1, n + 1):
            model.AddDisjunction([manager.NodeToIndex(node)], int(round(prize[node - 1] * cfg.minutes_per_priority)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if optimise:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(int(cfg.time_limit_s * 1000))
    sol = model.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("no feasible routes")
    routes = []
    for v in range(len(caps)):
        i, route = model.Start(v), []
        while not model.IsEnd(i):
            node = manager.IndexToNode(i)
            if node:
                route.append(node - 1)
            i = sol.Value(model.NextVar(i))
        routes.append(route)
    return routes


def leg_km(xy: np.ndarray, a: int, b: int) -> float:
    """City-block distance between two places (Chicago's streets are mostly a grid)."""
    return float(abs(xy[a, 0] - xy[b, 0]) + abs(xy[a, 1] - xy[b, 1]))


def route_km(xy: np.ndarray, routes: list[list[int]]) -> float:
    """Total driving distance of open routes."""
    return sum(leg_km(xy, a, b) for r in routes for a, b in zip(r, r[1:]))
