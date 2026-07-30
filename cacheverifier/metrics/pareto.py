from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    hit_rate: float
    error_rate: float


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Non-dominated subset for "maximize hit_rate, minimize error_rate".

    A point is dominated if another point has hit_rate >= it and
    error_rate <= it, with at least one strict inequality. Used to plot the
    four-group comparison in Section 5.3 and to drive the Go/No-Go check in
    Section 6.
    """
    frontier = []
    for p in points:
        dominated = any(
            (q.hit_rate >= p.hit_rate and q.error_rate <= p.error_rate)
            and (q.hit_rate > p.hit_rate or q.error_rate < p.error_rate)
            for q in points
            if q is not p
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p.hit_rate)
