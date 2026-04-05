# graders/base_grader.py + grader_task1.py + grader_task2.py + grader_task3.py
# (Combined into one file for clarity; split into separate files in project)
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field

from retail_replenish.server.state import Trajectory


# ===========================================================================
# Base
# ===========================================================================


class GradeResult(BaseModel):
    score: float = Field(ge=0, le=1)
    passed: bool = Field(description="True if agent met success criteria")
    breakdown: Dict[str, Any] = Field(
        default_factory=dict
    )  # per-component scores for transparency
    feedback: str = Field(...)  # human-readable summary


class BaseGrader(ABC):
    """All graders must return a deterministic score in [0.0, 1.0]."""

    @abstractmethod
    def grade(self, trajectory: Trajectory, total_received: int = 0) -> GradeResult: ...

    @staticmethod
    def _linear_scale(value: float, min_val: float, max_val: float) -> float:
        """Maps value linearly from [min_val, max_val] → [0.0, 1.0], clamped."""
        if max_val <= min_val:
            return 1.0 if value >= max_val else 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


# ===========================================================================
# Task 1 Grader — Single Store, Stable Demand (Easy)
# ===========================================================================
# Success: fill rate >= 90% over 7-day episode.
# Scoring: continuous linear scale from 50% fill (score=0) to 90%+ fill (score=1).
# ===========================================================================


class GraderTask1(BaseGrader):
    FILL_RATE_PASS = 0.90
    FILL_RATE_FLOOR = 0.50  # score=0 below this

    def grade(self, trajectory: Trajectory, total_received: int = 0) -> GradeResult:
        fill_rate = trajectory.fill_rate()

        score = self._linear_scale(
            fill_rate,
            min_val=self.FILL_RATE_FLOOR,
            max_val=self.FILL_RATE_PASS,
        )
        score = round(score, 4)
        passed = fill_rate >= self.FILL_RATE_PASS

        breakdown = {
            "fill_rate": round(fill_rate, 4),
            "fill_rate_score": score,
        }

        feedback = f"Fill rate: {fill_rate:.1%}. " + (
            "✅ Passed." if passed else f"❌ Target is {self.FILL_RATE_PASS:.0%}+."
        )

        return GradeResult(
            score=score, passed=passed, breakdown=breakdown, feedback=feedback
        )


# ===========================================================================
# Task 2 Grader — Multi-Store, Perishables (Medium)
# ===========================================================================
# Success: fill rate >= 85% AND waste rate <= 10% of units received.
# Scoring: weighted composite — 60% fill rate component, 40% waste component.
#   Fill component: linear scale from 60% → 85%.
#   Waste component: linear scale from 20% waste (score=0) → 0% waste (score=1).
# ===========================================================================


class GraderTask2(BaseGrader):
    FILL_RATE_PASS = 0.85
    FILL_RATE_FLOOR = 0.60
    WASTE_RATE_PASS = 0.10
    WASTE_RATE_CEIL = 0.20  # score=0 at or above this waste rate

    WEIGHT_FILL = 0.60
    WEIGHT_WASTE = 0.40

    def grade(self, trajectory: Trajectory, total_received: int = 0) -> GradeResult:
        fill_rate = trajectory.fill_rate()
        waste_rate = trajectory.waste_rate(total_received)

        fill_score = self._linear_scale(
            fill_rate, self.FILL_RATE_FLOOR, self.FILL_RATE_PASS
        )
        # Waste: lower is better, so invert
        waste_score = self._linear_scale(
            self.WASTE_RATE_CEIL - waste_rate,
            min_val=0.0,
            max_val=self.WASTE_RATE_CEIL - 0.0,  # 0% waste → full score
        )

        score = round(
            self.WEIGHT_FILL * fill_score + self.WEIGHT_WASTE * waste_score, 4
        )
        passed = fill_rate >= self.FILL_RATE_PASS and waste_rate <= self.WASTE_RATE_PASS

        breakdown = {
            "fill_rate": round(fill_rate, 4),
            "fill_score": round(fill_score, 4),
            "waste_rate": round(waste_rate, 4),
            "waste_score": round(waste_score, 4),
            "weighted_score": score,
        }

        feedback_parts = []
        if fill_rate >= self.FILL_RATE_PASS:
            feedback_parts.append(f"✅ Fill rate {fill_rate:.1%} meets target.")
        else:
            feedback_parts.append(
                f"❌ Fill rate {fill_rate:.1%} below {self.FILL_RATE_PASS:.0%} target."
            )
        if waste_rate <= self.WASTE_RATE_PASS:
            feedback_parts.append(f"✅ Waste rate {waste_rate:.1%} within limit.")
        else:
            feedback_parts.append(
                f"❌ Waste rate {waste_rate:.1%} exceeds {self.WASTE_RATE_PASS:.0%} limit."
            )

        return GradeResult(
            score=score,
            passed=passed,
            breakdown=breakdown,
            feedback=" ".join(feedback_parts),
        )


# ===========================================================================
# Task 3 Grader — Disruption Response (Hard)
# ===========================================================================
# Four components, all must be navigated well for a high score:
#
#  1. Full-episode fill rate (35%)     — overall availability across 28 days
#  2. Disruption-window fill rate (35%)— days 10-14 specifically (hardest window)
#  3. Waste rate (15%)                 — perishable management under crisis
#  4. Emergency order ratio (15%)      — penalises over-reliance on costly emergency orders
#
# Passing requires all four components above their thresholds.
# ===========================================================================


class GraderTask3(BaseGrader):
    # Thresholds for "passed"
    FILL_FULL_PASS = 0.75
    FILL_DISRUPTION_PASS = 0.60
    WASTE_RATE_PASS = 0.15
    EMERGENCY_RATIO_PASS = 0.20  # emergency units / total DC reorder units

    # Score floors (score=0 at or below these)
    FILL_FULL_FLOOR = 0.50
    FILL_DISRUPTION_FLOOR = 0.40
    WASTE_RATE_CEIL = 0.35
    EMERGENCY_RATIO_CEIL = 0.40

    DISRUPTION_DAYS = list(range(10, 15))  # days 10-14 inclusive

    WEIGHT_FILL_FULL = 0.35
    WEIGHT_FILL_DISRUPTION = 0.35
    WEIGHT_WASTE = 0.15
    WEIGHT_EMERGENCY = 0.15

    def grade(self, trajectory: Trajectory, total_received: int = 0) -> GradeResult:
        fill_full = trajectory.fill_rate()
        fill_disruption = trajectory.fill_rate(days=self.DISRUPTION_DAYS)
        waste_rate = trajectory.waste_rate(total_received)
        emergency_ratio = self._emergency_ratio(trajectory)

        fill_full_score = self._linear_scale(
            fill_full, self.FILL_FULL_FLOOR, self.FILL_FULL_PASS
        )
        fill_disruption_score = self._linear_scale(
            fill_disruption, self.FILL_DISRUPTION_FLOOR, self.FILL_DISRUPTION_PASS
        )
        waste_score = self._linear_scale(
            self.WASTE_RATE_CEIL - waste_rate, 0.0, self.WASTE_RATE_CEIL
        )
        emergency_score = self._linear_scale(
            self.EMERGENCY_RATIO_CEIL - emergency_ratio, 0.0, self.EMERGENCY_RATIO_CEIL
        )

        score = round(
            self.WEIGHT_FILL_FULL * fill_full_score
            + self.WEIGHT_FILL_DISRUPTION * fill_disruption_score
            + self.WEIGHT_WASTE * waste_score
            + self.WEIGHT_EMERGENCY * emergency_score,
            4,
        )

        passed = (
            fill_full >= self.FILL_FULL_PASS
            and fill_disruption >= self.FILL_DISRUPTION_PASS
            and waste_rate <= self.WASTE_RATE_PASS
            and emergency_ratio <= self.EMERGENCY_RATIO_PASS
        )

        breakdown = {
            "fill_rate_full_episode": round(fill_full, 4),
            "fill_rate_disruption_window": round(fill_disruption, 4),
            "waste_rate": round(waste_rate, 4),
            "emergency_order_ratio": round(emergency_ratio, 4),
            "fill_full_score": round(fill_full_score, 4),
            "fill_disruption_score": round(fill_disruption_score, 4),
            "waste_score": round(waste_score, 4),
            "emergency_score": round(emergency_score, 4),
            "final_score": score,
        }

        feedback = self._build_feedback(
            fill_full, fill_disruption, waste_rate, emergency_ratio, passed
        )

        return GradeResult(
            score=score, passed=passed, breakdown=breakdown, feedback=feedback
        )

    def _emergency_ratio(self, trajectory: Trajectory) -> float:
        """Emergency units as a fraction of all DC reorder units placed."""
        total_emergency = trajectory.total_emergency_units()
        # Total reorder = emergency + standard; we compute standard from step records
        # (Standard orders aren't tracked separately in StepRecord, so we proxy:
        #  emergency_ratio = emergency / (emergency + estimated_standard))
        # A simple, deterministic proxy: total_demand gives scale of operations.
        total_demand = trajectory.total_demand()
        if total_demand == 0:
            return 0.0
        # Ratio of emergency units relative to total demand served (bounded proxy)
        return min(1.0, total_emergency / max(1, total_demand))

    def _build_feedback(
        self,
        fill_full: float,
        fill_disruption: float,
        waste_rate: float,
        emergency_ratio: float,
        passed: bool,
    ) -> str:
        lines = []
        icon = lambda ok: "✅" if ok else "❌"  # noqa: E731
        lines.append(
            f"{icon(fill_full >= self.FILL_FULL_PASS)} "
            f"Full-episode fill rate: {fill_full:.1%} (target ≥{self.FILL_FULL_PASS:.0%})"
        )
        lines.append(
            f"{icon(fill_disruption >= self.FILL_DISRUPTION_PASS)} "
            f"Disruption-window fill rate: {fill_disruption:.1%} (target ≥{self.FILL_DISRUPTION_PASS:.0%})"
        )
        lines.append(
            f"{icon(waste_rate <= self.WASTE_RATE_PASS)} "
            f"Waste rate: {waste_rate:.1%} (target ≤{self.WASTE_RATE_PASS:.0%})"
        )
        lines.append(
            f"{icon(emergency_ratio <= self.EMERGENCY_RATIO_PASS)} "
            f"Emergency order ratio: {emergency_ratio:.1%} (target ≤{self.EMERGENCY_RATIO_PASS:.0%})"
        )
        lines.append("Overall: " + ("✅ PASSED" if passed else "❌ FAILED"))
        return " | ".join(lines)
