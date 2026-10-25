"""Result types of the digitisation pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class LeadResult(BaseModel):
    """What was recovered for one lead."""

    name: str
    start_s: float
    duration_s: float
    confidence: float = Field(ge=0, le=1)
    flags: list[str] = []


class DigitizationResult(BaseModel):
    """Summary of one digitisation run (the signal itself is returned alongside)."""

    fs: int = 500
    px_per_mm: float
    scale_source: Literal["grid", "calibration_pulse", "assumed"]
    leads: list[LeadResult]
    global_confidence: float = Field(ge=0, le=1)
    intervals_ms: dict[str, float | None] = {}
    warnings: list[str] = []
