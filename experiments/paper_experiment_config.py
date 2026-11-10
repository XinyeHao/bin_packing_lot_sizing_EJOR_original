"""Single source of truth for the paper's reproducible experiment suite.

Do not change seeds or time limits inside individual experiment scripts.  The
five paper experiments must use exactly these instance sets and budgets.
"""
from __future__ import annotations

from dataclasses import dataclass


PROCESS_ASSUMPTION = (
    "All items in a batch leave the autoclave and enter inventory together "
    "at the batch maximum curing duration."
)
MODEL_VERSION = "common_unload_v2_span_link_ebrp_bpc_gap1_medium_large"


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    periods: int
    items: int
    seeds: tuple[int, ...]
    time_limit_sec: float


SCALES: dict[str, ScaleSpec] = {
    # These are the manuscript's existing benchmark instances.  Do not
    # replace them with synthetic consecutive seed ranges: all paper
    # experiments must refer to the same three instance sets.
    "small": ScaleSpec("small", 30, 40, tuple(range(10101, 10116)), 900.0),
    "medium": ScaleSpec(
        "medium", 35, 50,
        (30101, 30201, 30301, 30401, 30501,
         31101, 31201, 31301, 31401, 31501,
         31601, 31701, 31801, 31901, 32001),
        1800.0,
    ),
    "large": ScaleSpec(
        "large", 40, 55,
        (40101, 40201, 40301, 40401, 40501,
         41101, 41201, 41301, 41401, 41501,
         41601, 41701, 41801, 41901, 42001),
        3600.0,
    ),
}


def scale_spec(name: str) -> ScaleSpec:
    try:
        return SCALES[name]
    except KeyError as exc:
        raise ValueError(f"scale must be one of {tuple(SCALES)}, got {name!r}") from exc


def make_set_config(spec: ScaleSpec):
    """Build the same generator configuration for every paper experiment."""
    from configuration import SetConfig

    return SetConfig("small_cg", spec.periods, 5, 0, spec.items, 5)


def generation_metadata() -> dict:
    """Snapshot the manuscript's existing benchmark-generation settings.

    Values are read from configuration.py rather than duplicated here, so the
    result file remains tied to the same settings used by the generator.
    """
    import configuration as cfg

    return {
        "set_type": "small_cg",
        "num_configs": 5,
        "num_machines": 5,
        "lead_time_choices": list(cfg.BENCH_LEAD_TIME_CHOICES),
        "time_window": dict(cfg.BENCH_WINDOW),
        "machine_capacity_range": list(cfg.SMALL_CG_GENERATION["machine_capacity_range"]),
        "tray_length_long_probability": cfg.TRAY_LENGTH_LONG_PROB,
        "tray_length_long_range": list(cfg.TRAY_LENGTH_LONG_RANGE),
        "tray_length_short_range": list(cfg.TRAY_LENGTH_SHORT_RANGE),
        "eligible_machine_count_range": list(cfg.SMALL_CG_ITEM_MACHINES_RANGE),
        "demand_periods_per_item": cfg.SMALL_CG_GENERATION["demand_periods_per_item"],
        "scrap_setup_factor": cfg.SMALL_CG_SCRAP_SETUP_FACTOR,
    }
