"""向后兼容：配置已迁移至项目根目录 configuration.py。"""

from configuration import (  # noqa: F401
    BOM_CHOICES,
    BOM_FALLBACK_CHOICES,
    DEMAND_CHOICES,
    LEAD_TIME_CHOICES,
    SET_A,
    SET_B,
    SET_CONFIGS,
    SetConfig,
)
