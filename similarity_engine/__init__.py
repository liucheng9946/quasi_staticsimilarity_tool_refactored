"""安全的 YAML/JSON 扩展组件引擎。"""

from .api import SimilarityAPI
from .models import CalculationOptions, CalculationResult

__all__ = ["SimilarityAPI", "CalculationOptions", "CalculationResult"]
