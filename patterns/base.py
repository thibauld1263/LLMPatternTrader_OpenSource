"""Pattern detection base class."""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class PatternDetector(ABC):
    """Base class for all pattern detectors."""

    name: str = ""
    category: str = ""

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return +1 (bullish), -1 (bearish), or 0 (no signal) per bar."""
        ...

    def __repr__(self):
        return f"<{self.category}.{self.name}>"
