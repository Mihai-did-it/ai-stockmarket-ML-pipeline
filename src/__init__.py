"""
Stock Market ML Pipeline
A comprehensive machine learning system for stock analysis and prediction.
Developed by Mihai Lache as a personal project.
"""

__version__ = "1.0.0"
__author__ = "Mihai Lache"

# Package imports for convenient access
from . import data_ingestion
from . import feature_engineering
from . import modeling
from . import backtesting
from . import utils

__all__ = [
    'data_ingestion',
    'feature_engineering',
    'modeling',
    'backtesting',
    'utils'
]
