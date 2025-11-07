"""
Feature Engineering Module
Computes technical indicators and merges all data sources.
"""

from .technical_indicators import TechnicalIndicatorCalculator
from .feature_merger import FeatureMerger
from .target_generator import TargetGenerator

__all__ = [
    'TechnicalIndicatorCalculator',
    'FeatureMerger',
    'TargetGenerator'
]
