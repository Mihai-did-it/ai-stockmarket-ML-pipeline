# Example: Quick Data Exploration
# This notebook shows how to use individual pipeline components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('..')

# Import components
from src.data_ingestion import PriceDataFetcher
from src.feature_engineering import TechnicalIndicatorCalculator

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Imports successful!")
print("\nThis notebook demonstrates:")
print("1. Fetching price data")
print("2. Calculating technical indicators") 
print("3. Basic visualization")
print("\nRun each cell sequentially")
