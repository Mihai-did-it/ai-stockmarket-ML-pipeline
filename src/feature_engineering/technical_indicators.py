"""
Technical Indicator Calculator
Computes a comprehensive set of technical indicators from OHLCV data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

# Try importing ta-lib, fall back to pandas_ta
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using pandas_ta")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas_ta not available")

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Calculates technical indicators from price data."""
    
    def __init__(self, use_talib: bool = True):
        """
        Initialize calculator.
        
        Args:
            use_talib: Whether to use TA-Lib (faster) or pandas_ta
        """
        self.use_talib = use_talib and TALIB_AVAILABLE
        
        if not self.use_talib and not PANDAS_TA_AVAILABLE:
            raise ImportError("Neither TA-Lib nor pandas_ta available")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
               Should have MultiIndex (date, ticker) or be for single ticker
        
        Returns:
            DataFrame with original data plus indicator columns
        """
        logger.info("Calculating technical indicators")
        
        # Check if multi-ticker data
        if isinstance(df.index, pd.MultiIndex):
            result_dfs = []
            tickers = df.index.get_level_values('ticker').unique()
            
            for ticker in tickers:
                ticker_df = df.xs(ticker, level='ticker').copy()
                ticker_df = self._calculate_indicators_single_ticker(ticker_df)
                ticker_df['ticker'] = ticker
                result_dfs.append(ticker_df)
            
            result = pd.concat(result_dfs, axis=0)
            result = result.set_index(['date', 'ticker']) if 'date' in result.columns else result
            
        else:
            result = self._calculate_indicators_single_ticker(df)
        
        logger.info(f"Calculated {len(result.columns)} total features")
        return result
    
    def _calculate_indicators_single_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for a single ticker."""
        
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract OHLCV
        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        volume = df['volume'].values
        
        # ========== TREND INDICATORS ==========
        
        # Moving Averages
        df['sma_10'] = self._sma(close_price, 10)
        df['sma_20'] = self._sma(close_price, 20)
        df['sma_50'] = self._sma(close_price, 50)
        df['sma_200'] = self._sma(close_price, 200)
        
        df['ema_12'] = self._ema(close_price, 12)
        df['ema_26'] = self._ema(close_price, 26)
        df['ema_50'] = self._ema(close_price, 50)
        
        # MACD
        macd, macd_signal, macd_hist = self._macd(close_price)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # ADX (Average Directional Index)
        df['adx'] = self._adx(high_price, low_price, close_price, period=14)
        
        # Aroon
        aroon_up, aroon_down = self._aroon(high_price, low_price, period=25)
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        
        # Parabolic SAR
        df['parabolic_sar'] = self._sar(high_price, low_price)
        
        # ========== MOMENTUM INDICATORS ==========
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._rsi(close_price, period=14)
        df['rsi_7'] = self._rsi(close_price, period=7)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._stochastic(high_price, low_price, close_price)
        df['stochastic_k'] = stoch_k
        df['stochastic_d'] = stoch_d
        
        # CCI (Commodity Channel Index)
        df['cci'] = self._cci(high_price, low_price, close_price, period=20)
        
        # Williams %R
        df['williams_r'] = self._williams_r(high_price, low_price, close_price, period=14)
        
        # ROC (Rate of Change)
        df['roc_10'] = self._roc(close_price, period=10)
        df['roc_20'] = self._roc(close_price, period=20)
        
        # Momentum
        df['momentum_10'] = self._momentum(close_price, period=10)
        
        # ========== VOLATILITY INDICATORS ==========
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close_price, period=20)
        df['bollinger_upper'] = bb_upper
        df['bollinger_middle'] = bb_middle
        df['bollinger_lower'] = bb_lower
        df['bollinger_width'] = (bb_upper - bb_lower) / bb_middle
        df['bollinger_pct'] = (close_price - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (Average True Range)
        df['atr'] = self._atr(high_price, low_price, close_price, period=14)
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self._keltner_channel(high_price, low_price, close_price)
        df['keltner_upper'] = kc_upper
        df['keltner_lower'] = kc_lower
        
        # Historical Volatility
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()
        
        # ========== VOLUME INDICATORS ==========
        
        # OBV (On-Balance Volume)
        df['obv'] = self._obv(close_price, volume)
        
        # Volume SMA
        df['volume_sma_20'] = self._sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = self._vwap(high_price, low_price, close_price, volume)
        
        # CMF (Chaikin Money Flow)
        df['cmf'] = self._cmf(high_price, low_price, close_price, volume, period=20)
        
        # MFI (Money Flow Index)
        df['mfi'] = self._mfi(high_price, low_price, close_price, volume, period=14)
        
        # Force Index
        df['force_index'] = self._force_index(close_price, volume)
        
        # ========== ADDITIONAL FEATURES ==========
        
        # Price ratios
        df['close_to_sma50'] = close_price / df['sma_50']
        df['close_to_sma200'] = close_price / df['sma_200']
        df['sma50_to_sma200'] = df['sma_50'] / df['sma_200']
        
        # Price changes
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # High-Low spread
        df['hl_spread'] = (high_price - low_price) / close_price
        df['oc_spread'] = (close_price - open_price) / open_price
        
        # Gaps
        df['gap'] = (open_price - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    # ========== HELPER METHODS ==========
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        if self.use_talib:
            return talib.SMA(data, timeperiod=period)
        else:
            return pd.Series(data).rolling(window=period).mean().values
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if self.use_talib:
            return talib.EMA(data, timeperiod=period)
        else:
            return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    def _macd(self, data: np.ndarray, fast=12, slow=26, signal=9):
        """MACD."""
        if self.use_talib:
            return talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        else:
            ema_fast = self._ema(data, fast)
            ema_slow = self._ema(data, slow)
            macd = ema_fast - ema_slow
            macd_signal = self._ema(macd, signal)
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        if self.use_talib:
            return talib.RSI(data, timeperiod=period)
        else:
            delta = pd.Series(data).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.values
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, std: float = 2):
        """Bollinger Bands."""
        if self.use_talib:
            return talib.BBANDS(data, timeperiod=period, nbdevup=std, nbdevdn=std)
        else:
            sma = self._sma(data, period)
            rolling_std = pd.Series(data).rolling(window=period).std().values
            upper = sma + (std * rolling_std)
            lower = sma - (std * rolling_std)
            return upper, sma, lower
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        """Average True Range."""
        if self.use_talib:
            return talib.ATR(high, low, close, timeperiod=period)
        else:
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return pd.Series(true_range).rolling(window=period).mean().values
    
    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        """Average Directional Index."""
        if self.use_talib:
            return talib.ADX(high, low, close, timeperiod=period)
        else:
            # Simplified ADX calculation
            plus_dm = high - np.roll(high, 1)
            minus_dm = np.roll(low, 1) - low
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = self._atr(high, low, close, 1)
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).rolling(period).mean()
            return adx.values
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        """Stochastic Oscillator."""
        if self.use_talib:
            return talib.STOCH(high, low, close, fastk_period=period, 
                             slowk_period=smooth_k, slowd_period=smooth_d)
        else:
            lowest_low = pd.Series(low).rolling(window=period).min()
            highest_high = pd.Series(high).rolling(window=period).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_k = pd.Series(stoch_k).rolling(window=smooth_k).mean()
            stoch_d = stoch_k.rolling(window=smooth_d).mean()
            
            return stoch_k.values, stoch_d.values
    
    def _cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20):
        """Commodity Channel Index."""
        if self.use_talib:
            return talib.CCI(high, low, close, timeperiod=period)
        else:
            tp = (high + low + close) / 3
            sma_tp = pd.Series(tp).rolling(window=period).mean()
            mean_dev = pd.Series(tp).rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            cci = (tp - sma_tp) / (0.015 * mean_dev)
            return cci.values
    
    def _williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        """Williams %R."""
        if self.use_talib:
            return talib.WILLR(high, low, close, timeperiod=period)
        else:
            highest_high = pd.Series(high).rolling(window=period).max()
            lowest_low = pd.Series(low).rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            return williams_r.values
    
    def _roc(self, data: np.ndarray, period: int = 10):
        """Rate of Change."""
        if self.use_talib:
            return talib.ROC(data, timeperiod=period)
        else:
            roc = 100 * (data - np.roll(data, period)) / np.roll(data, period)
            return roc
    
    def _momentum(self, data: np.ndarray, period: int = 10):
        """Momentum."""
        if self.use_talib:
            return talib.MOM(data, timeperiod=period)
        else:
            return data - np.roll(data, period)
    
    def _aroon(self, high: np.ndarray, low: np.ndarray, period: int = 25):
        """Aroon Indicator."""
        if self.use_talib:
            return talib.AROON(high, low, timeperiod=period)
        else:
            aroon_up = pd.Series(high).rolling(period).apply(
                lambda x: (period - (period - 1 - x.argmax())) / period * 100
            )
            aroon_down = pd.Series(low).rolling(period).apply(
                lambda x: (period - (period - 1 - x.argmin())) / period * 100
            )
            return aroon_up.values, aroon_down.values
    
    def _sar(self, high: np.ndarray, low: np.ndarray):
        """Parabolic SAR."""
        if self.use_talib:
            return talib.SAR(high, low)
        else:
            # Simplified SAR
            return self._ema(low, 20)  # Placeholder
    
    def _keltner_channel(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                        period: int = 20, multiplier: float = 2):
        """Keltner Channel."""
        middle = self._ema(close, period)
        atr = self._atr(high, low, close, period)
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return upper, middle, lower
    
    def _obv(self, close: np.ndarray, volume: np.ndarray):
        """On-Balance Volume."""
        if self.use_talib:
            return talib.OBV(close, volume)
        else:
            obv = np.zeros_like(volume, dtype=float)
            obv[0] = volume[0]
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            return obv
    
    def _vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def _cmf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            volume: np.ndarray, period: int = 20):
        """Chaikin Money Flow."""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = np.where(high == low, 0, mfm)  # Avoid division by zero
        mfv = mfm * volume
        cmf = pd.Series(mfv).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
        return cmf.values
    
    def _mfi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            volume: np.ndarray, period: int = 14):
        """Money Flow Index."""
        if self.use_talib:
            return talib.MFI(high, low, close, volume, timeperiod=period)
        else:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = pd.Series([mf if typical_price[i] > typical_price[i-1] else 0 
                                      for i, mf in enumerate(money_flow)])
            negative_flow = pd.Series([mf if typical_price[i] < typical_price[i-1] else 0 
                                      for i, mf in enumerate(money_flow)])
            
            positive_mf = positive_flow.rolling(period).sum()
            negative_mf = negative_flow.rolling(period).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi.values
    
    def _force_index(self, close: np.ndarray, volume: np.ndarray, period: int = 13):
        """Force Index."""
        force = pd.Series(close).diff() * volume
        force_index = force.ewm(span=period, adjust=False).mean()
        return force_index.values


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(300).cumsum() + 100,
        'high': np.random.randn(300).cumsum() + 105,
        'low': np.random.randn(300).cumsum() + 95,
        'close': np.random.randn(300).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 300)
    })
    df = df.set_index('date')
    
    calculator = TechnicalIndicatorCalculator(use_talib=False)
    result = calculator.calculate_all_indicators(df)
    
    print(result.head())
    print(f"\nShape: {result.shape}")
    print(f"\nColumns: {result.columns.tolist()}")
