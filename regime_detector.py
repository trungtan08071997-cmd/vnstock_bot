"""
Phát hiện chế độ thị trường để điều chỉnh weight động
"""
import pandas as pd
import numpy as np


def calculate_adx(df, period=14):
    """Tính ADX - đo độ mạnh xu hướng."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Khi cả 2 cùng dương, chọn cái lớn hơn
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx.iloc[-1] if not adx.empty else 20


def detect_regime(df, atr_pct=None):
    """
    Phát hiện chế độ thị trường.
    
    Returns:
        str: TREND_STRONG_VOLATILE | TREND_STRONG_CALM | 
             SIDEWAY_VOLATILE | SIDEWAY_CALM | TRANSITION
    """
    try:
        adx = calculate_adx(df, 14)
    except Exception:
        adx = 20

    if atr_pct is None:
        # Tính ATR% nếu chưa có
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_pct = atr / df['close'].iloc[-1] * 100

    # Phân loại
    if adx >= 25:
        if atr_pct >= 3.0:
            return "TREND_STRONG_VOLATILE"
        return "TREND_STRONG_CALM"
    elif adx <= 20:
        if atr_pct >= 3.0:
            return "SIDEWAY_VOLATILE"
        return "SIDEWAY_CALM"
    else:
        return "TRANSITION"
