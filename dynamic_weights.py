"""
Weight động theo chế độ thị trường
"""

# ===== WEIGHT CHO TỪNG CHẾ ĐỘ =====
REGIME_WEIGHTS = {
    # Trending mạnh + biến động cao
    "TREND_STRONG_VOLATILE": {
        "ema_ribbon": 30,
        "macd": 20,
        "rsi": 5,
        "volume": 15,
        "bollinger": 5,
        "atr": 25,          # ATR cao vì biến động mạnh
    },
    # Trending mạnh + ổn định
    "TREND_STRONG_CALM": {
        "ema_ribbon": 35,
        "macd": 25,
        "rsi": 5,
        "volume": 15,
        "bollinger": 5,
        "atr": 15,
    },
    # Sideway + biến động cao
    "SIDEWAY_VOLATILE": {
        "ema_ribbon": 10,
        "macd": 10,
        "rsi": 20,
        "volume": 15,
        "bollinger": 20,
        "atr": 25,
    },
    # Sideway + ổn định
    "SIDEWAY_CALM": {
        "ema_ribbon": 10,
        "macd": 15,
        "rsi": 25,
        "volume": 15,
        "bollinger": 25,
        "atr": 10,
    },
    # Chuyển tiếp
    "TRANSITION": {
        "ema_ribbon": 20,
        "macd": 20,
        "rsi": 15,
        "volume": 15,
        "bollinger": 20,
        "atr": 10,
    },
}


def get_weights(regime):
    """Lấy weight theo chế độ."""
    return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["TRANSITION"])
