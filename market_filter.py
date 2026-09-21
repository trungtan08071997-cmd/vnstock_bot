"""
Market Filter - Chỉ giao dịch khi thị trường thuận lợi
"""
import pandas as pd
import numpy as np
from data_fetcher import get_vnindex_history


def check_market_regime(symbol_df=None):
    """
    Kiểm tra thị trường có thuận lợi để giao dịch không.
    
    Returns:
        dict: {
            "allow_trading": bool,
            "regime": str,
            "vnindex": float,
            "ma50": float,
            "reason": str,
        }
    """
    # Lấy VN-Index
    vni = get_vnindex_history(200)
    
    if vni is None or len(vni) < 60:
        # Không có VN-Index → cho phép giao dịch nhưng cảnh báo
        return {
            "allow_trading": True,
            "regime": "UNKNOWN",
            "vnindex": 0,
            "ma50": 0,
            "reason": "Không lấy được VN-Index",
        }
    
    close = vni["close"].values
    current = close[-1]
    ma50 = np.mean(close[-50:])
    ma200 = np.mean(close[-200:]) if len(close) >= 200 else ma50
    
    # Đếm số phiên trên MA50 trong 20 phiên gần
    above_ma50 = sum(1 for c in close[-20:] if c > ma50) if len(close) >= 20 else 10
    ratio_above = above_ma50 / 20 if len(close) >= 20 else 0.5
    
    # Chế độ thị trường
    if current > ma50 and current > ma200 and ratio_above >= 0.6:
        regime = "BULL"
        allow = True
        reason = "VN-Index trên MA50 và MA200"
    elif current > ma50 and ratio_above >= 0.5:
        regime = "RECOVERY"
        allow = True
        reason = "VN-Index trên MA50, đang hồi phục"
    elif current < ma50 and current < ma200:
        regime = "BEAR"
        allow = False
        reason = "VN-Index dưới MA50 và MA200 - không giao dịch"
    else:
        regime = "SIDEWAY"
        allow = False
        reason = "VN-Index đi ngang - chờ tín hiệu"
    
    return {
        "allow_trading": allow,
        "regime": regime,
        "vnindex": round(current, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "reason": reason,
    }
