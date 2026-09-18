"""
Đánh giá chế độ thị trường qua VN-Index
"""
from data_fetcher import get_vnindex_history

from technical_analyzer import TechnicalAnalyzer


def get_market_regime():
    df = get_vnindex_history()
    if df is None or len(df) < 50:
        return {"regime": "KHÔNG XÁC ĐỊNH", "vnindex": 0, "change": 0,
                "rsi": 50, "recommendation": "Không đủ dữ liệu"}
    ta = TechnicalAnalyzer(df)
    s = ta.get_current_signals()
    close = s.get("close", 0)
    rsi = s.get("rsi") or 50
    sma20, sma50 = s.get("sma_20"), s.get("sma_50")
    change = 0
    if len(df) >= 2:
        prev = df.iloc[-2]["close"]
        change = (close - prev) / prev * 100 if prev else 0

    if rsi > 70 and sma20 and sma50 and sma20 > sma50:
        regime, rec = "TĂNG NÓNG", "Cảnh giác điều chỉnh, chốt lời dần"
    elif rsi > 55 and sma20 and sma50 and sma20 > sma50:
        regime, rec = "TĂNG", "Duy trì danh mục, có thể mua thêm"
    elif rsi < 30 and sma20 and sma50 and sma20 < sma50:
        regime, rec = "GIẢM MẠNH", "Cơ hội bắt đáy, giải ngân từng phần"
    elif rsi < 45 and sma20 and sma50 and sma20 < sma50:
        regime, rec = "GIẢM", "Thận trọng, ưu tiên phòng thủ"
    else:
        regime, rec = "ĐI NGANG", "Chờ tín hiệu rõ ràng, giao dịch biên"

    return {"regime": regime, "vnindex": round(close, 2),
            "change": round(change, 2), "rsi": round(rsi, 1),
            "recommendation": rec}