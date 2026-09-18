"""
Phân tích kỹ thuật chuyên sâu với 6 chỉ báo:
- EMA Ribbon (8/13/21/34/55)
- MACD
- RSI
- Volume
- Bollinger Bands
- ATR
"""
import pandas as pd
import numpy as np


# ===== TRỌNG SỐ CHỈ BÁO =====
INDICATOR_WEIGHTS = {
    "ema_ribbon": 25,   # 25 điểm - quan trọng nhất (trend)
    "macd": 20,         # 20 điểm - momentum
    "rsi": 15,          # 15 điểm - oscillator
    "volume": 15,       # 15 điểm - xác nhận
    "bollinger": 15,    # 15 điểm - biến động
    "atr": 10,          # 10 điểm - rủi ro
}


class TechnicalAnalyzer:
    def __init__(self, df):
        """
        Args:
            df: DataFrame OHLCV (time, open, high, low, close, volume)
        """
        self.df = df.copy()
        self._calculate_indicators()

    # ============ TÍNH TOÁN CHỈ BÁO ============
    def _calculate_indicators(self):
        d = self.df

        # ===== EMA RIBBON (8/13/21/34/55) =====
        for period in [8, 13, 21, 34, 55]:
            d[f"ema_{period}"] = d["close"].ewm(span=period, adjust=False).mean()

        # ===== SMA (bổ trợ) =====
        d["sma_20"] = d["close"].rolling(20).mean()
        d["sma_50"] = d["close"].rolling(50).mean()
        d["sma_200"] = d["close"].rolling(200).mean()

        # ===== MACD =====
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["macd"] = ema12 - ema26
        d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"] = d["macd"] - d["macd_signal"]

        # ===== RSI (Wilder's) =====
        delta = d["close"].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi_14"] = 100 - (100 / (1 + rs))

        # ===== Bollinger Bands =====
        d["bb_middle"] = d["close"].rolling(20).mean()
        std = d["close"].rolling(20).std()
        d["bb_upper"] = d["bb_middle"] + 2 * std
        d["bb_lower"] = d["bb_middle"] - 2 * std
        d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"] * 100

        # ===== ATR (Average True Range) =====
        high_low = d["high"] - d["low"]
        high_close = (d["high"] - d["close"].shift()).abs()
        low_close = (d["low"] - d["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        d["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        d["atr_pct"] = d["atr_14"] / d["close"] * 100  # ATR % giá

        # ===== Volume =====
        d["volume_sma_20"] = d["volume"].rolling(20).mean()
        d["volume_ratio"] = d["volume"] / d["volume_sma_20"]

    # ============ TÍN HIỆU HIỆN TẠI ============
    def get_current_signals(self):
        if len(self.df) < 55:
            return {}
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) >= 2 else latest

        return {
            "close": latest.get("close"),
            "prev_close": prev.get("close"),
            # EMA Ribbon
            "ema_8": latest.get("ema_8"),
            "ema_13": latest.get("ema_13"),
            "ema_21": latest.get("ema_21"),
            "ema_34": latest.get("ema_34"),
            "ema_55": latest.get("ema_55"),
            # SMA
            "sma_20": latest.get("sma_20"),
            "sma_50": latest.get("sma_50"),
            "sma_200": latest.get("sma_200"),
            # MACD
            "macd": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "macd_hist": latest.get("macd_hist"),
            "prev_macd_hist": prev.get("macd_hist"),
            # RSI
            "rsi": latest.get("rsi_14"),
            "prev_rsi": prev.get("rsi_14"),
            # Bollinger
            "bb_upper": latest.get("bb_upper"),
            "bb_middle": latest.get("bb_middle"),
            "bb_lower": latest.get("bb_lower"),
            "bb_width": latest.get("bb_width"),
            # ATR
            "atr": latest.get("atr_14"),
            "atr_pct": latest.get("atr_pct"),
            # Volume
            "volume": latest.get("volume"),
            "volume_sma_20": latest.get("volume_sma_20"),
            "volume_ratio": latest.get("volume_ratio"),
        }

    # ============ 1. EMA RIBBON (25 điểm) ============
    def _score_ema_ribbon(self, s):
        """
        Đánh giá xu hướng qua EMA Ribbon.
        - Full bull (8>13>21>34>55): 25 điểm
        - Bull rút gọn (8>13>21): 18 điểm
        - Chuyển tiếp: 10 điểm
        - Bear: 0-5 điểm
        """
        ema8, ema13, ema21, ema34, ema55 = (
            s.get("ema_8"), s.get("ema_13"), s.get("ema_21"),
            s.get("ema_34"), s.get("ema_55")
        )
        close = s.get("close", 0)

        if any(v is None or pd.isna(v) for v in [ema8, ema13, ema21, ema34, ema55]):
            return 0, {"status": "no_data"}

        score = 0
        status = ""
        detail = {}

        # Đếm số cặp EMA xếp đúng thứ tự tăng dần
        pairs = [(ema8, ema13), (ema13, ema21), (ema21, ema34), (ema34, ema55)]
        bull_pairs = sum(1 for a, b in pairs if a > b)

        if bull_pairs == 4:
            status = "FULL_BULL"
            score = 25
            detail["note"] = "EMA ribbon mở rộng hoàn hảo - xu hướng tăng mạnh"
        elif bull_pairs == 3:
            status = "BULL"
            score = 20
            detail["note"] = "EMA ribbon tăng - xu hướng tích cực"
        elif bull_pairs == 2:
            status = "WEAK_BULL"
            score = 12
            detail["note"] = "EMA ribbon chuyển tiếp - thận trọng"
        elif bull_pairs == 1:
            status = "WEAK_BEAR"
            score = 5
            detail["note"] = "EMA ribbon yếu - xu hướng giảm"
        else:
            status = "FULL_BEAR"
            score = 0
            detail["note"] = "EMA ribbon đảo ngược - xu hướng giảm mạnh"

        # Bonus nếu giá trên EMA8 (vào lệnh an toàn hơn)
        if close > ema8:
            score = min(25, score + 3)
            detail["close_above_ema8"] = True

        # Bonus nếu EMA8 > EMA21 > EMA55 (trend rõ ràng)
        if ema8 > ema21 > ema55:
            detail["trend_clean"] = True

        detail["status"] = status
        detail["bull_pairs"] = bull_pairs
        return score, detail

    # ============ 2. MACD (20 điểm) ============
    def _score_macd(self, s):
        """
        Đánh giá MACD.
        - MACD > Signal + Histogram dương tăng: 20 điểm
        - MACD > Signal: 15 điểm
        - MACD vừa cắt lên: 18 điểm
        - MACD < Signal: 0-8 điểm
        """
        macd = s.get("macd")
        sig = s.get("macd_signal")
        hist = s.get("macd_hist")
        prev_hist = s.get("prev_macd_hist")

        if any(v is None or pd.isna(v) for v in [macd, sig, hist]):
            return 0, {"status": "no_data"}

        score = 0
        detail = {}

        # MACD trên Signal?
        if macd > sig:
            score = 12
            detail["above_signal"] = True
        else:
            score = 3
            detail["above_signal"] = False

        # Histogram dương?
        if hist > 0:
            score += 5
            detail["hist_positive"] = True
        else:
            detail["hist_positive"] = False

        # Histogram đang tăng (momentum mạnh lên)?
        if prev_hist is not None and not pd.isna(prev_hist):
            if hist > prev_hist:
                score += 3
                detail["hist_rising"] = True
            else:
                detail["hist_rising"] = False

        # MACD vừa cắt lên Signal (tín hiệu mua mạnh)?
        if macd > sig and prev_hist is not None and prev_hist < 0 and hist > 0:
            score = max(score, 20)
            detail["fresh_cross"] = "bullish"
        elif macd < sig and prev_hist is not None and prev_hist > 0 and hist < 0:
            score = 0
            detail["fresh_cross"] = "bearish"

        # MACD trên 0?
        if macd > 0:
            detail["above_zero"] = True

        return min(20, score), detail

    # ============ 3. RSI (15 điểm) ============
    def _score_rsi(self, s):
        """
        Đánh giá RSI.
        - 40-60: vùng lý tưởng để mua: 15 điểm
        - 30-40: quá bán nhẹ - cơ hội: 13 điểm
        - <30: quá bán mạnh: 10 điểm (có thể bắt đáy)
        - 60-70: tăng nóng nhẹ: 8 điểm
        - >70: quá mua: 2 điểm
        """
        rsi = s.get("rsi")
        if rsi is None or pd.isna(rsi):
            return 0, {"status": "no_data"}

        score = 0
        detail = {"value": round(rsi, 1), "status": ""}

        if 40 <= rsi <= 60:
            score = 15
            detail["status"] = "neutral_bullish"
            detail["note"] = "Vùng lý tưởng, dư địa tăng tốt"
        elif 30 <= rsi < 40:
            score = 13
            detail["status"] = "oversold_mild"
            detail["note"] = "Quá bán nhẹ - cơ hội mua"
        elif rsi < 30:
            score = 10
            detail["status"] = "oversold_strong"
            detail["note"] = "Quá bán mạnh - chờ xác nhận đảo chiều"
        elif 60 < rsi <= 70:
            score = 8
            detail["status"] = "overbought_mild"
            detail["note"] = "Tăng nóng nhẹ - cần thận trọng"
        else:  # > 70
            score = 2
            detail["status"] = "overbought_strong"
            detail["note"] = "Quá mua - rủi ro điều chỉnh cao"

        # RSI đang tăng?
        prev_rsi = s.get("prev_rsi")
        if prev_rsi is not None and not pd.isna(prev_rsi):
            if rsi > prev_rsi:
                detail["rising"] = True
            else:
                detail["rising"] = False

        return score, detail

    # ============ 4. VOLUME (15 điểm) ============
    def _score_volume(self, s):
        """
        Đánh giá Volume so với SMA20.
        - Volume > 2x: đột biến mạnh: 15 điểm
        - 1.5-2x: tăng mạnh: 13 điểm
        - 1.0-1.5x: bình thường: 10 điểm
        - 0.7-1.0x: hơi thấp: 6 điểm
        - < 0.7x: thấp: 3 điểm
        """
        ratio = s.get("volume_ratio")
        if ratio is None or pd.isna(ratio):
            return 0, {"status": "no_data"}

        detail = {"ratio": round(ratio, 2), "status": ""}

        if ratio >= 2.0:
            score = 15
            detail["status"] = "surge"
            detail["note"] = "Volume đột biến - xác nhận mạnh"
        elif ratio >= 1.5:
            score = 13
            detail["status"] = "high"
            detail["note"] = "Volume cao - tín hiệu tốt"
        elif ratio >= 1.0:
            score = 10
            detail["status"] = "normal"
            detail["note"] = "Volume bình thường"
        elif ratio >= 0.7:
            score = 6
            detail["status"] = "low"
            detail["note"] = "Volume hơi thấp - thiếu xác nhận"
        else:
            score = 3
            detail["status"] = "very_low"
            detail["note"] = "Volume rất thấp - không nên vào lệnh"

        return score, detail

    # ============ 5. BOLLINGER BANDS (15 điểm) ============
    def _score_bollinger(self, s):
        """
        Đánh giá vị trí giá so với Bollinger Bands.
        - Giá chạm BB lower + RSI thấp: cơ hội: 15 điểm
        - Giá giữa BB: 10 điểm
        - Giá chạm BB upper: cảnh báo: 3 điểm
        - BB width nén: chuẩn bị breakout
        """
        close = s.get("close", 0)
        upper = s.get("bb_upper")
        lower = s.get("bb_lower")
        middle = s.get("bb_middle")
        width = s.get("bb_width")

        if any(v is None or pd.isna(v) for v in [upper, lower, middle]):
            return 0, {"status": "no_data"}

        score = 5
        detail = {}
        bb_range = upper - lower

        if bb_range <= 0:
            return 5, {"status": "no_range"}

        # Vị trí giá trong dải BB (0 = lower, 1 = upper)
        position = (close - lower) / bb_range
        detail["position"] = round(position, 2)

        if position < 0:
            score = 15
            detail["status"] = "below_lower"
            detail["note"] = "Giá dưới dải dưới - quá bán mạnh"
        elif position < 0.2:
            score = 14
            detail["status"] = "near_lower"
            detail["note"] = "Gần dải dưới - cơ hội mua"
        elif position < 0.4:
            score = 12
            detail["status"] = "lower_half"
            detail["note"] = "Nửa dưới dải - tích cực"
        elif position < 0.6:
            score = 10
            detail["status"] = "middle"
            detail["note"] = "Giữa dải - trung tính"
        elif position < 0.8:
            score = 6
            detail["status"] = "upper_half"
            detail["note"] = "Nửa trên dải - thận trọng"
        elif position < 1.0:
            score = 3
            detail["status"] = "near_upper"
            detail["note"] = "Gần dải trên - rủi ro cao"
        else:
            score = 2
            detail["status"] = "above_upper"
            detail["note"] = "Vượt dải trên - quá mua mạnh"

        # BB width nén (chuẩn bị breakout)
        if width is not None and not pd.isna(width):
            detail["width"] = round(width, 2)
            if width < 5:  # Nén mạnh
                detail["squeeze"] = True
                score = min(15, score + 2)

        return score, detail

    # ============ 6. ATR (10 điểm) ============
    def _score_atr(self, s):
        """
        Đánh giá ATR để quản trị rủi ro.
        - ATR% 1-3%: biến động tốt: 10 điểm
        - ATR% 3-5%: biến động cao: 6 điểm
        - ATR% < 1%: quá ít biến động: 4 điểm
        - ATR% > 5%: quá rủi ro: 2 điểm
        """
        atr_pct = s.get("atr_pct")
        if atr_pct is None or pd.isna(atr_pct):
            return 0, {"status": "no_data"}

        detail = {"atr_pct": round(atr_pct, 2), "status": ""}

        if 1.0 <= atr_pct <= 3.0:
            score = 10
            detail["status"] = "optimal"
            detail["note"] = "Biến động lý tưởng cho swing trade"
        elif 3.0 < atr_pct <= 5.0:
            score = 6
            detail["status"] = "high"
            detail["note"] = "Biến động cao - giảm size"
        elif atr_pct < 1.0:
            score = 4
            detail["status"] = "low"
            detail["note"] = "Biến động thấp - khó có lợi nhuận"
        else:  # > 5%
            score = 2
            detail["status"] = "extreme"
            detail["note"] = "Biến động cực cao - rủi ro lớn"

        return score, detail

    # ============ TỔNG ĐIỂM KỸ THUẬT ============
    def technical_score(self):
        """
        Chấm điểm kỹ thuật (0-100) và độ tin cậy.
        
        Returns:
            (score, details)
        """
        s = self.get_current_signals()
        if not s:
            return 0, {"error": "Không đủ dữ liệu"}

        # Chấm từng chỉ báo
        ema_score, ema_detail = self._score_ema_ribbon(s)
        macd_score, macd_detail = self._score_macd(s)
        rsi_score, rsi_detail = self._score_rsi(s)
        vol_score, vol_detail = self._score_volume(s)
        bb_score, bb_detail = self._score_bollinger(s)
        atr_score, atr_detail = self._score_atr(s)

        total = ema_score + macd_score + rsi_score + vol_score + bb_score + atr_score

        # ===== ĐỘ TIN CẬY =====
        # Đếm số chỉ báo "ủng hộ mua"
        bullish_count = 0
        if ema_score >= 18: bullish_count += 1
        if macd_score >= 15: bullish_count += 1
        if rsi_score >= 13: bullish_count += 1
        if vol_score >= 13: bullish_count += 1
        if bb_score >= 12: bullish_count += 1
        if atr_score >= 6: bullish_count += 1

        # Đếm chỉ báo "cảnh báo bán"
        bearish_count = 0
        if ema_score <= 5: bearish_count += 1
        if macd_score <= 5: bearish_count += 1
        if rsi_score <= 5: bearish_count += 1
        if bb_score <= 5: bearish_count += 1

        if bullish_count >= 5:
            confidence = "HIGH"
        elif bullish_count >= 3:
            confidence = "MEDIUM"
        elif bearish_count >= 3:
            confidence = "LOW"
        else:
            confidence = "LOW"

        # ===== KHUYẾN NGHỊ VÀO LỆNH =====
        signal, entry_action, entry_reason = self._entry_decision(
            total, confidence, bullish_count, bearish_count, s
        )

        # ===== STOP LOSS & TAKE PROFIT (dựa trên ATR) =====
        close = s.get("close", 0)
        atr = s.get("atr", 0)
        stop_loss = round(close - 1.5 * atr, 2) if atr else None
        take_profit_1 = round(close + 2.0 * atr, 2) if atr else None
        take_profit_2 = round(close + 3.5 * atr, 2) if atr else None
        risk_reward = round((take_profit_1 - close) / (close - stop_loss), 2) \
            if stop_loss and close > stop_loss else None

        details = {
            "score_breakdown": {
                "ema_ribbon": {"score": ema_score, "max": 25, **ema_detail},
                "macd":       {"score": macd_score, "max": 20, **macd_detail},
                "rsi":        {"score": rsi_score,  "max": 15, **rsi_detail},
                "volume":     {"score": vol_score,  "max": 15, **vol_detail},
                "bollinger":  {"score": bb_score,   "max": 15, **bb_detail},
                "atr":        {"score": atr_score,  "max": 10, **atr_detail},
            },
            "confidence": confidence,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "entry_action": entry_action,
            "entry_reason": entry_reason,
            "risk_management": {
                "close": close,
                "atr": round(atr, 2) if atr else None,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "risk_reward_ratio": risk_reward,
            },
        }

        return min(100, total), details

    # ============ QUYẾT ĐỊNH VÀO LỆNH ============
    def _entry_decision(self, score, confidence, bull, bear, s):
        """
        Đưa ra quyết định vào lệnh dựa trên score + confidence.
        
        Returns:
            (signal, entry_action, reason)
        """
        rsi = s.get("rsi", 50)

        # ===== MUA MẠNH =====
        if score >= 80 and confidence == "HIGH":
            return ("MUA MẠNH", "ENTER_NOW",
                    f"Điểm {score} + confidence cao. Vào lệnh ngay 70% vị thế, "
                    f"gia tăng khi vượt đỉnh gần nhất.")

        # ===== MUA =====
        if score >= 65 and confidence in ("HIGH", "MEDIUM"):
            return ("MUA", "ENTER_PARTIAL",
                    f"Điểm {score} + confidence {confidence}. Vào 40-50% vị thế, "
                    f"chờ xác nhận thêm để gia tăng.")

        # ===== MUA THĂM DÒ =====
        if score >= 55 and bull >= 3:
            return ("MUA THĂM DÒ", "ENTER_SMALL",
                    f"Điểm {score}. Vào 20-30% vị thế để thăm dò, "
                    f"cắt lỗ nếu giảm dưới ATR.")

        # ===== CHỜ =====
        if 45 <= score < 55:
            return ("CHỜ", "WAIT",
                    f"Điểm {score} - tín hiệu chưa rõ. Chờ breakout hoặc "
                    f"chờ RSI về vùng lý tưởng.")

        # ===== KHÔNG VÀO =====
        if 30 <= score < 45:
            return ("KHÔNG VÀO", "AVOID",
                    f"Điểm {score} - tín hiệu yếu. Không nên vào lệnh lúc này.")

        # ===== BÁN / ĐỨNG NGOÀI =====
        if score < 30 or bear >= 3:
            return ("BÁN", "EXIT_OR_SHORT",
                    f"Điểm {score} - tín hiệu tiêu cực. Thoát hàng nếu đang giữ, "
                    f"cân nhắc short nếu có.")

        return ("TRUNG TÍNH", "NEUTRAL", "Tín hiệu hỗn hợp, không hành động.")