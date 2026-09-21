"""
Technical Analyzer - Phiên bản nâng cao
- Dynamic weights theo chế độ thị trường (TREND/SIDEWAY)
- EMA Ribbon 3 khung: intraday (5,9,21), swing (8,13,21,34,55), position (50,100,200)
- RSI threshold ĐỘNG theo regime
- 6 chỉ báo: EMA, MACD, RSI, Volume, Bollinger, ATR
- Risk management: SL/TP dựa trên ATR
"""
import pandas as pd
import numpy as np


# ============================================================
# DYNAMIC WEIGHTS THEO REGIME
# ============================================================
REGIME_WEIGHTS = {
    "TREND_STRONG_VOLATILE": {
        "ema_ribbon": 30, "macd": 20, "rsi": 5,
        "volume": 15, "bollinger": 5, "atr": 25,
    },
    "TREND_STRONG_CALM": {
        "ema_ribbon": 35, "macd": 25, "rsi": 5,
        "volume": 15, "bollinger": 5, "atr": 15,
    },
    "SIDEWAY_VOLATILE": {
        "ema_ribbon": 10, "macd": 10, "rsi": 20,
        "volume": 15, "bollinger": 20, "atr": 25,
    },
    "SIDEWAY_CALM": {
        "ema_ribbon": 10, "macd": 15, "rsi": 25,
        "volume": 15, "bollinger": 25, "atr": 10,
    },
    "TRANSITION": {
        "ema_ribbon": 20, "macd": 20, "rsi": 15,
        "volume": 15, "bollinger": 20, "atr": 10,
    },
}


# ============================================================
# CLASS CHÍNH
# ============================================================
class TechnicalAnalyzer:
    def __init__(self, df):
        """
        Args:
            df: DataFrame OHLCV (time, open, high, low, close, volume)
        """
        self.df = df.copy()
        self._calculate_indicators()
        self.regime = self._detect_regime()
        self.weights = REGIME_WEIGHTS.get(self.regime, REGIME_WEIGHTS["TRANSITION"])

    # ============================================================
    # 1. TÍNH TOÁN CHỈ BÁO
    # ============================================================
    def _calculate_indicators(self):
        d = self.df

        # ===== EMA RIBBON - KHUNG NGẮN (intraday 5,9,21) =====
        for p in [5, 9, 21]:
            d[f"ema_s_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

        # ===== EMA RIBBON - KHUNG TRUNG (swing 8,13,21,34,55) =====
        for p in [8, 13, 21, 34, 55]:
            d[f"ema_m_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

        # ===== EMA RIBBON - KHUNG DÀI (position 50,100,200) =====
        for p in [50, 100, 200]:
            d[f"ema_l_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

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

        # ===== ATR =====
        high_low = d["high"] - d["low"]
        high_close = (d["high"] - d["close"].shift()).abs()
        low_close = (d["low"] - d["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        d["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        d["atr_pct"] = d["atr_14"] / d["close"] * 100

        # ===== Volume =====
        d["volume_sma_20"] = d["volume"].rolling(20).mean()
        d["volume_ratio"] = d["volume"] / d["volume_sma_20"]

        # ===== ADX (để detect regime) =====
        plus_dm = d["high"].diff()
        minus_dm = -d["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / d["atr_14"]
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / d["atr_14"]
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        d["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # ============================================================
    # 2. DETECT REGIME
    # ============================================================
    def _detect_regime(self):
        """
        Phát hiện chế độ thị trường.
        - ADX >= 25: TREND_STRONG
        - ADX <= 20: SIDEWAY
        - ATR% >= 3.0: VOLATILE
        """
        if len(self.df) < 30:
            return "TRANSITION"

        latest = self.df.iloc[-1]
        adx = latest.get("adx_14", 20)
        atr_pct = latest.get("atr_pct", 2.0)

        if pd.isna(adx):
            adx = 20
        if pd.isna(atr_pct):
            atr_pct = 2.0

        if adx >= 25:
            return "TREND_STRONG_VOLATILE" if atr_pct >= 3.0 else "TREND_STRONG_CALM"
        elif adx <= 20:
            return "SIDEWAY_VOLATILE" if atr_pct >= 3.0 else "SIDEWAY_CALM"
        else:
            return "TRANSITION"

    # ============================================================
    # 3. LẤY TÍN HIỆU HIỆN TẠI
    # ============================================================
    def get_current_signals(self):
        if len(self.df) < 55:
            return {}
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) >= 2 else latest

        return {
            "close": latest.get("close"),
            "prev_close": prev.get("close"),
            # EMA khung ngắn
            "ema_s_5": latest.get("ema_s_5"),
            "ema_s_9": latest.get("ema_s_9"),
            "ema_s_21": latest.get("ema_s_21"),
            # EMA khung trung
            "ema_m_8": latest.get("ema_m_8"),
            "ema_m_13": latest.get("ema_m_13"),
            "ema_m_21": latest.get("ema_m_21"),
            "ema_m_34": latest.get("ema_m_34"),
            "ema_m_55": latest.get("ema_m_55"),
            # EMA khung dài
            "ema_l_50": latest.get("ema_l_50"),
            "ema_l_100": latest.get("ema_l_100"),
            "ema_l_200": latest.get("ema_l_200"),
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
            # ADX
            "adx": latest.get("adx_14"),
        }

    # ============================================================
    # 4. CHẤM ĐIỂM EMA RIBBON (3 khung)
    # ============================================================
    def _score_ema_ribbon(self, s):
        """
        Chấm điểm EMA với 3 khung:
        - Intraday (5,9,21): 30%
        - Swing (8,13,21,34,55): 50%
        - Position (50,100,200): 20%
        """
        close = s.get("close", 0)

        # ===== KHUNG NGẮN (Intraday) =====
        intra_score = 0
        ema5 = s.get("ema_s_5")
        ema9 = s.get("ema_s_9")
        ema21 = s.get("ema_s_21")

        if all(v is not None and not pd.isna(v) for v in [ema5, ema9, ema21]):
            if close > ema5 > ema9 > ema21:
                intra_score = 100
                intra_sig = "STRONG_BULL_INTRADAY"
            elif close > ema9 > ema21:
                intra_score = 75
                intra_sig = "BULL_INTRADAY"
            elif close < ema5 < ema9 < ema21:
                intra_score = 0
                intra_sig = "STRONG_BEAR_INTRADAY"
            elif close < ema9 < ema21:
                intra_score = 25
                intra_sig = "BEAR_INTRADAY"
            else:
                intra_score = 50
                intra_sig = "NEUTRAL_INTRADAY"
        else:
            intra_sig = "NO_DATA"

        # ===== KHUNG TRUNG (Swing) =====
        swing_score = 0
        ema8 = s.get("ema_m_8")
        ema13 = s.get("ema_m_13")
        ema21m = s.get("ema_m_21")
        ema34 = s.get("ema_m_34")
        ema55 = s.get("ema_m_55")

        bull_pairs = 0
        if all(v is not None and not pd.isna(v) for v in
               [ema8, ema13, ema21m, ema34, ema55]):
            pairs = [(ema8, ema13), (ema13, ema21m),
                     (ema21m, ema34), (ema34, ema55)]
            bull_pairs = sum(1 for a, b in pairs if a > b)

            swing_map = {4: 100, 3: 80, 2: 50, 1: 20, 0: 0}
            swing_score = swing_map.get(bull_pairs, 0)

            if bull_pairs == 4:
                swing_sig = "FULL_BULL"
            elif bull_pairs == 3:
                swing_sig = "BULL"
            elif bull_pairs == 2:
                swing_sig = "WEAK_BULL"
            elif bull_pairs == 1:
                swing_sig = "WEAK_BEAR"
            else:
                swing_sig = "FULL_BEAR"
        else:
            swing_sig = "NO_DATA"

        # ===== KHUNG DÀI (Position) =====
        pos_score = 0
        ema50 = s.get("ema_l_50")
        ema100 = s.get("ema_l_100")
        ema200 = s.get("ema_l_200")

        if all(v is not None and not pd.isna(v) for v in [ema50, ema100, ema200]):
            if close > ema50 > ema100 > ema200:
                pos_score = 100
                pos_sig = "LONG_BULL"
            elif close > ema50 > ema200:
                pos_score = 75
                pos_sig = "BULL"
            elif close < ema50 < ema100 < ema200:
                pos_score = 0
                pos_sig = "LONG_BEAR"
            else:
                pos_score = 40
                pos_sig = "MIXED"
        else:
            pos_sig = "NO_DATA"

        # ===== TỔNG HỢP =====
        final_raw = (
            intra_score * 0.30
            + swing_score * 0.50
            + pos_score * 0.20
        )

        # Scale về [0, 25] để tương thích với max cũ
        final_score = final_raw / 100 * 25

        detail = {
            "status": swing_sig,
            "bull_pairs": bull_pairs,
            "intraday": {
                "signal": intra_sig,
                "raw_score": intra_score,
                "ema5": round(ema5, 2) if ema5 else None,
                "ema9": round(ema9, 2) if ema9 else None,
                "ema21": round(ema21, 2) if ema21 else None,
            },
            "swing": {
                "signal": swing_sig,
                "raw_score": swing_score,
                "ema8": round(ema8, 2) if ema8 else None,
                "ema13": round(ema13, 2) if ema13 else None,
                "ema21": round(ema21m, 2) if ema21m else None,
                "ema34": round(ema34, 2) if ema34 else None,
                "ema55": round(ema55, 2) if ema55 else None,
            },
            "position": {
                "signal": pos_sig,
                "raw_score": pos_score,
                "ema50": round(ema50, 2) if ema50 else None,
                "ema100": round(ema100, 2) if ema100 else None,
                "ema200": round(ema200, 2) if ema200 else None,
            },
        }

        return round(final_score, 2), detail

    # ============================================================
    # 5. CHẤM ĐIỂM MACD (max 20)
    # ============================================================
    def _score_macd(self, s):
        macd = s.get("macd")
        sig = s.get("macd_signal")
        hist = s.get("macd_hist")
        prev_hist = s.get("prev_macd_hist")

        if any(v is None or pd.isna(v) for v in [macd, sig, hist]):
            return 0, {"status": "no_data"}

        score = 0
        detail = {}

        if macd > sig:
            score = 12
            detail["above_signal"] = True
        else:
            score = 3
            detail["above_signal"] = False

        if hist > 0:
            score += 5
            detail["hist_positive"] = True
        else:
            detail["hist_positive"] = False

        if prev_hist is not None and not pd.isna(prev_hist):
            if hist > prev_hist:
                score += 3
                detail["hist_rising"] = True
            else:
                detail["hist_rising"] = False

        # Fresh cross
        if macd > sig and prev_hist is not None and prev_hist < 0 and hist > 0:
            score = max(score, 20)
            detail["fresh_cross"] = "bullish"
        elif macd < sig and prev_hist is not None and prev_hist > 0 and hist < 0:
            score = 0
            detail["fresh_cross"] = "bearish"

        if macd > 0:
            detail["above_zero"] = True

        detail["values"] = {
            "macd": round(macd, 3),
            "signal": round(sig, 3),
            "hist": round(hist, 3),
        }
        detail["status"] = "ok"

        return min(20, score), detail

    # ============================================================
    # 6. CHẤM ĐIỂM RSI (max 15) - THRESHOLD ĐỘNG
    # ============================================================
    def _score_rsi(self, s):
        """
        RSI threshold ĐỘNG theo chế độ thị trường.
        - TREND tăng: oversold ở 40, overbought ở 80
        - SIDEWAY: oversold 35, overbought 65
        - VOLATILE: oversold 30, overbought 70
        """
        rsi = s.get("rsi")
        if rsi is None or pd.isna(rsi):
            return 0, {"status": "no_data"}

        regime = self.regime

        # Xác định threshold theo regime
        if "TREND_STRONG" in regime:
            over_sold_mild = 40
            over_bought_mild = 75
            over_sold_strong = 25
            over_bought_strong = 85
        elif "SIDEWAY_CALM" in regime:
            over_sold_mild = 35
            over_bought_mild = 65
            over_sold_strong = 25
            over_bought_strong = 75
        elif "SIDEWAY_VOLATILE" in regime:
            over_sold_mild = 30
            over_bought_mild = 70
            over_sold_strong = 20
            over_bought_strong = 80
        else:  # TRANSITION
            over_sold_mild = 40
            over_bought_mild = 60
            over_sold_strong = 30
            over_bought_strong = 70

        detail = {
            "value": round(rsi, 1),
            "regime": regime,
            "thresholds": {
                "oversold_mild": over_sold_mild,
                "overbought_mild": over_bought_mild,
            },
        }

        # ===== CHẤM ĐIỂM =====
        if over_sold_mild <= rsi <= over_bought_mild:
            score = 15
            detail["status"] = "ideal"
            detail["note"] = f"Vùng lý tưởng ({over_sold_mild}-{over_bought_mild})"
        elif over_sold_strong <= rsi < over_sold_mild:
            score = 13
            detail["status"] = "oversold_mild"
            detail["note"] = f"Quá bán nhẹ (< {over_sold_mild})"
        elif rsi < over_sold_strong:
            score = 10
            detail["status"] = "oversold_strong"
            detail["note"] = f"Quá bán mạnh (< {over_sold_strong})"
        elif over_bought_mild < rsi <= over_bought_strong:
            score = 8
            detail["status"] = "overbought_mild"
            detail["note"] = f"Tăng nóng (> {over_bought_mild})"
        else:
            score = 2
            detail["status"] = "overbought_strong"
            detail["note"] = f"Quá mua mạnh (> {over_bought_strong})"

        prev_rsi = s.get("prev_rsi")
        if prev_rsi is not None and not pd.isna(prev_rsi):
            detail["rising"] = rsi > prev_rsi

        return score, detail

    # ============================================================
    # 7. CHẤM ĐIỂM VOLUME (max 15)
    # ============================================================
    def _score_volume(self, s):
        ratio = s.get("volume_ratio")
        if ratio is None or pd.isna(ratio):
            return 0, {"status": "no_data"}

        detail = {"ratio": round(ratio, 2)}

        if ratio >= 2.0:
            score = 15
            detail["status"] = "surge"
            detail["note"] = "Volume đột biến"
        elif ratio >= 1.5:
            score = 13
            detail["status"] = "high"
            detail["note"] = "Volume cao"
        elif ratio >= 1.0:
            score = 10
            detail["status"] = "normal"
            detail["note"] = "Volume bình thường"
        elif ratio >= 0.7:
            score = 6
            detail["status"] = "low"
            detail["note"] = "Volume thấp"
        else:
            score = 3
            detail["status"] = "very_low"
            detail["note"] = "Volume rất thấp"

        return score, detail

    # ============================================================
    # 8. CHẤM ĐIỂM BOLLINGER (max 15)
    # ============================================================
    def _score_bollinger(self, s):
        close = s.get("close", 0)
        upper = s.get("bb_upper")
        lower = s.get("bb_lower")
        middle = s.get("bb_middle")
        width = s.get("bb_width")

        if any(v is None or pd.isna(v) for v in [upper, lower, middle]):
            return 0, {"status": "no_data"}

        bb_range = upper - lower
        if bb_range <= 0:
            return 5, {"status": "no_range"}

        position = (close - lower) / bb_range
        detail = {"position": round(position, 2), "width": round(width, 2) if width else None}

        # Threshold ĐỘNG theo regime
        if "SIDEWAY" in self.regime:
            # Sideway: gần lower là mua, gần upper là bán
            if position < 0.1:
                score = 15
                detail["status"] = "near_lower"
            elif position < 0.3:
                score = 13
                detail["status"] = "lower_zone"
            elif position < 0.5:
                score = 10
                detail["status"] = "lower_half"
            elif position < 0.7:
                score = 7
                detail["status"] = "upper_half"
            elif position < 0.9:
                score = 4
                detail["status"] = "near_upper"
            else:
                score = 2
                detail["status"] = "above_upper"
        else:
            # Trend: chỉ cần giá không quá xa middle
            if 0.2 <= position <= 0.8:
                score = 12
                detail["status"] = "mid_range"
            elif position < 0.2:
                score = 10
                detail["status"] = "near_lower"
            else:
                score = 6
                detail["status"] = "near_upper"

        if width and width < 5:
            detail["squeeze"] = True
            score = min(15, score + 2)

        return score, detail

    # ============================================================
    # 9. CHẤM ĐIỂM ATR (max 10)
    # ============================================================
    def _score_atr(self, s):
        atr_pct = s.get("atr_pct")
        if atr_pct is None or pd.isna(atr_pct):
            return 0, {"status": "no_data"}

        detail = {"atr_pct": round(atr_pct, 2)}

        if 1.0 <= atr_pct <= 3.0:
            score = 10
            detail["status"] = "optimal"
        elif 3.0 < atr_pct <= 5.0:
            score = 6
            detail["status"] = "high"
        elif atr_pct < 1.0:
            score = 4
            detail["status"] = "low"
        else:
            score = 2
            detail["status"] = "extreme"

        return score, detail

    # ============================================================
    # 10. TỔNG ĐIỂM KỸ THUẬT (dùng weight động)
    # ============================================================
    def technical_score(self):
        """
        Chấm điểm với WEIGHT ĐỘNG theo regime.
        """
        s = self.get_current_signals()
        if not s:
            return 0, {"error": "Không đủ dữ liệu"}

        # Lấy weight động
        W = self.weights
        total_weight = sum(W.values())

        # Chấm từng chỉ báo (raw score với max gốc)
        ema_raw, ema_detail = self._score_ema_ribbon(s)   # max 25
        macd_raw, macd_detail = self._score_macd(s)       # max 20
        rsi_raw, rsi_detail = self._score_rsi(s)          # max 15
        vol_raw, vol_detail = self._score_volume(s)       # max 15
        bb_raw, bb_detail = self._score_bollinger(s)      # max 15
        atr_raw, atr_detail = self._score_atr(s)          # max 10

        # Chuẩn hóa về [0, 100]
        ema_pct = ema_raw / 25 * 100
        macd_pct = macd_raw / 20 * 100
        rsi_pct = rsi_raw / 15 * 100
        vol_pct = vol_raw / 15 * 100
        bb_pct = bb_raw / 15 * 100
        atr_pct_scaled = atr_raw / 10 * 100

        # Tính điểm tổng với weight động
        weighted = (
            ema_pct * W["ema_ribbon"]
            + macd_pct * W["macd"]
            + rsi_pct * W["rsi"]
            + vol_pct * W["volume"]
            + bb_pct * W["bollinger"]
            + atr_pct_scaled * W["atr"]
        ) / total_weight

        total = round(weighted, 1)

        # ===== CONFIDENCE =====
        bullish_count = 0
        if ema_pct >= 70: bullish_count += 1
        if macd_pct >= 70: bullish_count += 1
        if rsi_pct >= 80: bullish_count += 1
        if vol_pct >= 80: bullish_count += 1
        if bb_pct >= 70: bullish_count += 1
        if atr_pct_scaled >= 60: bullish_count += 1

        bearish_count = 0
        if ema_pct <= 25: bearish_count += 1
        if macd_pct <= 25: bearish_count += 1
        if rsi_pct <= 30: bearish_count += 1
        if bb_pct <= 30: bearish_count += 1

        if bullish_count >= 5:
            confidence = "HIGH"
        elif bullish_count >= 3:
            confidence = "MEDIUM"
        elif bearish_count >= 3:
            confidence = "LOW"
        else:
            confidence = "LOW"

        # ===== ENTRY DECISION =====
        signal, entry_action, entry_reason = self._entry_decision(
            total, confidence, bullish_count, bearish_count, s
        )

        # ===== RISK MANAGEMENT =====
        close = s.get("close", 0)
        atr = s.get("atr", 0)
        stop_loss = round(close - 1.5 * atr, 2) if atr else None
        tp1 = round(close + 2.0 * atr, 2) if atr else None
        tp2 = round(close + 3.5 * atr, 2) if atr else None
        rr = round((tp1 - close) / (close - stop_loss), 2) \
            if stop_loss and close > stop_loss else None

        details = {
            "regime": self.regime,
            "weights": W,
            "score_breakdown": {
                "ema_ribbon": {"score": round(ema_raw, 1), "max": 25,
                                "weight": W["ema_ribbon"], **ema_detail},
                "macd":       {"score": round(macd_raw, 1), "max": 20,
                                "weight": W["macd"], **macd_detail},
                "rsi":        {"score": round(rsi_raw, 1), "max": 15,
                                "weight": W["rsi"], **rsi_detail},
                "volume":     {"score": round(vol_raw, 1), "max": 15,
                                "weight": W["volume"], **vol_detail},
                "bollinger":  {"score": round(bb_raw, 1), "max": 15,
                                "weight": W["bollinger"], **bb_detail},
                "atr":        {"score": round(atr_raw, 1), "max": 10,
                                "weight": W["atr"], **atr_detail},
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
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "risk_reward_ratio": rr,
            },
        }

        return min(100, total), details

    # ============================================================
    # 11. ENTRY DECISION
    # ============================================================
    def _entry_decision(self, score, confidence, bull, bear, s):
        rsi = s.get("rsi", 50)
        regime = self.regime

        # Regime TREND_STRONG: chỉ vào lệnh theo xu hướng
        if "TREND_STRONG" in regime:
            if score >= 75 and confidence == "HIGH":
                return ("MUA MẠNH", "ENTER_NOW",
                        f"Trend mạnh + điểm {score}. Vào 70% vị thế.")
            if score >= 60:
                return ("MUA", "ENTER_PARTIAL",
                        f"Trend mạnh + điểm {score}. Vào 40-50% vị thế.")
            if score <= 30:
                return ("BÁN", "EXIT_OR_SHORT",
                        f"Trend đảo chiều. Thoát hàng.")
            return ("GIỮ", "HOLD",
                    f"Trend mạnh nhưng điểm {score} chưa đủ.")

        # Regime SIDEWAY: mua gần lower, bán gần upper
        if "SIDEWAY" in regime:
            if score >= 70 and rsi < 40:
                return ("MUA", "ENTER_PARTIAL",
                        f"Sideway - RSI {rsi:.0f} quá bán. Mua 30-40%.")
            if score <= 35 and rsi > 65:
                return ("BÁN", "EXIT_OR_SHORT",
                        f"Sideway - RSI {rsi:.0f} quá mua. Bán.")
            return ("CHỜ", "WAIT",
                    f"Sideway - chờ RSI về vùng cực.")

        # TRANSITION
        if score >= 70 and confidence in ("HIGH", "MEDIUM"):
            return ("MUA", "ENTER_PARTIAL",
                    f"Điểm {score}, confidence {confidence}. Vào 40%.")
        if score >= 55:
            return ("MUA THĂM DÒ", "ENTER_SMALL",
                    f"Điểm {score}. Vào 20-30% thăm dò.")
        if score >= 40:
            return ("CHỜ", "WAIT", f"Điểm {score} - chờ tín hiệu.")
        return ("KHÔNG VÀO", "AVOID", f"Điểm {score} - không nên vào.")
