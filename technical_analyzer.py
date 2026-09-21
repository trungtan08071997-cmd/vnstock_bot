"""
Technical Analyzer - Phiên bản tối ưu từ Bayesian Optimization
Config áp dụng (từ 100 trials trên VN30):
- Static weights (không dynamic)
- RSI threshold cố định 40/60
- SL = 3.0 ATR, TP = 5.0 ATR
- Entry threshold = 55
- Max hold = 25 ngày
"""
import pandas as pd
import numpy as np
import config


# ============================================================
# STATIC WEIGHTS (config tối ưu)
# ============================================================
STATIC_WEIGHTS = {
    "ema_ribbon": 25,
    "macd": 20,
    "rsi": 15,
    "volume": 15,
    "bollinger": 15,
    "atr": 10,
}

# Dynamic weights (giữ lại để có thể bật lại sau)
REGIME_WEIGHTS = {
    "TREND_STRONG_VOLATILE": {"ema_ribbon": 30, "macd": 20, "rsi": 5,
                              "volume": 15, "bollinger": 5, "atr": 25},
    "TREND_STRONG_CALM":     {"ema_ribbon": 35, "macd": 25, "rsi": 5,
                              "volume": 15, "bollinger": 5, "atr": 15},
    "SIDEWAY_VOLATILE":      {"ema_ribbon": 10, "macd": 10, "rsi": 20,
                              "volume": 15, "bollinger": 20, "atr": 25},
    "SIDEWAY_CALM":          {"ema_ribbon": 10, "macd": 15, "rsi": 25,
                              "volume": 15, "bollinger": 25, "atr": 10},
    "TRANSITION":            {"ema_ribbon": 20, "macd": 20, "rsi": 15,
                              "volume": 15, "bollinger": 20, "atr": 10},
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

        # Đọc config
        tc = getattr(config, "TRADING_CONFIG", {})
        use_dynamic = tc.get("use_dynamic_weights", False)

        if use_dynamic:
            self.regime = self._detect_regime()
            self.weights = REGIME_WEIGHTS.get(
                self.regime, REGIME_WEIGHTS["TRANSITION"]
            )
        else:
            self.regime = "STATIC"
            self.weights = STATIC_WEIGHTS.copy()

    # ============================================================
    # 1. TÍNH CHỈ BÁO
    # ============================================================
    def _calculate_indicators(self):
        d = self.df

        # ===== EMA khung ngắn (intraday) =====
        for p in [5, 9, 21]:
            d[f"ema_s_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

        # ===== EMA khung trung (swing) =====
        for p in [8, 13, 21, 34, 55]:
            d[f"ema_m_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

        # ===== EMA khung dài (position) =====
        for p in [50, 100, 200]:
            d[f"ema_l_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

        # ===== MACD =====
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["macd"] = ema12 - ema26
        d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"] = d["macd"] - d["macd_signal"]

        # ===== RSI =====
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

        # ===== ADX (chỉ dùng nếu bật dynamic) =====
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
    # 2. DETECT REGIME (chỉ dùng nếu bật dynamic)
    # ============================================================
    def _detect_regime(self):
        if len(self.df) < 30:
            return "TRANSITION"
        latest = self.df.iloc[-1]
        adx = latest.get("adx_14", 20)
        atr_pct = latest.get("atr_pct", 2.0)
        if pd.isna(adx): adx = 20
        if pd.isna(atr_pct): atr_pct = 2.0

        if adx >= 25:
            return "TREND_STRONG_VOLATILE" if atr_pct >= 3.0 else "TREND_STRONG_CALM"
        elif adx <= 20:
            return "SIDEWAY_VOLATILE" if atr_pct >= 3.0 else "SIDEWAY_CALM"
        return "TRANSITION"

    # ============================================================
    # 3. LẤY TÍN HIỆU
    # ============================================================
    def get_current_signals(self):
        if len(self.df) < 55:
            return {}
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) >= 2 else latest

        return {
            "close": latest.get("close"),
            "prev_close": prev.get("close"),
            "ema_s_5": latest.get("ema_s_5"),
            "ema_s_9": latest.get("ema_s_9"),
            "ema_s_21": latest.get("ema_s_21"),
            "ema_m_8": latest.get("ema_m_8"),
            "ema_m_13": latest.get("ema_m_13"),
            "ema_m_21": latest.get("ema_m_21"),
            "ema_m_34": latest.get("ema_m_34"),
            "ema_m_55": latest.get("ema_m_55"),
            "ema_l_50": latest.get("ema_l_50"),
            "ema_l_100": latest.get("ema_l_100"),
            "ema_l_200": latest.get("ema_l_200"),
            "macd": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "macd_hist": latest.get("macd_hist"),
            "prev_macd_hist": prev.get("macd_hist"),
            "rsi": latest.get("rsi_14"),
            "prev_rsi": prev.get("rsi_14"),
            "bb_upper": latest.get("bb_upper"),
            "bb_middle": latest.get("bb_middle"),
            "bb_lower": latest.get("bb_lower"),
            "bb_width": latest.get("bb_width"),
            "atr": latest.get("atr_14"),
            "atr_pct": latest.get("atr_pct"),
            "volume": latest.get("volume"),
            "volume_sma_20": latest.get("volume_sma_20"),
            "volume_ratio": latest.get("volume_ratio"),
            "adx": latest.get("adx_14"),
        }

    # ============================================================
    # 4. CHẤM ĐIỂM EMA (max 25)
    # ============================================================
    def _score_ema_ribbon(self, s):
        close = s.get("close", 0)

        # Intraday
        intra_score = 50
        ema5 = s.get("ema_s_5")
        ema9 = s.get("ema_s_9")
        ema21s = s.get("ema_s_21")
        intra_sig = "NEUTRAL"
        if not any(v is None or pd.isna(v) for v in [ema5, ema9, ema21s]):
            if close > ema5 > ema9 > ema21s:
                intra_score, intra_sig = 100, "STRONG_BULL"
            elif close > ema9 > ema21s:
                intra_score, intra_sig = 75, "BULL"
            elif close < ema5 < ema9 < ema21s:
                intra_score, intra_sig = 0, "STRONG_BEAR"
            elif close < ema9 < ema21s:
                intra_score, intra_sig = 25, "BEAR"

        # Swing
        swing_score = 50
        bull_pairs = 0
        ema8 = s.get("ema_m_8")
        ema13 = s.get("ema_m_13")
        ema21 = s.get("ema_m_21")
        ema34 = s.get("ema_m_34")
        ema55 = s.get("ema_m_55")
        swing_sig = "NEUTRAL"
        if not any(v is None or pd.isna(v) for v in
                   [ema8, ema13, ema21, ema34, ema55]):
            pairs = [(ema8, ema13), (ema13, ema21),
                     (ema21, ema34), (ema34, ema55)]
            bull_pairs = sum(1 for a, b in pairs if a > b)
            swing_score = {4: 100, 3: 80, 2: 50, 1: 20, 0: 0}.get(bull_pairs, 50)
            swing_sig = {4: "FULL_BULL", 3: "BULL", 2: "WEAK_BULL",
                         1: "WEAK_BEAR", 0: "FULL_BEAR"}.get(bull_pairs, "NEUTRAL")

        # Position
        pos_score = 50
        ema50 = s.get("ema_l_50")
        ema100 = s.get("ema_l_100")
        ema200 = s.get("ema_l_200")
        pos_sig = "NEUTRAL"
        if not any(v is None or pd.isna(v) for v in [ema50, ema100, ema200]):
            if close > ema50 > ema100 > ema200:
                pos_score, pos_sig = 100, "LONG_BULL"
            elif close > ema50 > ema200:
                pos_score, pos_sig = 75, "BULL"
            elif close < ema50 < ema100 < ema200:
                pos_score, pos_sig = 0, "LONG_BEAR"
            else:
                pos_score, pos_sig = 40, "MIXED"

        # Tổng hợp
        final_raw = intra_score * 0.30 + swing_score * 0.50 + pos_score * 0.20
        final_score = final_raw / 100 * 25

        return round(final_score, 2), {
            "status": swing_sig,
            "bull_pairs": bull_pairs,
            "intraday": {"signal": intra_sig, "raw": intra_score},
            "swing": {
                "signal": swing_sig, "raw": swing_score,
                "ema8": round(ema8, 2) if ema8 else None,
                "ema13": round(ema13, 2) if ema13 else None,
                "ema21": round(ema21, 2) if ema21 else None,
                "ema34": round(ema34, 2) if ema34 else None,
                "ema55": round(ema55, 2) if ema55 else None,
            },
            "position": {"signal": pos_sig, "raw": pos_score},
        }

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
        detail = {"status": "ok"}

        score = 12 if macd > sig else 3
        detail["above_signal"] = macd > sig

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

        if macd > sig and prev_hist is not None and prev_hist < 0 and hist > 0:
            score = max(score, 20)
            detail["fresh_cross"] = "bullish"
        elif macd < sig and prev_hist is not None and prev_hist > 0 and hist < 0:
            score = 0
            detail["fresh_cross"] = "bearish"

        detail["values"] = {
            "macd": round(macd, 3),
            "signal": round(sig, 3),
            "hist": round(hist, 3),
        }
        return min(20, score), detail

    # ============================================================
    # 6. CHẤM ĐIỂM RSI (max 15) — Threshold cố định từ config tối ưu
    # ============================================================
    def _score_rsi(self, s):
        rsi = s.get("rsi")
        if rsi is None or pd.isna(rsi):
            return 0, {"status": "no_data"}

        # Threshold cố định (config tối ưu)
        over_sold_mild = 40
        over_bought_mild = 60
        over_sold_strong = 30
        over_bought_strong = 70

        detail = {
            "value": round(rsi, 1),
            "thresholds": {
                "oversold": over_sold_mild,
                "overbought": over_bought_mild,
            },
        }

        if over_sold_mild <= rsi <= over_bought_mild:
            score = 15
            detail["status"] = "ideal"
            detail["note"] = f"Vùng lý tưởng ({over_sold_mild}-{over_bought_mild})"
        elif over_sold_strong <= rsi < over_sold_mild:
            score = 13
            detail["status"] = "oversold_mild"
            detail["note"] = f"Quá bán nhẹ"
        elif rsi < over_sold_strong:
            score = 10
            detail["status"] = "oversold_strong"
            detail["note"] = f"Quá bán mạnh"
        elif over_bought_mild < rsi <= over_bought_strong:
            score = 8
            detail["status"] = "overbought_mild"
            detail["note"] = f"Tăng nóng"
        else:
            score = 2
            detail["status"] = "overbought_strong"
            detail["note"] = f"Quá mua mạnh"

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
            score, detail["status"] = 15, "surge"
        elif ratio >= 1.5:
            score, detail["status"] = 13, "high"
        elif ratio >= 1.0:
            score, detail["status"] = 10, "normal"
        elif ratio >= 0.7:
            score, detail["status"] = 6, "low"
        else:
            score, detail["status"] = 3, "very_low"

        return score, detail

    # ============================================================
    # 8. CHẤM ĐIỂM BOLLINGER (max 15)
    # ============================================================
    def _score_bollinger(self, s):
        close = s.get("close", 0)
        upper = s.get("bb_upper")
        lower = s.get("bb_lower")
        width = s.get("bb_width")

        if any(v is None or pd.isna(v) for v in [upper, lower]):
            return 0, {"status": "no_data"}

        rng = upper - lower
        if rng <= 0:
            return 5, {"status": "no_range"}

        pos = (close - lower) / rng
        detail = {
            "position": round(pos, 2),
            "width": round(width, 2) if width else None,
            "status": "ok",
        }

        if pos < 0.2:
            score, detail["status"] = 15, "near_lower"
        elif pos < 0.4:
            score, detail["status"] = 12, "lower_half"
        elif pos < 0.6:
            score, detail["status"] = 10, "middle"
        elif pos < 0.8:
            score, detail["status"] = 6, "upper_half"
        else:
            score, detail["status"] = 3, "near_upper"

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
            score, detail["status"] = 10, "optimal"
        elif 3.0 < atr_pct <= 5.0:
            score, detail["status"] = 6, "high"
        elif atr_pct < 1.0:
            score, detail["status"] = 4, "low"
        else:
            score, detail["status"] = 2, "extreme"

        return score, detail

    # ============================================================
    # 10. TỔNG ĐIỂM KỸ THUẬT
    # ============================================================
    def technical_score(self):
        s = self.get_current_signals()
        if not s:
            return 0, {"error": "Không đủ dữ liệu"}

        W = self.weights
        total_w = sum(W.values())

        ema_raw, ema_detail = self._score_ema_ribbon(s)
        macd_raw, macd_detail = self._score_macd(s)
        rsi_raw, rsi_detail = self._score_rsi(s)
        vol_raw, vol_detail = self._score_volume(s)
        bb_raw, bb_detail = self._score_bollinger(s)
        atr_raw, atr_detail = self._score_atr(s)

        # Scale từng chỉ báo về [0,100]
        ema_pct = ema_raw / 25 * 100
        macd_pct = macd_raw / 20 * 100
        rsi_pct = rsi_raw / 15 * 100
        vol_pct = vol_raw / 15 * 100
        bb_pct = bb_raw / 15 * 100
        atr_pct_scaled = atr_raw / 10 * 100

        # Weighted sum
        total = (
            ema_pct * W["ema_ribbon"]
            + macd_pct * W["macd"]
            + rsi_pct * W["rsi"]
            + vol_pct * W["volume"]
            + bb_pct * W["bollinger"]
            + atr_pct_scaled * W["atr"]
        ) / total_w

        total = round(total, 1)

        # Confidence
        bull = 0
        if ema_pct >= 70: bull += 1
        if macd_pct >= 70: bull += 1
        if rsi_pct >= 80: bull += 1
        if vol_pct >= 80: bull += 1
        if bb_pct >= 70: bull += 1
        if atr_pct_scaled >= 60: bull += 1

        bear = 0
        if ema_pct <= 25: bear += 1
        if macd_pct <= 25: bear += 1
        if rsi_pct <= 30: bear += 1
        if bb_pct <= 30: bear += 1

        if bull >= 5: confidence = "HIGH"
        elif bull >= 3: confidence = "MEDIUM"
        elif bear >= 3: confidence = "LOW"
        else: confidence = "LOW"

        # Entry decision
        signal, action, reason = self._entry_decision(
            total, confidence, bull, bear, s
        )

        # Risk Management — dùng config tối ưu
        tc = getattr(config, "TRADING_CONFIG", {})
        sl_mult = tc.get("sl_mult", 3.0)
        tp_mult = tc.get("tp_mult", 5.0)

        close = s.get("close", 0)
        atr = s.get("atr", 0)
        stop_loss = round(close - sl_mult * atr, 2) if atr else None
        tp1 = round(close + tp_mult * atr, 2) if atr else None
        tp2 = round(close + (tp_mult * 1.5) * atr, 2) if atr else None
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
            "bullish_count": bull,
            "bearish_count": bear,
            "entry_action": action,
            "entry_reason": reason,
            "risk_management": {
                "close": close,
                "atr": round(atr, 2) if atr else None,
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
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
        tc = getattr(config, "TRADING_CONFIG", {})
        entry_thr = tc.get("entry_threshold", 55)
        exit_thr = tc.get("exit_threshold", 30)

        # ===== MUA MẠNH =====
        if score >= 75 and confidence == "HIGH":
            return ("MUA MẠNH", "ENTER_NOW",
                    f"Điểm {score} + confidence cao. Vào ngay 70% vị thế.")

        # ===== MUA =====
        if score >= entry_thr and confidence in ("HIGH", "MEDIUM"):
            return ("MUA", "ENTER_PARTIAL",
                    f"Điểm {score} >= {entry_thr}. Vào 40-50% vị thế.")

        # ===== MUA THĂM DÒ =====
        if score >= entry_thr - 5 and bull >= 3:
            return ("MUA THĂM DÒ", "ENTER_SMALL",
                    f"Điểm {score}. Vào 20-30% thăm dò.")

        # ===== CHỜ =====
        if exit_thr <= score < entry_thr:
            return ("CHỜ", "WAIT",
                    f"Điểm {score} - tín hiệu chưa rõ.")

        # ===== KHÔNG VÀO / BÁN =====
        if score < exit_thr or bear >= 3:
            return ("BÁN", "EXIT_OR_SHORT",
                    f"Điểm {score} - tín hiệu tiêu cực. Thoát hàng.")

        return ("TRUNG TÍNH", "NEUTRAL", "Tín hiệu hỗn hợp.")
