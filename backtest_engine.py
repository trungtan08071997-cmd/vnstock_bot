"""
Backtest Engine v2 - Tối ưu, chạy trong 2-5 phút cho 30 mã
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from data_fetcher import get_stock_history


# ============================================================
# WEIGHTS
# ============================================================
STATIC_WEIGHTS = {
    "ema_ribbon": 25, "macd": 20, "rsi": 15,
    "volume": 15, "bollinger": 15, "atr": 10,
}

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
# TÍNH CHỈ BÁO 1 LẦN DUY NHẤT
# ============================================================
def compute_all_indicators(df):
    """Tính toàn bộ chỉ báo 1 lần trên toàn bộ DataFrame."""
    d = df.copy()

    # EMA 3 khung
    for p in [5, 9, 21]:
        d[f"ema_s_{p}"] = d["close"].ewm(span=p, adjust=False).mean()
    for p in [8, 13, 21, 34, 55]:
        d[f"ema_m_{p}"] = d["close"].ewm(span=p, adjust=False).mean()
    for p in [50, 100, 200]:
        d[f"ema_l_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # RSI
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger
    d["bb_middle"] = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * std
    d["bb_lower"] = d["bb_middle"] - 2 * std

    # ATR
    high_low = d["high"] - d["low"]
    high_close = (d["high"] - d["close"].shift()).abs()
    low_close = (d["low"] - d["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["atr_pct"] = d["atr_14"] / d["close"] * 100

    # Volume
    d["volume_sma_20"] = d["volume"].rolling(20).mean()
    d["volume_ratio"] = d["volume"] / d["volume_sma_20"]

    # ADX
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

    return d


# ============================================================
# CHẤM ĐIỂM TỪ CHỈ BÁO ĐÃ TÍNH (siêu nhanh)
# ============================================================
def score_at_index(df, i, use_dynamic=True):
    """Chấm điểm tại phiên i, dùng chỉ báo đã tính sẵn."""
    row = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else row

    close = row["close"]

    # ----- Detect regime -----
    if use_dynamic:
        adx = row.get("adx_14", 20)
        atr_pct = row.get("atr_pct", 2.0)
        if pd.isna(adx): adx = 20
        if pd.isna(atr_pct): atr_pct = 2.0

        if adx >= 25:
            regime = "TREND_STRONG_VOLATILE" if atr_pct >= 3.0 else "TREND_STRONG_CALM"
        elif adx <= 20:
            regime = "SIDEWAY_VOLATILE" if atr_pct >= 3.0 else "SIDEWAY_CALM"
        else:
            regime = "TRANSITION"
        W = REGIME_WEIGHTS[regime]
    else:
        regime = "STATIC"
        W = STATIC_WEIGHTS

    total_w = sum(W.values())

    # ----- EMA Ribbon -----
    ema5 = row.get("ema_s_5")
    ema9 = row.get("ema_s_9")
    ema21s = row.get("ema_s_21")
    ema8 = row.get("ema_m_8")
    ema13 = row.get("ema_m_13")
    ema21 = row.get("ema_m_21")
    ema34 = row.get("ema_m_34")
    ema55 = row.get("ema_m_55")

    intra_score = 50
    if not any(pd.isna(v) for v in [ema5, ema9, ema21s]):
        if close > ema5 > ema9 > ema21s:
            intra_score = 100
        elif close > ema9 > ema21s:
            intra_score = 75
        elif close < ema5 < ema9 < ema21s:
            intra_score = 0
        elif close < ema9 < ema21s:
            intra_score = 25

    swing_score = 50
    if not any(pd.isna(v) for v in [ema8, ema13, ema21, ema34, ema55]):
        pairs = [(ema8, ema13), (ema13, ema21), (ema21, ema34), (ema34, ema55)]
        bull = sum(1 for a, b in pairs if a > b)
        swing_score = {4: 100, 3: 80, 2: 50, 1: 20, 0: 0}.get(bull, 50)

    pos_score = 50
    ema50 = row.get("ema_l_50")
    ema100 = row.get("ema_l_100")
    ema200 = row.get("ema_l_200")
    if not any(pd.isna(v) for v in [ema50, ema100, ema200]):
        if close > ema50 > ema100 > ema200:
            pos_score = 100
        elif close > ema50 > ema200:
            pos_score = 75
        elif close < ema50 < ema100 < ema200:
            pos_score = 0
        else:
            pos_score = 40

    ema_raw = (intra_score * 0.3 + swing_score * 0.5 + pos_score * 0.2) / 100 * 25

    # ----- MACD -----
    macd = row.get("macd")
    macd_sig = row.get("macd_signal")
    macd_hist = row.get("macd_hist")
    prev_hist = prev.get("macd_hist")
    macd_raw = 0
    if not any(pd.isna(v) for v in [macd, macd_sig, macd_hist]):
        if macd > macd_sig: macd_raw += 12
        else: macd_raw += 3
        if macd_hist > 0: macd_raw += 5
        if not pd.isna(prev_hist) and macd_hist > prev_hist:
            macd_raw += 3
        if (macd > macd_sig and not pd.isna(prev_hist)
                and prev_hist < 0 and macd_hist > 0):
            macd_raw = max(macd_raw, 20)
    macd_raw = min(20, macd_raw)

    # ----- RSI -----
    rsi = row.get("rsi_14")
    rsi_raw = 0
    if not pd.isna(rsi):
        if "TREND_STRONG" in regime:
            os_m, ob_m = 40, 75
        elif "SIDEWAY_CALM" in regime:
            os_m, ob_m = 35, 65
        elif "SIDEWAY_VOLATILE" in regime:
            os_m, ob_m = 30, 70
        else:
            os_m, ob_m = 40, 60
        if os_m <= rsi <= ob_m: rsi_raw = 15
        elif rsi < os_m: rsi_raw = 13
        elif rsi <= ob_m + 10: rsi_raw = 8
        else: rsi_raw = 2

    # ----- Volume -----
    vr = row.get("volume_ratio")
    vol_raw = 0
    if not pd.isna(vr):
        if vr >= 2.0: vol_raw = 15
        elif vr >= 1.5: vol_raw = 13
        elif vr >= 1.0: vol_raw = 10
        elif vr >= 0.7: vol_raw = 6
        else: vol_raw = 3

    # ----- Bollinger -----
    upper = row.get("bb_upper")
    lower = row.get("bb_lower")
    bb_raw = 5
    if not any(pd.isna(v) for v in [upper, lower]):
        rng = upper - lower
        if rng > 0:
            pos = (close - lower) / rng
            if pos < 0.2: bb_raw = 15
            elif pos < 0.4: bb_raw = 12
            elif pos < 0.6: bb_raw = 10
            elif pos < 0.8: bb_raw = 6
            else: bb_raw = 3

    # ----- ATR -----
    atr_pct = row.get("atr_pct")
    atr_raw = 0
    if not pd.isna(atr_pct):
        if 1.0 <= atr_pct <= 3.0: atr_raw = 10
        elif 3.0 < atr_pct <= 5.0: atr_raw = 6
        elif atr_pct < 1.0: atr_raw = 4
        else: atr_raw = 2

    # ----- Tổng -----
    score = (
        (ema_raw / 25 * 100) * W["ema_ribbon"]
        + (macd_raw / 20 * 100) * W["macd"]
        + (rsi_raw / 15 * 100) * W["rsi"]
        + (vol_raw / 15 * 100) * W["volume"]
        + (bb_raw / 15 * 100) * W["bollinger"]
        + (atr_raw / 10 * 100) * W["atr"]
    ) / total_w

    return round(score, 1), {
        "regime": regime,
        "atr": row.get("atr_14"),
        "close": close,
    }


# ============================================================
# MÔ PHỎNG GIAO DỊCH (siêu nhanh)
# ============================================================
def simulate_symbol(symbol, df, use_dynamic=True, start_idx=210,
                     max_hold=20, fee=0.0015):
    """Mô phỏng giao dịch cho 1 mã."""
    if df is None or len(df) < start_idx + 30:
        return []

    trades = []
    in_pos = False
    entry_price = 0
    entry_idx = 0
    entry_score = 0
    stop_loss = 0
    take_profit = 0

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        close = row["close"]

        if in_pos:
            days = i - entry_idx
            exit_reason = None
            if close <= stop_loss:
                exit_reason = "STOP_LOSS"
            elif close >= take_profit:
                exit_reason = "TAKE_PROFIT"
            elif days >= max_hold:
                exit_reason = "MAX_HOLD"
            else:
                score, _ = score_at_index(df, i, use_dynamic)
                if score < 40:
                    exit_reason = "SIGNAL_REVERSAL"

            if exit_reason:
                gross = (close - entry_price) / entry_price
                net = gross - 2 * fee
                trades.append({
                    "entry_date": df.iloc[entry_idx]["time"],
                    "exit_date": row["time"],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(close, 2),
                    "days_held": days,
                    "return_pct": round(net * 100, 2),
                    "exit_reason": exit_reason,
                    "signal_score": entry_score,
                })
                in_pos = False
        else:
            score, det = score_at_index(df, i, use_dynamic)
            if score >= 60:
                entry_price = next_row["open"]
                entry_idx = i + 1
                entry_score = score
                in_pos = True
                atr = det.get("atr") or (entry_price * 0.03)
                stop_loss = entry_price - 1.5 * atr
                take_profit = entry_price + 2.0 * atr

    # Đóng lệnh cuối
    if in_pos:
        final = df.iloc[-1]
        gross = (final["close"] - entry_price) / entry_price
        net = gross - 2 * fee
        trades.append({
            "entry_date": df.iloc[entry_idx]["time"],
            "exit_date": final["time"],
            "entry_price": round(entry_price, 2),
            "exit_price": round(final["close"], 2),
            "days_held": len(df) - 1 - entry_idx,
            "return_pct": round(net * 100, 2),
            "exit_reason": "END_OF_DATA",
            "signal_score": entry_score,
        })

    return trades


# ============================================================
# METRICS
# ============================================================
def compute_metrics(trades):
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "avg_return": 0,
                "total_return": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
                "avg_hold_days": 0}

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) * 100
    avg_return = np.mean(returns)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    total_return = (np.prod([1 + r/100 for r in returns]) - 1) * 100

    sharpe = ((np.mean(returns) / np.std(returns)) * np.sqrt(252/10)
              if np.std(returns) > 0 else 0)

    cumulative = np.cumprod([1 + r/100 for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / running_max
    max_dd = abs(dd.min()) * 100 if len(dd) > 0 else 0

    sw = sum(wins) if wins else 0
    sl = abs(sum(losses)) if losses else 0
    pf = sw / sl if sl > 0 else 0

    return {
        "n_trades": len(returns),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "total_return": round(total_return, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_hold_days": round(np.mean([t["days_held"] for t in trades]), 1),
    }


# ============================================================
# CHẠY BACKTEST
# ============================================================
def load_vn30_data(days=500):
    """Tải + tính chỉ báo 1 lần."""
    print(f"\n📥 Tải dữ liệu VN30 ({days} phiên)...")
    data = {}

    def fetch(sym):
        df = get_stock_history(sym, days)
        if df is None or len(df) < 250:
            return sym, None
        return sym, compute_all_indicators(df)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch, s): s for s in config.VN30_LIST}
        for f in as_completed(futures):
            sym, df = f.result()
            if df is not None:
                data[sym] = df
                print(f"  ✓ {sym}: {len(df)} phiên")
            else:
                print(f"  ✗ {sym}")

    return data


def run_backtest(name, use_dynamic, data):
    print(f"\n🚀 {name}")
    all_trades = []
    for i, (sym, df) in enumerate(data.items(), 1):
        trades = simulate_symbol(sym, df, use_dynamic=use_dynamic)
        all_trades.extend(trades)
        print(f"  [{i}/{len(data)}] {sym}: {len(trades)} lệnh")

    return {"name": name, "metrics": compute_metrics(all_trades),
            "trades": all_trades}


def print_comparison(static, dynamic):
    s, d = static["metrics"], dynamic["metrics"]
    print("\n" + "="*72)
    print("📊 SO SÁNH: STATIC vs DYNAMIC")
    print("="*72)
    print(f"{'Metric':<25} {'STATIC':>12} {'DYNAMIC':>12} {'Δ':>12}")
    print("-"*72)

    rows = [
        ("Số lệnh", "n_trades", "int", "neutral"),
        ("Win rate (%)", "win_rate", "f2", "high"),
        ("Lợi nhuận TB (%)", "avg_return", "f2", "high"),
        ("Tổng lợi nhuận (%)", "total_return", "f1", "high"),
        ("Sharpe Ratio", "sharpe", "f2", "high"),
        ("Max Drawdown (%)", "max_drawdown", "f1", "low"),
        ("Profit Factor", "profit_factor", "f2", "high"),
        ("Lãi TB (%)", "avg_win", "f2", "high"),
        ("Lỗ TB (%)", "avg_loss", "f2", "high"),
        ("Ngày giữ TB", "avg_hold_days", "f1", "neutral"),
    ]

    for label, key, fmt, direction in rows:
        sv, dv = s.get(key, 0), d.get(key, 0)
        delta = dv - sv

        if fmt == "int":
            print(f"{label:<25} {sv:>12} {dv:>12} {delta:>+12d}")
        elif fmt == "f1":
            marker = ""
            if direction == "high": marker = " ✅" if dv > sv else (" ❌" if dv < sv else "")
            if direction == "low": marker = " ✅" if dv < sv else (" ❌" if dv > sv else "")
            print(f"{label:<25} {sv:>12.1f} {dv:>12.1f} {delta:>+12.1f}{marker}")
        elif fmt == "f2":
            marker = ""
            if direction == "high": marker = " ✅" if dv > sv else (" ❌" if dv < sv else "")
            if direction == "low": marker = " ✅" if dv < sv else (" ❌" if dv > sv else "")
            print(f"{label:<25} {sv:>12.2f} {dv:>12.2f} {delta:>+12.2f}{marker}")

    print("="*72)

    # Kết luận
    dw = sum([
        1 if d["win_rate"] > s["win_rate"] else 0,
        1 if d["total_return"] > s["total_return"] else 0,
        1 if d["sharpe"] > s["sharpe"] else 0,
        1 if d["max_drawdown"] < s["max_drawdown"] else 0,
        1 if d["profit_factor"] > s["profit_factor"] else 0,
    ])
    sw = 5 - dw

    print(f"\n🎯 DYNAMIC thắng: {dw}/5 | STATIC thắng: {sw}/5")
    if dw >= 4:
        print("✅ KHUYẾN NGHỊ: DÙNG DYNAMIC WEIGHTS")
    elif sw >= 4:
        print("⚠️  KHUYẾN NGHỊ: DÙNG STATIC WEIGHTS")
    else:
        print("⚖️  Tương đương — cần thêm dữ liệu")


def save_report(static, dynamic, path="output/backtest_report.json"):
    os.makedirs("output", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "static": {"metrics": static["metrics"],
                       "sample_trades": static["trades"][:50]},
            "dynamic": {"metrics": dynamic["metrics"],
                        "sample_trades": dynamic["trades"][:50]},
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Đã lưu: {path}")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   BACKTEST: DYNAMIC vs STATIC (tối ưu)              ║")
    print("╚══════════════════════════════════════════════════════╝")

    data = load_vn30_data(days=500)
    if len(data) < 3:
        print("❌ Không đủ dữ liệu")
        return

    static_r = run_backtest("STATIC", False, data)
    dynamic_r = run_backtest("DYNAMIC", True, data)

    print_comparison(static_r, dynamic_r)
    save_report(static_r, dynamic_r)


if __name__ == "__main__":
    main()
