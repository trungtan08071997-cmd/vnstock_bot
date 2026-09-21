"""
Backtest Engine v2 - Kết hợp Market + Fundamental + Technical + Sentiment
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from data_fetcher import get_stock_history, get_financial_ratios, get_vnindex_history
from fundamental_filter import passes_fundamental_filter, score_fundamental_quality
from sector_analyzer import get_sector


# ============================================================
# TÍNH CHỈ BÁO
# ============================================================
def compute_all_indicators(df):
    """Tính toàn bộ chỉ báo."""
    d = df.copy()

    for p in [5, 9, 21]:
        d[f"ema_s_{p}"] = d["close"].ewm(span=p, adjust=False).mean()
    for p in [8, 13, 21, 34, 55]:
        d[f"ema_m_{p}"] = d["close"].ewm(span=p, adjust=False).mean()
    for p in [50, 100, 200]:
        d[f"ema_l_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    d["bb_middle"] = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * std
    d["bb_lower"] = d["bb_middle"] - 2 * std

    high_low = d["high"] - d["low"]
    high_close = (d["high"] - d["close"].shift()).abs()
    low_close = (d["low"] - d["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["atr_pct"] = d["atr_14"] / d["close"] * 100

    d["volume_sma_20"] = d["volume"].rolling(20).mean()
    d["volume_ratio"] = d["volume"] / d["volume_sma_20"]

    return d


# ============================================================
# TÍNH ĐIỂM TECHNICAL (từ chỉ báo đã tính)
# ============================================================
def compute_technical_score(df, i):
    """Chấm điểm kỹ thuật tại phiên i (0-100)."""
    row = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else row
    close = row["close"]

    # ===== EMA (max 25) =====
    ema8 = row.get("ema_m_8")
    ema13 = row.get("ema_m_13")
    ema21 = row.get("ema_m_21")
    ema34 = row.get("ema_m_34")
    ema55 = row.get("ema_m_55")

    swing_score = 50
    if not any(pd.isna(v) for v in [ema8, ema13, ema21, ema34, ema55]):
        pairs = [(ema8, ema13), (ema13, ema21), (ema21, ema34), (ema34, ema55)]
        bull = sum(1 for a, b in pairs if a > b)
        swing_score = {4: 100, 3: 80, 2: 50, 1: 20, 0: 0}.get(bull, 50)

    ema_score = swing_score / 100 * 25

    # ===== MACD (max 20) =====
    macd = row.get("macd")
    sig = row.get("macd_signal")
    hist = row.get("macd_hist")
    prev_hist = prev.get("macd_hist")
    macd_score = 0
    if not any(pd.isna(v) for v in [macd, sig, hist]):
        macd_score = 12 if macd > sig else 3
        if hist > 0: macd_score += 5
        if not pd.isna(prev_hist) and hist > prev_hist: macd_score += 3
    macd_score = min(20, macd_score)

    # ===== RSI (max 15) =====
    rsi = row.get("rsi_14")
    rsi_score = 0
    if not pd.isna(rsi):
        if 40 <= rsi <= 60: rsi_score = 15
        elif 30 <= rsi < 40: rsi_score = 13
        elif rsi < 30: rsi_score = 10
        elif 60 < rsi <= 70: rsi_score = 8
        else: rsi_score = 2

    # ===== Volume (max 15) =====
    vr = row.get("volume_ratio")
    vol_score = 0
    if not pd.isna(vr):
        if vr >= 2.0: vol_score = 15
        elif vr >= 1.5: vol_score = 13
        elif vr >= 1.0: vol_score = 10
        elif vr >= 0.7: vol_score = 6
        else: vol_score = 3

    # ===== Bollinger (max 15) =====
    upper = row.get("bb_upper")
    lower = row.get("bb_lower")
    bb_score = 5
    if not any(pd.isna(v) for v in [upper, lower]):
        rng = upper - lower
        if rng > 0:
            pos = (close - lower) / rng
            if pos < 0.2: bb_score = 15
            elif pos < 0.4: bb_score = 12
            elif pos < 0.6: bb_score = 10
            elif pos < 0.8: bb_score = 6
            else: bb_score = 3

    # ===== ATR (max 10) =====
    atr_pct = row.get("atr_pct")
    atr_score = 0
    if not pd.isna(atr_pct):
        if 1.0 <= atr_pct <= 3.0: atr_score = 10
        elif 3.0 < atr_pct <= 5.0: atr_score = 6
        elif atr_pct < 1.0: atr_score = 4
        else: atr_score = 2

    total = ema_score + macd_score + rsi_score + vol_score + bb_score + atr_score
    return round(total, 1)


# ============================================================
# SIMULATE với MULTI-FILTER
# ============================================================
def simulate_symbol_v2(symbol, df, ratios, vnindex_df,
                        params, start_idx=150):
    """
    Mô phỏng giao dịch với:
    - Market filter (VN-Index)
    - Fundamental filter
    - Trend filter (giá > EMA200)
    - Technical entry timing
    """
    if df is None or len(df) < start_idx + 20:
        return []

    # ===== FUNDAMENTAL FILTER (1 lần) =====
    if not passes_fundamental_filter(ratios)[0]:
        return []  # Loại hẳn mã này

    # ===== CHUẨN BỊ VN-INDEX =====
    vni_close = None
    vni_ma50 = None
    if vnindex_df is not None and len(vnindex_df) >= 50:
        vni_close = vnindex_df["close"].values
        vni_ma50_series = pd.Series(vni_close).rolling(50).mean().values

    # ===== PARAMS =====
    entry_thr = params["entry_threshold"]
    exit_thr = params.get("exit_threshold", 40)
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]
    max_hold = params["max_hold"]
    fee = 0.0015

    # ===== TRADING LOOP =====
    trades = []
    in_pos = False
    entry_price = 0
    entry_idx = 0
    entry_score = 0
    stop_loss = 0
    take_profit = 0

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        close = row["close"]

        # ===== ĐANG GIỮ LỆNH =====
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
                # Check technical exit
                score = compute_technical_score(df, i)
                if score < exit_thr:
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

        # ===== KHÔNG GIỮ LỆNH =====
        else:
            # ===== MARKET FILTER (nếu có VN-Index) =====
            if vni_close is not None and vni_ma50 is not None:
                if i < len(vni_close) and i < len(vni_ma50):
                    vni_cur = vni_close[i]
                    vni_ma = vni_ma50[i]
                    if not pd.isna(vni_ma) and vni_cur < vni_ma:
                        continue
            # Nếu không có VN-Index → bỏ qua filter này

            # ===== TREND FILTER =====
            ema200 = row.get("ema_l_200")
            if pd.isna(ema200) or close <= ema200:
                continue

            # ===== TECHNICAL SCORE =====
            score = compute_technical_score(df, i)

            if score >= entry_thr:
                entry_price = next_row["open"]
                entry_idx = i + 1
                entry_score = score
                in_pos = True

                atr = row.get("atr_14") or (entry_price * 0.03)
                stop_loss = entry_price - sl_mult * atr
                take_profit = entry_price + tp_mult * atr

    # Đóng cuối
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
def compute_metrics_v2(trades, position_size=0.10):
    """Metrics với position size cố định."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0,
            "total_return": 0, "sharpe": -99, "max_drawdown": 0,
            "profit_factor": 0, "calmar": 0,
        }

    returns = np.array([t["return_pct"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    n = len(returns)
    win_rate = len(wins) / n * 100
    avg_return = float(np.mean(returns))

    per_trade_pnl = returns * position_size
    total_return = float(np.sum(per_trade_pnl))

    std = float(np.std(returns))
    sharpe = (np.mean(returns) / std * np.sqrt(252 / 10)) if std > 0 else 0

    cumulative = np.cumsum(per_trade_pnl) + 100
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / running_max
    max_dd = float(abs(dd.min()) * 100) if len(dd) > 0 else 0

    sw = float(np.sum(wins)) if len(wins) else 0
    sl = float(abs(np.sum(losses))) if len(losses) else 0
    pf = sw / sl if sl > 0 else 0

    calmar = (total_return / max_dd) if max_dd > 0 else 0

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 3),
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(pf, 3),
        "calmar": round(calmar, 3),
    }


# ============================================================
# LOAD DATA
# ============================================================
def load_data_v2(days=400):
    """Tải data + ratios + VN-Index."""
    print(f"\n📥 Đang tải dữ liệu VN30 ({days} phiên)...")

    data = {}
    ratios_dict = {}

    def fetch(sym):
        try:
            df = get_stock_history(sym, days)
            ratios = get_financial_ratios(sym)
            if df is None or len(df) < 200:
                return sym, None, None
            return sym, compute_all_indicators(df), ratios
        except Exception:
            return sym, None, None

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch, s): s for s in config.VN30_LIST}
        for f in as_completed(futures):
            sym, df, ratios = f.result()
            if df is not None:
                data[sym] = df
                ratios_dict[sym] = ratios
                print(f"  ✓ {sym}: {len(df)} phiên")
            else:
                print(f"  ✗ {sym}")

    # VN-Index
    print(f"\n📊 Đang tải VN-Index...")
    vni = get_vnindex_history(days)
    if vni is not None:
        vni = compute_all_indicators(vni)
        print(f"  ✓ VN-Index: {len(vni)} phiên")
    else:
        print(f"  ✗ Không lấy được VN-Index")

    return data, ratios_dict, vni


# ============================================================
# CHẠY BACKTEST
# ============================================================
def run_backtest_v2(data, ratios_dict, vnindex_df, params):
    """Chạy backtest toàn universe với multi-filter."""
    all_trades = []
    passed_symbols = []

    print(f"\n🔍 Filter fundamental:")
    for sym, ratios in ratios_dict.items():
        ok, reason = passes_fundamental_filter(ratios)
        if ok:
            passed_symbols.append(sym)
            print(f"  ✅ {sym}")
        else:
            print(f"  ❌ {sym}: {reason}")

    print(f"\n📊 {len(passed_symbols)}/{len(ratios_dict)} mã đạt tiêu chuẩn cơ bản")

    print(f"\n🚀 Chạy backtest {len(passed_symbols)} mã...")
    for sym in passed_symbols:
        df = data[sym]
        ratios = ratios_dict[sym]
        trades = simulate_symbol_v2(sym, df, ratios, vnindex_df, params)
        all_trades.extend(trades)
        print(f"  {sym}: {len(trades)} lệnh")

    return all_trades, passed_symbols


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("🎯 BACKTEST v2 — Technical + Fundamental + Market Filter")
    print("=" * 72)

    # Config test
    params = {
        "entry_threshold": 70,
        "exit_threshold": 40,
        "sl_mult": 2.5,
        "tp_mult": 5.0,
        "max_hold": 20,
    }

    print(f"\n⚙️  Params: {params}")

    # Load data
    data, ratios_dict, vni = load_data_v2(days=400)

    if len(data) < 3:
        print("❌ Không đủ dữ liệu")
        exit()

    # Backtest
    trades, passed = run_backtest_v2(data, ratios_dict, vni, params)

    # Metrics
    metrics = compute_metrics_v2(trades)

    print("\n" + "=" * 72)
    print("📊 KẾT QUẢ")
    print("=" * 72)
    print(f"  Mã đạt cơ bản:      {len(passed)}/{len(data)}")
    print(f"  Tổng lệnh:          {metrics['n_trades']}")
    print(f"  Win rate:           {metrics['win_rate']}%")
    print(f"  Lợi nhuận TB/lệnh:  {metrics['avg_return']}%")
    print(f"  Tổng lợi nhuận:     {metrics['total_return']}%")
    print(f"  Sharpe:             {metrics['sharpe']}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']}%")
    print(f"  Profit Factor:      {metrics['profit_factor']}")
    print(f"  Calmar:             {metrics['calmar']}")

    # Đánh giá
    print("\n🎯 ĐÁNH GIÁ:")
    if metrics["sharpe"] > 0.5 and metrics["win_rate"] > 50:
        print("  ✅ PASS — Chiến lược khả thi")
    elif metrics["sharpe"] > 0 and metrics["win_rate"] > 45:
        print("  ⚠️  TRUNG BÌNH — Cần tinh chỉnh")
    else:
        print("  ❌ FAIL — Cần đổi chiến lược")

    # Lưu
    os.makedirs("output", exist_ok=True)
    with open("output/backtest_v2_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "params": params,
            "passed_symbols": passed,
            "metrics": metrics,
            "trades": trades[:100],
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Đã lưu: output/backtest_v2_report.json")
