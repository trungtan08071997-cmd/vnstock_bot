"""
Optimization v3 - Tối ưu theo TOTAL RETURN
Sử dụng multi-filter strategy từ backtest_engine_v2
- Market filter
- Fundamental filter (hardcoded VN30)
- Trend filter
- Trailing stop option
- Signal exit option
"""
import os
import json
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import config
from data_fetcher import get_stock_history, get_financial_ratios
from backtest_engine_v2 import compute_all_indicators
from fundamental_filter import passes_fundamental_filter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CONFIG
# ============================================================
POSITION_SIZE = 0.10
FEE_RATE = 0.0015
MIN_TRADES = 20
MAX_TRADES = 100


# ============================================================
# CACHE DATA
# ============================================================
_DATA_CACHE = {}


def load_and_cache_data(days=400):
    """Tải data 1 lần."""
    global _DATA_CACHE
    if _DATA_CACHE:
        return _DATA_CACHE

    print(f"\n📥 Đang tải dữ liệu VN30 ({days} phiên)...")
    data = {}
    ratios_dict = {}

    def fetch(sym):
        try:
            df = get_stock_history(sym, days)
            ratios = get_financial_ratios(sym)
            if df is None or len(df) < 150:
                return sym, None, None
            return sym, compute_all_indicators(df), ratios
        except Exception:
            return sym, None, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch, s): s for s in config.VN30_LIST}
        for f in as_completed(futures):
            sym, df, ratios = f.result()
            if df is not None:
                data[sym] = df
                ratios_dict[sym] = ratios
                print(f"  ✓ {sym}: {len(df)} phiên")

    print(f"📊 Đã tải {len(data)}/{len(config.VN30_LIST)} mã")
    _DATA_CACHE = {"data": data, "ratios": ratios_dict}
    return _DATA_CACHE


# ============================================================
# TECHNICAL SCORE (từ chỉ báo đã tính)
# ============================================================
def compute_technical_score(df, i):
    """Chấm điểm kỹ thuật tại phiên i (0-100)."""
    row = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else row
    close = row["close"]

    # EMA (25 điểm)
    ema8, ema13, ema21, ema34, ema55 = (
        row.get("ema_m_8"), row.get("ema_m_13"), row.get("ema_m_21"),
        row.get("ema_m_34"), row.get("ema_m_55")
    )
    ema_score = 12.5  # Trung bình
    if not any(pd.isna(v) for v in [ema8, ema13, ema21, ema34, ema55]):
        pairs = [(ema8, ema13), (ema13, ema21), (ema21, ema34), (ema34, ema55)]
        bull = sum(1 for a, b in pairs if a > b)
        ema_score = {4: 25, 3: 20, 2: 12.5, 1: 5, 0: 0}.get(bull, 12.5)

    # MACD (20 điểm)
    macd, sig, hist = row.get("macd"), row.get("macd_signal"), row.get("macd_hist")
    prev_hist = prev.get("macd_hist")
    macd_score = 0
    if not any(pd.isna(v) for v in [macd, sig, hist]):
        macd_score = 12 if macd > sig else 3
        if hist > 0: macd_score += 5
        if not pd.isna(prev_hist) and hist > prev_hist: macd_score += 3
    macd_score = min(20, macd_score)

    # RSI (15 điểm)
    rsi = row.get("rsi_14")
    rsi_score = 0
    if not pd.isna(rsi):
        if 40 <= rsi <= 60: rsi_score = 15
        elif 30 <= rsi < 40: rsi_score = 13
        elif rsi < 30: rsi_score = 10
        elif 60 < rsi <= 70: rsi_score = 8
        else: rsi_score = 2

    # Volume (15 điểm)
    vr = row.get("volume_ratio")
    vol_score = 0
    if not pd.isna(vr):
        if vr >= 2.0: vol_score = 15
        elif vr >= 1.5: vol_score = 13
        elif vr >= 1.0: vol_score = 10
        elif vr >= 0.7: vol_score = 6
        else: vol_score = 3

    # Bollinger (15 điểm)
    upper, lower = row.get("bb_upper"), row.get("bb_lower")
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

    # ATR (10 điểm)
    atr_pct = row.get("atr_pct")
    atr_score = 0
    if not pd.isna(atr_pct):
        if 1.0 <= atr_pct <= 3.0: atr_score = 10
        elif 3.0 < atr_pct <= 5.0: atr_score = 6
        elif atr_pct < 1.0: atr_score = 4
        else: atr_score = 2

    return round(ema_score + macd_score + rsi_score + vol_score + bb_score + atr_score, 1)


# ============================================================
# SIMULATE VỚI PARAMS MỚI
# ============================================================
def simulate_symbol_v3(symbol, df, ratios, params, start_idx=150):
    """
    Simulate với:
    - Trailing stop option
    - Signal exit option
    - Fundamental filter
    - Trend filter
    """
    if df is None or len(df) < start_idx + 20:
        return []

    # Fundamental filter
    if not passes_fundamental_filter(ratios)[0]:
        return []

    # Params
    entry_thr = params["entry_threshold"]
    exit_thr = params.get("exit_threshold", 40)
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]
    max_hold = params["max_hold"]
    use_signal_exit = params.get("use_signal_exit", False)
    use_trailing = params.get("use_trailing_stop", False)

    trades = []
    in_pos = False
    entry_price = 0
    entry_idx = 0
    entry_score = 0
    stop_loss = 0
    take_profit = 0
    highest_price = 0

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        close = row["close"]

        # ===== ĐANG GIỮ LỆNH =====
        if in_pos:
            days = i - entry_idx
            exit_reason = None
            exit_price = close

            # Cập nhật trailing stop
            if use_trailing:
                if close > highest_price:
                    highest_price = close
                    # Kéo SL lên: giữ khoảng cách = sl_mult × ATR
                    atr = row.get("atr_14") or (entry_price * 0.03)
                    new_sl = highest_price - sl_mult * atr
                    if new_sl > stop_loss:
                        stop_loss = new_sl

            # Check exit conditions
            if close <= stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = stop_loss
            elif close >= take_profit:
                exit_reason = "TAKE_PROFIT"
                exit_price = take_profit
            elif days >= max_hold:
                exit_reason = "MAX_HOLD"
            elif use_signal_exit:
                score = compute_technical_score(df, i)
                if score < exit_thr:
                    exit_reason = "SIGNAL_REVERSAL"

            if exit_reason:
                gross = (exit_price - entry_price) / entry_price
                net = gross - 2 * FEE_RATE
                trades.append({
                    "entry_date": df.iloc[entry_idx]["time"],
                    "exit_date": row["time"],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "days_held": days,
                    "return_pct": round(net * 100, 2),
                    "exit_reason": exit_reason,
                    "signal_score": entry_score,
                })
                in_pos = False

        # ===== KHÔNG GIỮ LỆNH =====
        else:
            # Trend filter
            ema200 = row.get("ema_l_200")
            if pd.isna(ema200) or close <= ema200:
                continue

            # Technical score
            score = compute_technical_score(df, i)

            if score >= entry_thr:
                entry_price = next_row["open"]
                entry_idx = i + 1
                entry_score = score
                in_pos = True
                highest_price = entry_price

                atr = row.get("atr_14") or (entry_price * 0.03)
                stop_loss = entry_price - sl_mult * atr
                take_profit = entry_price + tp_mult * atr

    # Đóng cuối
    if in_pos:
        final = df.iloc[-1]
        gross = (final["close"] - entry_price) / entry_price
        net = gross - 2 * FEE_RATE
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
    """Metrics với position size cố định."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0,
            "total_return": 0, "sharpe": -99, "max_drawdown": 0,
            "profit_factor": 0, "calmar": 0,
            "avg_win": 0, "avg_loss": 0, "avg_hold": 0,
        }

    returns = np.array([t["return_pct"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    n = len(returns)
    win_rate = len(wins) / n * 100
    avg_return = float(np.mean(returns))

    per_trade_pnl = returns * POSITION_SIZE
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
        "avg_win": round(float(np.mean(wins)), 2) if len(wins) else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if len(losses) else 0,
        "avg_hold": round(float(np.mean([t["days_held"] for t in trades])), 1),
    }


def backtest_all(data, ratios_dict, params):
    """Chạy backtest toàn universe."""
    all_trades = []
    for sym, df in data.items():
        ratios = ratios_dict.get(sym)
        trades = simulate_symbol_v3(sym, df, ratios, params)
        all_trades.extend(trades)
    return all_trades


# ============================================================
# OBJECTIVE v3 - TỐI ƯU TOTAL RETURN
# ============================================================
def create_objective(data, ratios_dict):
    """Objective v3: tập trung vào total_return."""

    def objective(trial):
        # Đề xuất params
        params = {
            "entry_threshold": trial.suggest_int("entry_threshold", 60, 80, step=5),
            "exit_threshold": trial.suggest_int("exit_threshold", 30, 50, step=5),
            "sl_mult": trial.suggest_float("sl_mult", 2.0, 4.0, step=0.25),
            "tp_mult": trial.suggest_float("tp_mult", 2.0, 4.0, step=0.25),
            "max_hold": trial.suggest_int("max_hold", 10, 30, step=5),
            "use_signal_exit": trial.suggest_categorical("use_signal_exit", [True, False]),
            "use_trailing_stop": trial.suggest_categorical("use_trailing_stop", [True, False]),
        }

        # R:R check
        rr = params["tp_mult"] / params["sl_mult"]
        if rr < 1.0:
            return -99

        # Backtest
        trades = backtest_all(data, ratios_dict, params)
        metrics = compute_metrics(trades)

        # Constraint
        if metrics["n_trades"] < MIN_TRADES:
            return -99
        if metrics["n_trades"] > MAX_TRADES:
            return -99

        # ===== OBJECTIVE MỚI =====
        score = (
            metrics["total_return"] * 0.40
            + metrics["sharpe"] * 15 * 0.30
            + (metrics["win_rate"] - 45) * 0.20
            - metrics["max_drawdown"] * 0.10
        )

        trial.set_user_attr("metrics", metrics)
        return score

    return objective


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   OPTIMIZATION v3 — TOTAL RETURN FOCUS              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Trials:   {args.trials}")
    print(f"  Days:     {args.days}")
    print(f"  Timeout:  {args.timeout}s")

    # Load
    cache = load_and_cache_data(days=args.days)
    data = cache["data"]
    ratios_dict = cache["ratios"]

    if len(data) < 5:
        print("❌ Không đủ dữ liệu")
        return

    # Study
    sampler = TPESampler(n_startup_trials=20, seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="vn30_optimization_v3",
    )

    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == args.trials - 1:
            completed = [t for t in study.trials
                         if t.value is not None and t.value > -99]
            if not completed:
                return
            best = max(completed, key=lambda t: t.value)
            m = best.user_attrs.get("metrics", {})
            print(
                f"  [{trial.number:3d}/{args.trials}] "
                f"Best: {best.value:.3f} | "
                f"Return: {m.get('total_return', 0):+.1f}% | "
                f"Sharpe: {m.get('sharpe', 0):.2f} | "
                f"WR: {m.get('win_rate', 0):.1f}% | "
                f"DD: {m.get('max_drawdown', 0):.1f}% | "
                f"Trades: {m.get('n_trades', 0)}"
            )

    print("\n🔍 Bắt đầu tìm kiếm...\n")
    objective = create_objective(data, ratios_dict)

    try:
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                       callbacks=[callback], show_progress_bar=False)
    except KeyboardInterrupt:
        print("\n⚠️ Dừng bởi người dùng")

    # Báo cáo
    completed = [t for t in study.trials
                 if t.value is not None and t.value > -99]

    if not completed:
        print("\n❌ Không có trial hợp lệ")
        return

    best = max(completed, key=lambda t: t.value)
    print("\n" + "=" * 72)
    print("🏆 KẾT QUẢ OPTIMIZATION v3")
    print("=" * 72)
    print(f"\n📌 Best Trial #{best.number} — score {best.value:.4f}")

    print(f"\n📊 Best Parameters:")
    for k, v in best.params.items():
        print(f"   {k:25s} = {v}")

    print(f"\n📈 Best Metrics:")
    m = best.user_attrs.get("metrics", {})
    for k, v in m.items():
        print(f"   {k:25s} = {v}")

    # Top 5
    print(f"\n🥇 TOP 5 CONFIGS:")
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(sorted_trials, 1):
        mm = t.user_attrs.get("metrics", {})
        print(f"\n   #{i} — Trial {t.number} (score {t.value:.3f})")
        print(f"       Entry {t.params['entry_threshold']} | "
              f"SL {t.params['sl_mult']} | TP {t.params['tp_mult']} | "
              f"Hold {t.params['max_hold']}d")
        print(f"       SignalExit={t.params['use_signal_exit']} | "
              f"Trailing={t.params['use_trailing_stop']}")
        print(f"       Return: {mm.get('total_return', 0):+.1f}% | "
              f"Sharpe: {mm.get('sharpe', 0):.2f} | "
              f"WR: {mm.get('win_rate', 0):.1f}% | "
              f"DD: {mm.get('max_drawdown', 0):.1f}% | "
              f"Trades: {mm.get('n_trades', 0)}")

    # Lưu
    os.makedirs("output", exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "version": "v3",
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "best_trial": {
            "number": best.number,
            "objective_value": best.value,
            "params": best.params,
            "metrics": m,
        },
        "top_5_trials": [
            {
                "number": t.number,
                "objective_value": t.value,
                "params": t.params,
                "metrics": t.user_attrs.get("metrics", {}),
            }
            for t in sorted_trials
        ],
    }
    with open("output/optimization_v3_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Đã lưu: output/optimization_v3_result.json")

    # Config để copy
    print("\n" + "=" * 72)
    print("📋 CONFIG ĐỂ COPY VÀO config.py")
    print("=" * 72)
    print(f"""
TRADING_CONFIG = {{
    "entry_threshold": {best.params['entry_threshold']},
    "exit_threshold": {best.params.get('exit_threshold', 40)},
    "sl_mult": {best.params['sl_mult']},
    "tp_mult": {best.params['tp_mult']},
    "max_hold_days": {best.params['max_hold']},
    "use_trend_filter": True,
    "use_signal_exit": {best.params['use_signal_exit']},
    "use_trailing_stop": {best.params['use_trailing_stop']},
    "use_dynamic_weights": False,
    "signal_thresholds": {{
        "strong_buy": 75,
        "buy": {best.params['entry_threshold']},
        "hold": {best.params.get('exit_threshold', 40) + 5},
        "sell": {best.params.get('exit_threshold', 40)},
    }},
}}
""")


if __name__ == "__main__":
    main()
