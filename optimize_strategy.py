"""
Bayesian Optimization v2 - Tối ưu theo Sharpe + WinRate + DD thấp
Fix các vấn đề của v1:
- Dùng position size cố định (không compound return)
- Objective tập trung Sharpe/WinRate/DD thay vì total return
- Constraint: trades 15-60, DD < 35%
- Hỗ trợ trend filter
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
from data_fetcher import get_stock_history
from backtest_engine import compute_all_indicators, score_at_index

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CONFIG CHO OPTIMIZATION
# ============================================================
POSITION_SIZE = 0.10          # 10% vốn mỗi lệnh
MIN_TRADES = 15               # Ít nhất 15 lệnh
MAX_TRADES = 60               # Nhiều nhất 60 lệnh
MAX_DD_ALLOWED = 35           # Max drawdown cho phép
FEE_RATE = 0.0015             # 0.15% mỗi chiều


# ============================================================
# CACHE DATA
# ============================================================
_DATA_CACHE = {}


def load_and_cache_data(days=500):
    """Tải + tính chỉ báo 1 lần cho tất cả VN30."""
    global _DATA_CACHE
    if _DATA_CACHE:
        return _DATA_CACHE

    print(f"\n📥 Đang tải dữ liệu VN30 ({days} phiên)...")
    data = {}

    def fetch(sym):
        try:
            df = get_stock_history(sym, days)
            if df is None or len(df) < 250:
                return sym, None
            return sym, compute_all_indicators(df)
        except Exception as e:
            return sym, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch, s): s for s in config.VN30_LIST}
        for f in as_completed(futures):
            sym, df = f.result()
            if df is not None:
                data[sym] = df
                print(f"  ✓ {sym}: {len(df)} phiên")
            else:
                print(f"  ✗ {sym}: fail")

    print(f"📊 Đã tải {len(data)}/{len(config.VN30_LIST)} mã")
    _DATA_CACHE = data
    return data


# ============================================================
# SIMULATE VỚI PARAMS
# ============================================================
def simulate_with_params(df, params, start_idx=210):
    """
    Mô phỏng giao dịch với params.
    
    params keys:
        entry_threshold, exit_threshold, sl_mult, tp_mult,
        max_hold, use_dynamic, use_trend_filter
    """
    if df is None or len(df) < start_idx + 30:
        return []

    entry_thr = params["entry_threshold"]
    exit_thr = params.get("exit_threshold", 40)
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]
    max_hold = params["max_hold"]
    use_dynamic = params["use_dynamic"]
    use_trend_filter = params.get("use_trend_filter", True)

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
                score, _ = score_at_index(df, i, use_dynamic)
                if score < exit_thr:
                    exit_reason = "SIGNAL_REVERSAL"

            if exit_reason:
                gross = (close - entry_price) / entry_price
                net = gross - 2 * FEE_RATE
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
            score, det = score_at_index(df, i, use_dynamic)

            if score >= entry_thr:
                # Trend filter: giá phải > EMA200
                if use_trend_filter:
                    ema200 = row.get("ema_l_200")
                    if pd.isna(ema200) or close <= ema200:
                        continue

                entry_price = next_row["open"]
                entry_idx = i + 1
                entry_score = score
                in_pos = True

                atr = det.get("atr") or (entry_price * 0.03)
                stop_loss = entry_price - sl_mult * atr
                take_profit = entry_price + tp_mult * atr

    # Đóng lệnh cuối
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
# METRICS v2 - FIXED POSITION SIZE (không compound)
# ============================================================
def compute_metrics(trades, position_size=POSITION_SIZE):
    """
    Tính metrics với POSITION SIZE cố định.
    - Total return = sum(return × position_size) — KHÔNG compound
    - Sharpe dùng return per trade (đã tính %)
    - Max DD tính từ cumulative P&L
    """
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0,
            "total_return": 0, "sharpe": -99, "max_drawdown": 100,
            "profit_factor": 0, "calmar": 0,
        }

    returns = np.array([t["return_pct"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    n = len(returns)
    win_rate = len(wins) / n * 100
    avg_return = float(np.mean(returns))

    # ===== TOTAL RETURN với position size cố định =====
    per_trade_pnl = returns * position_size
    total_return = float(np.sum(per_trade_pnl))

    # ===== SHARPE =====
    std = float(np.std(returns))
    sharpe = (np.mean(returns) / std * np.sqrt(252 / 10)) if std > 0 else 0

    # ===== MAX DRAWDOWN từ cumulative P&L =====
    cumulative = np.cumsum(per_trade_pnl) + 100
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / running_max
    max_dd = float(abs(dd.min()) * 100) if len(dd) > 0 else 0

    # ===== PROFIT FACTOR =====
    sw = float(np.sum(wins)) if len(wins) else 0
    sl = float(abs(np.sum(losses))) if len(losses) else 0
    pf = sw / sl if sl > 0 else 0

    # ===== CALMAR =====
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


def backtest_all(data, params):
    """Chạy backtest toàn universe."""
    all_trades = []
    for sym, df in data.items():
        trades = simulate_with_params(df, params)
        all_trades.extend(trades)
    return all_trades


# ============================================================
# OBJECTIVE v2
# ============================================================
def create_objective(data, min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                     max_dd_allowed=MAX_DD_ALLOWED):
    """
    Objective v2: Sharpe × WinRate × (1 - DD) với constraint.
    KHÔNG dùng total_return (vì gây overfit).
    """
    def objective(trial):
        # ===== ĐỀ XUẤT THAM SỐ =====
        params = {
            "entry_threshold": trial.suggest_int(
                "entry_threshold", 60, 80, step=5
            ),
            "exit_threshold": trial.suggest_int(
                "exit_threshold", 30, 50, step=5
            ),
            "use_trend_filter": trial.suggest_categorical(
                "use_trend_filter", [True, False]
            ),
            "sl_mult": trial.suggest_float(
                "sl_mult", 1.0, 2.5, step=0.25
            ),
            "tp_mult": trial.suggest_float(
                "tp_mult", 2.0, 4.0, step=0.25
            ),
            "max_hold": trial.suggest_int(
                "max_hold", 5, 20, step=5
            ),
            "use_dynamic": trial.suggest_categorical(
                "use_dynamic", [True, False]
            ),
        }

        # Đảm bảo R:R >= 1.5
        rr = params["tp_mult"] / params["sl_mult"]
        if rr < 1.5:
            return -99

        # ===== CHẠY BACKTEST =====
        trades = backtest_all(data, params)
        metrics = compute_metrics(trades)

        # ===== CONSTRAINT =====
        if metrics["n_trades"] < min_trades:
            return -99
        if metrics["n_trades"] > max_trades:
            return -99
        if metrics["max_drawdown"] > max_dd_allowed:
            return -99

        # ===== OBJECTIVE MỚI =====
        sharpe = metrics["sharpe"]
        win_rate = metrics["win_rate"]
        dd = metrics["max_drawdown"]
        pf = metrics["profit_factor"]

        score = (
            sharpe * 0.50
            + (win_rate - 45) / 50 * 0.25
            + (pf - 1.0) * 0.15
            - (dd / 100) * 0.10
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
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   BAYESIAN OPTIMIZATION v2                          ║")
    print("║   Objective: Sharpe × WinRate × (1-DD)              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Trials:   {args.trials}")
    print(f"  Days:     {args.days}")
    print(f"  Timeout:  {args.timeout}s")
    print(f"  Constraint: trades {MIN_TRADES}-{MAX_TRADES}, "
          f"DD < {MAX_DD_ALLOWED}%")

    # Tải data 1 lần
    data = load_and_cache_data(days=args.days)
    if len(data) < 5:
        print("❌ Không đủ dữ liệu")
        return

    # Tạo study
    sampler = TPESampler(
        n_startup_trials=20,
        n_ei_candidates=24,
        seed=42,
    )
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="vn30_strategy_optimization_v2",
    )

    # Callback
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == args.trials - 1:
            completed = [t for t in study.trials
                         if t.value is not None and t.value > -99]
            if not completed:
                print(f"  [Trial {trial.number:3d}] chưa có trial hợp lệ")
                return
            best = max(completed, key=lambda t: t.value)
            m = best.user_attrs.get("metrics", {})
            print(
                f"  [Trial {trial.number:3d}/{args.trials}] "
                f"Best: {best.value:.3f} | "
                f"Sharpe: {m.get('sharpe', 0):.2f} | "
                f"WR: {m.get('win_rate', 0):.1f}% | "
                f"DD: {m.get('max_drawdown', 0):.1f}% | "
                f"Trades: {m.get('n_trades', 0)}"
            )

    # Chạy
    print("\n🔍 Bắt đầu tìm kiếm...\n")
    objective = create_objective(data)

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            timeout=args.timeout,
            callbacks=[callback],
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng bởi người dùng")

    # ===== BÁO CÁO =====
    completed = [t for t in study.trials
                 if t.value is not None and t.value > -99]

    if not completed:
        print("\n❌ Không có trial hợp lệ nào")
        print("   → Cần nới lỏng constraint hoặc tăng trials")
        return

    best = max(completed, key=lambda t: t.value)
    print("\n" + "=" * 72)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA v2")
    print("=" * 72)
    print(f"\n📌 Best Trial #{best.number} — score {best.value:.4f}")

    print(f"\n📊 Best Parameters:")
    for k, v in best.params.items():
        print(f"   {k:25s} = {v}")

    print(f"\n📈 Best Metrics:")
    m = best.user_attrs.get("metrics", {})
    for k, v in m.items():
        print(f"   {k:25s} = {v}")

    # ===== TOP 5 =====
    print(f"\n🥇 TOP 5 CONFIGS:")
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    for i, t in enumerate(sorted_trials, 1):
        mm = t.user_attrs.get("metrics", {})
        print(f"\n   #{i} — Trial {t.number} (score {t.value:.3f})")
        print(f"       Entry: {t.params['entry_threshold']} | "
              f"Trend: {t.params['use_trend_filter']} | "
              f"Dynamic: {t.params['use_dynamic']}")
        print(f"       SL: {t.params['sl_mult']} ATR | "
              f"TP: {t.params['tp_mult']} ATR | "
              f"Hold: {t.params['max_hold']}d")
        print(f"       Sharpe: {mm.get('sharpe', 0):.2f} | "
              f"WR: {mm.get('win_rate', 0):.1f}% | "
              f"Return: {mm.get('total_return', 0):.1f}% | "
              f"DD: {mm.get('max_drawdown', 0):.1f}% | "
              f"Trades: {mm.get('n_trades', 0)}")

    # ===== LƯU =====
    os.makedirs("output", exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "version": "v2",
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

    path = "output/optimization_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Đã lưu: {path}")

    # ===== CONFIG ĐỂ COPY =====
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
    "use_trend_filter": {best.params['use_trend_filter']},
    "use_dynamic_weights": {best.params['use_dynamic']},
    "signal_thresholds": {{
        "strong_buy": 75,
        "buy": {best.params['entry_threshold']},
        "hold": {best.params.get('exit_threshold', 40) + 5},
        "sell": {best.params.get('exit_threshold', 40)},
    }},
}}
""")

    # ===== SO SÁNH =====
    print("=" * 72)
    print("📊 SO SÁNH DYNAMIC vs STATIC")
    print("=" * 72)

    dyn = [t for t in completed if t.params.get("use_dynamic")]
    stat = [t for t in completed if not t.params.get("use_dynamic")]

    if dyn:
        bd = max(dyn, key=lambda t: t.value)
        print(f"\n✅ Best DYNAMIC: score {bd.value:.3f}")
        print(f"   Sharpe: {bd.user_attrs['metrics']['sharpe']} | "
              f"DD: {bd.user_attrs['metrics']['max_drawdown']}%")

    if stat:
        bs = max(stat, key=lambda t: t.value)
        print(f"\n✅ Best STATIC:  score {bs.value:.3f}")
        print(f"   Sharpe: {bs.user_attrs['metrics']['sharpe']} | "
              f"DD: {bs.user_attrs['metrics']['max_drawdown']}%")

    if dyn and stat:
        if bd.value > bs.value:
            print(f"\n🏆 DYNAMIC tốt hơn (chênh {bd.value - bs.value:.3f})")
        else:
            print(f"\n🏆 STATIC tốt hơn (chênh {bs.value - bd.value:.3f})")


if __name__ == "__main__":
    main()
