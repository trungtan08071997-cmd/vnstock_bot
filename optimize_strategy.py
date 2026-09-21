"""
Bayesian Optimization với Optuna
Tự động tìm config tối ưu cho chiến lược giao dịch.

Chạy: python optimize_strategy.py --trials 100
"""
import os
import sys
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
# CACHE DATA (tính 1 lần, dùng cho mọi trial)
# ============================================================
_DATA_CACHE = {}


def load_and_cache_data(days=500):
    """Tải + tính chỉ báo 1 lần duy nhất."""
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
        except Exception:
            return sym, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch, s): s for s in config.VN30_LIST}
        for f in as_completed(futures):
            sym, df = f.result()
            if df is not None:
                data[sym] = df
                print(f"  ✓ {sym}: {len(df)} phiên")

    print(f"📊 Đã tải {len(data)}/{len(config.VN30_LIST)} mã")
    _DATA_CACHE = data
    return data


# ============================================================
# BACKTEST VỚI THAM SỐ ĐỘNG
# ============================================================
def simulate_with_params(df, params, start_idx=210):
    """
    Mô phỏng giao dịch với bộ tham số cho trước.
    
    Args:
        df: DataFrame có chỉ báo sẵn
        params: dict chứa hyperparameters
        start_idx: bắt đầu từ phiên nào
    
    Returns:
        List trades
    """
    if df is None or len(df) < start_idx + 30:
        return []

    entry_threshold = params["entry_threshold"]
    use_trend_filter = params["use_trend_filter"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]
    max_hold = params["max_hold"]
    use_dynamic = params["use_dynamic"]
    exit_threshold = params.get("exit_threshold", 40)
    fee = 0.0015

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

        # ===== NẾU ĐANG GIỮ LỆNH =====
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
                if score < exit_threshold:
                    exit_reason = "SIGNAL_REVERSAL"

            if exit_reason:
                gross = (close - entry_price) / entry_price
                net = gross - 2 * fee
                trades.append({
                    "entry_date": df.iloc[entry_idx]["time"],
                    "exit_date": row["time"],
                    "return_pct": net * 100,
                    "days_held": days,
                    "exit_reason": exit_reason,
                    "entry_score": entry_score,
                })
                in_pos = False

        # ===== NẾU KHÔNG GIỮ LỆNH =====
        else:
            score, det = score_at_index(df, i, use_dynamic)

            if score >= entry_threshold:
                # ===== TREND FILTER =====
                if use_trend_filter:
                    ema200 = row.get("ema_l_200")
                    if pd.isna(ema200) or close <= ema200:
                        continue  # Bỏ qua tín hiệu này

                # ===== VÀO LỆNH =====
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
        net = gross - 2 * fee
        trades.append({
            "entry_date": df.iloc[entry_idx]["time"],
            "exit_date": final["time"],
            "return_pct": net * 100,
            "days_held": len(df) - 1 - entry_idx,
            "exit_reason": "END_OF_DATA",
            "entry_score": entry_score,
        })

    return trades


# ============================================================
# METRICS
# ============================================================
def compute_metrics(trades):
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

    total_return = float(
        (np.prod([1 + r/100 for r in returns]) - 1) * 100
    )

    std = float(np.std(returns))
    sharpe = (np.mean(returns) / std * np.sqrt(252 / 10)) if std > 0 else 0

    cumulative = np.cumprod([1 + r/100 for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / running_max
    max_dd = float(abs(dd.min()) * 100) if len(dd) > 0 else 0

    sw = float(np.sum(wins)) if len(wins) else 0
    sl = float(abs(np.sum(losses))) if len(losses) else 0
    pf = sw / sl if sl > 0 else 0

    # Calmar ratio = annual return / max drawdown
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
    """Chạy backtest toàn bộ universe với 1 bộ params."""
    all_trades = []
    for sym, df in data.items():
        trades = simulate_with_params(df, params)
        all_trades.extend(trades)
    return all_trades


# ============================================================
# OBJECTIVE FUNCTION CHO OPTUNA
# ============================================================
def create_objective(data, min_trades=30):
    """Tạo objective function dùng data đã cache."""

    def objective(trial):
        # ===== ĐỀ XUẤT THAM SỐ =====
        params = {
            "entry_threshold": trial.suggest_int(
                "entry_threshold", 55, 80, step=5
            ),
            "exit_threshold": trial.suggest_int(
                "exit_threshold", 30, 50, step=5
            ),
            "use_trend_filter": trial.suggest_categorical(
                "use_trend_filter", [True, False]
            ),
            "sl_mult": trial.suggest_float(
                "sl_mult", 0.8, 3.0, step=0.2
            ),
            "tp_mult": trial.suggest_float(
                "tp_mult", 1.5, 5.0, step=0.25
            ),
            "max_hold": trial.suggest_int(
                "max_hold", 5, 30, step=5
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

        # Lọc trial không đủ mẫu
        if metrics["n_trades"] < min_trades:
            return -99

        # ===== OBJECTIVE = SHARPE =====
        # Kết hợp Sharpe + Win Rate + Profit Factor để balance
        # Sharpe là chính (70%), các chỉ số khác phụ (30%)
        score = (
            metrics["sharpe"] * 0.7
            + (metrics["win_rate"] - 50) / 50 * 0.15
            + (metrics["profit_factor"] - 1.0) * 0.15
        )

        # Penalty nếu drawdown quá lớn
        if metrics["max_drawdown"] > 30:
            score -= (metrics["max_drawdown"] - 30) / 10 * 0.1

        # Lưu metrics vào trial user_attrs
        trial.set_user_attr("metrics", metrics)

        return score

    return objective


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100,
                        help="Số trials chạy (mặc định 100)")
    parser.add_argument("--days", type=int, default=500,
                        help="Số phiên dữ liệu (mặc định 500)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout giây (mặc định 1800 = 30 phút)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   BAYESIAN OPTIMIZATION — TÌM CONFIG TỐI ƯU         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Trials: {args.trials}")
    print(f"  Days:   {args.days}")
    print(f"  Timeout: {args.timeout}s")

    # ===== TẢI DATA 1 LẦN =====
    data = load_and_cache_data(days=args.days)
    if len(data) < 5:
        print("❌ Không đủ dữ liệu để optimize")
        return

    # ===== TẠO STUDY =====
    sampler = TPESampler(
        n_startup_trials=20,  # 20 trial đầu random để explore
        n_ei_candidates=24,   # Số candidate cho Expected Improvement
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
        study_name="vn30_strategy_optimization",
    )

    # ===== CALLBACK IN TIẾN ĐỘ =====
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == args.trials - 1:
            best = study.best_trial
            m = best.user_attrs.get("metrics", {})
            print(
                f"  [Trial {trial.number:3d}/{args.trials}] "
                f"Best score: {best.value:.3f} | "
                f"Sharpe: {m.get('sharpe', 0):.2f} | "
                f"WinRate: {m.get('win_rate', 0):.1f}% | "
                f"Return: {m.get('total_return', 0):.1f}% | "
                f"Trades: {m.get('n_trades', 0)}"
            )

    # ===== CHẠY OPTIMIZATION =====
    print("\n🔍 Bắt đầu tìm kiếm config tối ưu...")
    objective = create_objective(data, min_trades=30)

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

    # ===== BÁO CÁO KẾT QUẢ =====
    print("\n" + "=" * 72)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA")
    print("=" * 72)

    best = study.best_trial
    print(f"\n📌 Best Trial #{best.number}")
    print(f"   Objective score: {best.value:.4f}")

    print(f"\n📊 Best Parameters:")
    for k, v in best.params.items():
        print(f"   {k:25s} = {v}")

    print(f"\n📈 Best Metrics:")
    m = best.user_attrs.get("metrics", {})
    for k, v in m.items():
        print(f"   {k:25s} = {v}")

    # ===== TOP 5 TRIALS =====
    print(f"\n🥇 TOP 5 CONFIGS:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -99],
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    for i, t in enumerate(sorted_trials, 1):
        m = t.user_attrs.get("metrics", {})
        print(f"\n   #{i} — Trial {t.number} (score {t.value:.3f})")
        print(f"       Entry: {t.params['entry_threshold']} | "
              f"Trend filter: {t.params['use_trend_filter']} | "
              f"Dynamic: {t.params['use_dynamic']}")
        print(f"       SL: {t.params['sl_mult']} ATR | "
              f"TP: {t.params['tp_mult']} ATR | "
              f"Hold: {t.params['max_hold']}d")
        print(f"       Sharpe: {m.get('sharpe', 0):.2f} | "
              f"WinRate: {m.get('win_rate', 0):.1f}% | "
              f"Return: {m.get('total_return', 0):.1f}% | "
              f"DD: {m.get('max_drawdown', 0):.1f}% | "
              f"Trades: {m.get('n_trades', 0)}")

    # ===== LƯU KẾT QUẢ =====
    os.makedirs("output", exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials
                             if t.value is not None and t.value > -99]),
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

    # ===== IN CONFIG ĐỂ COPY VÀO CODE =====
    print("\n" + "=" * 72)
    print("📋 CONFIG ĐỂ COPY VÀO CODE")
    print("=" * 72)
    print(f"""
# Config tối ưu cho backtest_engine.py:
BEST_PARAMS = {json.dumps(best.params, indent=4)}

# Hoặc dùng trong main.py:
BEST_ENTRY_THRESHOLD = {best.params.get('entry_threshold')}
BEST_USE_TREND_FILTER = {best.params.get('use_trend_filter')}
BEST_SL_MULT = {best.params.get('sl_mult')}
BEST_TP_MULT = {best.params.get('tp_mult')}
BEST_MAX_HOLD = {best.params.get('max_hold')}
BEST_USE_DYNAMIC = {best.params.get('use_dynamic')}
""")

    # ===== SO SÁNH DYNAMIC vs STATIC =====
    print("\n" + "=" * 72)
    print("📊 SO SÁNH DYNAMIC vs STATIC (từ optimization)")
    print("=" * 72)

    dynamic_trials = [t for t in study.trials
                       if t.value is not None and t.value > -99
                       and t.params.get("use_dynamic") is True]
    static_trials = [t for t in study.trials
                      if t.value is not None and t.value > -99
                      and t.params.get("use_dynamic") is False]

    if dynamic_trials:
        best_dyn = max(dynamic_trials, key=lambda t: t.value)
        print(f"\n✅ Best DYNAMIC: score {best_dyn.value:.3f}")
        print(f"   Params: {best_dyn.params}")
        print(f"   Metrics: {best_dyn.user_attrs.get('metrics')}")

    if static_trials:
        best_stat = max(static_trials, key=lambda t: t.value)
        print(f"\n✅ Best STATIC:  score {best_stat.value:.3f}")
        print(f"   Params: {best_stat.params}")
        print(f"   Metrics: {best_stat.user_attrs.get('metrics')}")

    if dynamic_trials and static_trials:
        if best_dyn.value > best_stat.value:
            print(f"\n🏆 KẾT LUẬN: DYNAMIC tốt hơn STATIC "
                  f"(chênh {best_dyn.value - best_stat.value:.3f})")
        else:
            print(f"\n🏆 KẾT LUẬN: STATIC tốt hơn DYNAMIC "
                  f"(chênh {best_stat.value - best_dyn.value:.3f})")


if __name__ == "__main__":
    main()
