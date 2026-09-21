"""
Backtest Engine - So sánh Dynamic Weights vs Static Weights
Chạy: python backtest_engine.py
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from data_fetcher import get_stock_history
from technical_analyzer import TechnicalAnalyzer


# ============================================================
# STATIC WEIGHTS (bản cũ để so sánh)
# ============================================================
STATIC_WEIGHTS = {
    "ema_ribbon": 25,
    "macd": 20,
    "rsi": 15,
    "volume": 15,
    "bollinger": 15,
    "atr": 10,
}


# ============================================================
# STATIC ANALYZER (bản cũ, không có dynamic weights)
# ============================================================
class StaticAnalyzer:
    """Technical Analyzer với weight cố định (baseline để so sánh)."""

    def __init__(self, df):
        self.df = df.copy()
        # Tái sử dụng tính toán chỉ báo từ TechnicalAnalyzer
        ta = TechnicalAnalyzer(df)
        # Nhưng ghi đè weight thành static
        self.df = ta.df
        self.weights = STATIC_WEIGHTS
        self.regime = "STATIC"

    def technical_score(self):
        """Chấm điểm với weight cố định."""
        # Tạo instance tạm để dùng các hàm _score_* của TechnicalAnalyzer
        ta = TechnicalAnalyzer(self.df)
        s = ta.get_current_signals()
        if not s:
            return 0, {}

        # Dùng static weights
        W = STATIC_WEIGHTS
        total_w = sum(W.values())

        ema_raw, ema_detail = ta._score_ema_ribbon(s)
        macd_raw, macd_detail = ta._score_macd(s)
        rsi_raw, rsi_detail = ta._score_rsi(s)
        vol_raw, vol_detail = ta._score_volume(s)
        bb_raw, bb_detail = ta._score_bollinger(s)
        atr_raw, atr_detail = ta._score_atr(s)

        # Scale về 100 và dùng static weights
        score = (
            (ema_raw / 25 * 100) * W["ema_ribbon"]
            + (macd_raw / 20 * 100) * W["macd"]
            + (rsi_raw / 15 * 100) * W["rsi"]
            + (vol_raw / 15 * 100) * W["volume"]
            + (bb_raw / 15 * 100) * W["bollinger"]
            + (atr_raw / 10 * 100) * W["atr"]
        ) / total_w

        return round(score, 1), {
            "regime": "STATIC",
            "weights": W,
            "confidence": "N/A",
            "entry_action": "N/A",
            "risk_management": {},
        }


# ============================================================
# MÔ PHỎNG GIAO DỊCH
# ============================================================
class TradeSimulator:
    """
    Mô phỏng giao dịch dựa trên tín hiệu.
    
    Quy tắc:
    - Vào lệnh khi score >= 60 (MUA)
    - Thoát khi: giá chạm SL, hoặc chạm TP1, hoặc score < 40
    - Giữ tối đa 20 ngày
    - Phí giao dịch: 0.15% mỗi chiều (mua + bán = 0.3%)
    """

    FEE_RATE = 0.0015   # 0.15% mỗi chiều
    MAX_HOLD_DAYS = 20

    def __init__(self, analyzer_class, stock_data):
        """
        Args:
            analyzer_class: TechnicalAnalyzer hoặc StaticAnalyzer
            stock_data: dict {symbol: DataFrame}
        """
        self.analyzer_class = analyzer_class
        self.stock_data = stock_data

    def simulate(self, symbol, lookback=100, start_idx=200):
        """
        Chạy simulation cho 1 mã.
        
        Args:
            symbol: mã cổ phiếu
            lookback: số phiên dùng để tính chỉ báo
            start_idx: bắt đầu từ phiên thứ mấy (cần đủ data để tính)
        
        Returns:
            dict: kết quả backtest cho mã này
        """
        df = self.stock_data.get(symbol)
        if df is None or len(df) < start_idx + 50:
            return None

        trades = []
        in_position = False
        entry_price = 0
        entry_idx = 0
        entry_signal = None
        stop_loss = None
        take_profit = None

        for i in range(start_idx, len(df) - 1):
            # Lấy dữ liệu đến ngày i
            df_slice = df.iloc[:i+1].copy()
            current_close = df.iloc[i]["close"]
            next_open = df.iloc[i+1]["open"]

            # ===== NẾU ĐANG GIỮ LỆNH =====
            if in_position:
                days_held = i - entry_idx

                # Check exit conditions
                exit_reason = None

                # 1. Stop loss
                if current_close <= stop_loss:
                    exit_reason = "STOP_LOSS"
                # 2. Take profit
                elif current_close >= take_profit:
                    exit_reason = "TAKE_PROFIT"
                # 3. Max hold
                elif days_held >= self.MAX_HOLD_DAYS:
                    exit_reason = "MAX_HOLD"
                # 4. Signal đảo chiều
                else:
                    try:
                        analyzer = self.analyzer_class(df_slice)
                        score, _ = analyzer.technical_score()
                        if score < 40:
                            exit_reason = "SIGNAL_REVERSAL"
                    except Exception:
                        pass

                if exit_reason:
                    # Thoát tại close của ngày i
                    exit_price = current_close
                    # Tính P&L (trừ phí 2 chiều)
                    gross_return = (exit_price - entry_price) / entry_price
                    net_return = gross_return - 2 * self.FEE_RATE

                    trades.append({
                        "entry_date": df.iloc[entry_idx]["time"],
                        "exit_date": df.iloc[i]["time"],
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "days_held": days_held,
                        "return_pct": round(net_return * 100, 2),
                        "exit_reason": exit_reason,
                        "signal_score": entry_signal,
                    })
                    in_position = False
                    entry_signal = None

            # ===== NẾU KHÔNG GIỮ LỆNH =====
            else:
                try:
                    analyzer = self.analyzer_class(df_slice)
                    score, details = analyzer.technical_score()

                    # Vào lệnh nếu score >= 60
                    if score >= 60:
                        # Vào tại open ngày tiếp theo (tránh look-ahead bias)
                        entry_price = next_open
                        entry_idx = i + 1
                        entry_signal = score
                        in_position = True

                        # Đặt SL/TP dựa trên ATR
                        rm = details.get("risk_management", {})
                        stop_loss = rm.get("stop_loss") or (entry_price * 0.95)
                        take_profit = rm.get("take_profit_1") or (entry_price * 1.08)
                except Exception:
                    continue

        # Đóng lệnh cuối nếu còn
        if in_position:
            final_close = df.iloc[-1]["close"]
            gross_return = (final_close - entry_price) / entry_price
            net_return = gross_return - 2 * self.FEE_RATE
            trades.append({
                "entry_date": df.iloc[entry_idx]["time"],
                "exit_date": df.iloc[-1]["time"],
                "entry_price": round(entry_price, 2),
                "exit_price": round(final_close, 2),
                "days_held": len(df) - 1 - entry_idx,
                "return_pct": round(net_return * 100, 2),
                "exit_reason": "END_OF_DATA",
                "signal_score": entry_signal,
            })

        return {
            "symbol": symbol,
            "trades": trades,
            "n_trades": len(trades),
        }


# ============================================================
# TÍNH METRICS
# ============================================================
def compute_metrics(all_trades):
    """Tính các metrics từ danh sách trades."""
    if not all_trades:
        return {
            "n_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "avg_hold_days": 0,
        }

    returns = [t["return_pct"] for t in all_trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    # Win rate
    win_rate = len(wins) / len(returns) * 100 if returns else 0

    # Average
    avg_return = np.mean(returns)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Total return (compounded)
    total_return = (np.prod([1 + r/100 for r in returns]) - 1) * 100

    # Sharpe (giả định rf = 0)
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / 10)  # ~10 ngày/lệnh
    else:
        sharpe = 0

    # Max drawdown (từ cumulative returns)
    cumulative = np.cumprod([1 + r/100 for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0

    # Profit factor
    sum_wins = sum(wins) if wins else 0
    sum_losses = abs(sum(losses)) if losses else 0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0

    # Average hold days
    avg_hold = np.mean([t["days_held"] for t in all_trades])

    return {
        "n_trades": len(returns),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "total_return": round(total_return, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_hold_days": round(avg_hold, 1),
    }


# ============================================================
# CHẠY BACKTEST
# ============================================================
def load_vn30_data(days=750):
    """Tải dữ liệu VN30 (750 phiên ~ 3 năm)."""
    print(f"\n📥 Đang tải dữ liệu VN30 ({days} phiên)...")
    stock_data = {}

    symbols = config.VN30_LIST

    def fetch(sym):
        try:
            df = get_stock_history(sym, days)
            return sym, df
        except Exception as e:
            return sym, None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, s): s for s in symbols}
        for future in as_completed(futures):
            sym, df = future.result()
            if df is not None and len(df) >= 250:
                stock_data[sym] = df
                print(f"  ✓ {sym}: {len(df)} phiên")
            else:
                print(f"  ✗ {sym}: không đủ dữ liệu")

    print(f"📊 Có {len(stock_data)}/{len(symbols)} mã")
    return stock_data


def run_backtest(name, analyzer_class, stock_data):
    """Chạy backtest cho 1 hệ thống."""
    print(f"\n{'='*60}")
    print(f"🚀 CHẠY BACKTEST: {name}")
    print(f"{'='*60}")

    simulator = TradeSimulator(analyzer_class, stock_data)
    all_trades = []
    per_symbol = {}

    for i, sym in enumerate(stock_data.keys(), 1):
        print(f"  [{i}/{len(stock_data)}] Đang backtest {sym}...", end=" ")
        result = simulator.simulate(sym)
        if result and result["trades"]:
            all_trades.extend(result["trades"])
            per_symbol[sym] = result
            print(f"{result['n_trades']} lệnh")
        else:
            print("không có lệnh")

    metrics = compute_metrics(all_trades)
    return {
        "name": name,
        "metrics": metrics,
        "trades": all_trades,
        "per_symbol": per_symbol,
    }


# ============================================================
# SO SÁNH & BÁO CÁO
# ============================================================
def print_comparison(static_result, dynamic_result):
    """In bảng so sánh 2 hệ thống."""
    s = static_result["metrics"]
    d = dynamic_result["metrics"]

    print("\n" + "="*70)
    print("📊 KẾT QUẢ SO SÁNH: STATIC vs DYNAMIC")
    print("="*70)

    header = f"{'Metric':<25} {'STATIC':>15} {'DYNAMIC':>15} {'Δ':>12}"
    print(header)
    print("-" * 70)

    metrics_to_compare = [
        ("Số lệnh", "n_trades", "int"),
        ("Win rate (%)", "win_rate", "pct"),
        ("Lợi nhuận TB (%)", "avg_return", "pct"),
        ("Tổng lợi nhuận (%)", "total_return", "pct"),
        ("Sharpe Ratio", "sharpe", "float"),
        ("Max Drawdown (%)", "max_drawdown", "neg"),
        ("Profit Factor", "profit_factor", "float"),
        ("Lãi TB khi thắng (%)", "avg_win", "pct"),
        ("Lỗ TB khi thua (%)", "avg_loss", "pct"),
        ("Ngày giữ TB", "avg_hold_days", "float"),
    ]

    for label, key, fmt in metrics_to_compare:
        sv = s.get(key, 0)
        dv = d.get(key, 0)
        delta = dv - sv

        if fmt == "int":
            s_str, d_str = f"{sv}", f"{dv}"
            delta_str = f"{delta:+d}"
        elif fmt == "pct":
            s_str, d_str = f"{sv:.2f}", f"{dv:.2f}"
            delta_str = f"{delta:+.2f}"
        elif fmt == "neg":
            s_str, d_str = f"-{sv:.1f}", f"-{dv:.1f}"
            delta_str = f"{-delta:+.1f}"
        else:
            s_str, d_str = f"{sv:.2f}", f"{dv:.2f}"
            delta_str = f"{delta:+.2f}"

        # Đánh dấu cái tốt hơn
        better = ""
        if key in ("win_rate", "avg_return", "total_return", "sharpe",
                   "profit_factor", "avg_win"):
            better = " ✅" if dv > sv else (" ❌" if dv < sv else "")
        elif key in ("max_drawdown", "avg_loss"):
            better = " ✅" if dv < sv else (" ❌" if dv > sv else "")

        print(f"{label:<25} {s_str:>15} {d_str:>15} {delta_str:>12}{better}")

    print("="*70)

    # ===== KẾT LUẬN =====
    print("\n🎯 KẾT LUẬN:")

    # Đếm số metric dynamic tốt hơn
    dynamic_wins = 0
    static_wins = 0

    if d["win_rate"] > s["win_rate"]: dynamic_wins += 1
    else: static_wins += 1

    if d["total_return"] > s["total_return"]: dynamic_wins += 1
    else: static_wins += 1

    if d["sharpe"] > s["sharpe"]: dynamic_wins += 1
    else: static_wins += 1

    if d["max_drawdown"] < s["max_drawdown"]: dynamic_wins += 1
    else: static_wins += 1

    if d["profit_factor"] > s["profit_factor"]: dynamic_wins += 1
    else: static_wins += 1

    print(f"  Dynamic thắng: {dynamic_wins}/5 tiêu chí")
    print(f"  Static thắng:  {static_wins}/5 tiêu chí")

    if dynamic_wins > static_wins:
        print("\n  ✅ KHUYẾN NGHỊ: Dùng DYNAMIC WEIGHTS")
        print("     → Hệ thống động vượt trội trên dữ liệu VN30")
    elif static_wins > dynamic_wins:
        print("\n  ⚠️  KHUYẾN NGHỊ: Dùng STATIC WEIGHTS")
        print("     → Hệ thống động chưa hiệu quả, cần tinh chỉnh")
    else:
        print("\n  ⚖️  Hai hệ thống tương đương")
        print("     → Cần thêm dữ liệu hoặc tinh chỉnh threshold")


def save_report(static_result, dynamic_result, filename="backtest_report.json"):
    """Lưu báo cáo dạng JSON."""
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", filename)

    def clean_trades(trades):
        return trades[:100]  # Chỉ lưu 100 trades đầu

    report = {
        "generated_at": datetime.now().isoformat(),
        "static": {
            "metrics": static_result["metrics"],
            "sample_trades": clean_trades(static_result["trades"]),
        },
        "dynamic": {
            "metrics": dynamic_result["metrics"],
            "sample_trades": clean_trades(dynamic_result["trades"]),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 Đã lưu báo cáo: {path}")
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BACKTEST: DYNAMIC WEIGHTS vs STATIC WEIGHTS (VN30)     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Tải dữ liệu
    stock_data = load_vn30_data(days=750)
    if len(stock_data) < 5:
        print("❌ Không đủ dữ liệu để backtest. Cần ít nhất 5 mã.")
        return

    # 2. Chạy Static
    static_result = run_backtest(
        "STATIC WEIGHTS",
        StaticAnalyzer,
        stock_data
    )

    # 3. Chạy Dynamic
    dynamic_result = run_backtest(
        "DYNAMIC WEIGHTS",
        TechnicalAnalyzer,
        stock_data
    )

    # 4. So sánh
    print_comparison(static_result, dynamic_result)

    # 5. Lưu
    save_report(static_result, dynamic_result)


if __name__ == "__main__":
    main()
