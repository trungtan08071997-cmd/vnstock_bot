"""
Validate Out-of-Sample — Kiểm tra overfitting
Chia dữ liệu thành nhiều đoạn, chạy config tối ưu trên TỪNG đoạn.
Nếu config tốt trên hầu hết các đoạn → không overfit.
"""
import json
from datetime import datetime
from backtest_engine import load_vn30_data
from optimize_strategy import simulate_with_params, compute_metrics
import config


def main():
    print("=" * 72)
    print("🔬 VALIDATE OUT-OF-SAMPLE — Kiểm tra overfitting")
    print("=" * 72)

    # Config tối ưu
    tc = config.TRADING_CONFIG
    params = {
        "entry_threshold": tc["entry_threshold"],
        "exit_threshold": tc["exit_threshold"],
        "sl_mult": tc["sl_mult"],
        "tp_mult": tc["tp_mult"],
        "max_hold": tc["max_hold_days"],
        "use_trend_filter": tc["use_trend_filter"],
        "use_dynamic": tc["use_dynamic_weights"],
    }

    print(f"\n⚙️  Config: {params}")

    # Tải dữ liệu
    data = load_vn30_data(days=750)  # 750 phiên ~ 3 năm
    if len(data) < 5:
        print("❌ Không đủ dữ liệu")
        return

    # ===== CHIA THÀNH 4 GIAI ĐOẠN =====
    periods = [
        ("Giai đoạn 1 (2024 Q1-Q2)", 200, 350),
        ("Giai đoạn 2 (2024 Q3-Q4)", 350, 500),
        ("Giai đoạn 3 (2025 Q1-Q2)", 500, 650),
        ("Giai đoạn 4 (2025 Q3-Q4)", 650, 750),
    ]

    print(f"\n📊 Chia dữ liệu thành {len(periods)} giai đoạn:\n")

    results = []
    for name, start_idx, end_idx in periods:
        all_trades = []
        for sym, df in data.items():
            if len(df) < end_idx:
                continue
            df_slice = df.iloc[:end_idx].copy()
            trades = simulate_with_params(df_slice, params, start_idx=start_idx)
            all_trades.extend(trades)

        metrics = compute_metrics(all_trades)
        results.append({
            "period": name,
            "metrics": metrics,
        })

        # In kết quả
        print(f"  {name}:")
        print(f"     Trades: {metrics['n_trades']} | "
              f"WinRate: {metrics['win_rate']}% | "
              f"Return: {metrics['total_return']}% | "
              f"Sharpe: {metrics['sharpe']} | "
              f"DD: {metrics['max_drawdown']}% | "
              f"PF: {metrics['profit_factor']}")
        print()

    # ===== ĐÁNH GIÁ =====
    print("=" * 72)
    print("📊 ĐÁNH GIÁ OVERFITTING")
    print("=" * 72)

    # Đếm số giai đoạn có lợi nhuận
    positive_periods = sum(1 for r in results
                            if r["metrics"]["total_return"] > 0)
    total_periods = len(results)

    # Sharpe trung bình
    avg_sharpe = sum(r["metrics"]["sharpe"] for r in results) / total_periods

    # Max drawdown trung bình
    avg_dd = sum(r["metrics"]["max_drawdown"] for r in results) / total_periods

    # Win rate trung bình
    avg_wr = sum(r["metrics"]["win_rate"] for r in results) / total_periods

    print(f"\n📈 Tổng hợp:")
    print(f"   Giai đoạn có lãi:  {positive_periods}/{total_periods}")
    print(f"   Sharpe trung bình:  {avg_sharpe:.3f}")
    print(f"   Win rate trung bình: {avg_wr:.2f}%")
    print(f"   Drawdown trung bình: {avg_dd:.2f}%")

    print(f"\n🎯 KẾT LUẬN:")
    if positive_periods >= 3 and avg_sharpe >= 0.5:
        verdict = "✅ PASS — Config không overfit, dùng được"
        confidence = "HIGH"
    elif positive_periods >= 2 and avg_sharpe >= 0.3:
        verdict = "⚠️  CẢNH BÁO — Có dấu hiệu overfit nhẹ"
        confidence = "MEDIUM"
    else:
        verdict = "❌ FAIL — Config bị overfit, không nên dùng"
        confidence = "LOW"

    print(f"   {verdict}")
    print(f"   Confidence: {confidence}")

    # ===== LƯU BÁO CÁO =====
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": params,
        "periods": results,
        "summary": {
            "positive_periods": positive_periods,
            "total_periods": total_periods,
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_win_rate": round(avg_wr, 2),
            "avg_drawdown": round(avg_dd, 2),
            "verdict": verdict,
            "confidence": confidence,
        },
    }

    path = "output/validation_result.json"
    import os
    os.makedirs("output", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Đã lưu: {path}")

    # In config để copy
    print("\n" + "=" * 72)
    if confidence == "HIGH":
        print("✅ CÓ THỂ ÁP DỤNG VÀO BOT THẬT")
        print("   Theo dõi 1-2 tuần với vốn nhỏ trước khi tăng.")
    elif confidence == "MEDIUM":
        print("⚠️  NÊN THEO DÕI THÊM")
        print("   Chạy walk-forward thêm 2-3 lần để xác nhận.")
    else:
        print("❌ KHÔNG NÊN ÁP DỤNG")
        print("   Cần chạy lại optimization với dữ liệu khác.")
    print("=" * 72)


if __name__ == "__main__":
    main()
