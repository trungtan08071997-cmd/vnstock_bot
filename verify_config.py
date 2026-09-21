"""
Verify Config - Xác nhận config tối ưu đã áp dụng đúng.
Kỳ vọng: Kết quả giống với lúc optimize (Return ~198.9%, Sharpe ~0.599)
"""
import config
from backtest_engine import load_vn30_data
from optimize_strategy import simulate_with_params, compute_metrics, backtest_all


def main():
    print("=" * 72)
    print("🔍 VERIFY CONFIG — Kiểm tra config đã áp dụng đúng chưa")
    print("=" * 72)

    # Lấy config hiện tại
    tc = config.TRADING_CONFIG
    print(f"\n⚙️  Config đang dùng:")
    for k, v in tc.items():
        if k != "signal_thresholds":
            print(f"   {k:25s} = {v}")

    # Config kỳ vọng (từ optimization)
    expected = {
        "entry_threshold": 55,
        "exit_threshold": 30,
        "sl_mult": 3.0,
        "tp_mult": 5.0,
        "max_hold_days": 25,
        "use_trend_filter": False,
        "use_dynamic_weights": False,
    }

    print(f"\n🎯 Config kỳ vọng từ optimization:")
    for k, v in expected.items():
        print(f"   {k:25s} = {v}")

    # So sánh
    print(f"\n📊 So sánh:")
    mismatch = []
    for k, v in expected.items():
        actual = tc.get(k)
        status = "✅" if actual == v else "❌"
        if actual != v:
            mismatch.append(k)
        print(f"   {k:25s} {status} actual={actual} | expected={v}")

    if mismatch:
        print(f"\n⚠️  Có {len(mismatch)} config KHÔNG khớp: {mismatch}")
        print("   → Kiểm tra lại config.py")
        return

    print("\n✅ Config đã đúng. Bắt đầu chạy backtest verify...")

    # Chạy backtest với config hiện tại
    data = load_vn30_data(days=500)
    if len(data) < 5:
        print("❌ Không đủ dữ liệu")
        return

    params = {
        "entry_threshold": tc["entry_threshold"],
        "exit_threshold": tc["exit_threshold"],
        "sl_mult": tc["sl_mult"],
        "tp_mult": tc["tp_mult"],
        "max_hold": tc["max_hold_days"],
        "use_trend_filter": tc["use_trend_filter"],
        "use_dynamic": tc["use_dynamic_weights"],
    }

    print(f"\n🚀 Chạy backtest với params: {params}")
    trades = backtest_all(data, params)
    metrics = compute_metrics(trades)

    print(f"\n📊 KẾT QUẢ BACKTEST:")
    for k, v in metrics.items():
        print(f"   {k:25s} = {v}")

    # So sánh với kết quả optimization
    expected_metrics = {
        "n_trades": 162,
        "win_rate": 46.91,
        "total_return": 198.94,
        "sharpe": 0.599,
        "max_drawdown": 40.84,
        "profit_factor": 1.374,
    }

    print(f"\n🎯 Kỳ vọng từ optimization:")
    for k, v in expected_metrics.items():
        actual = metrics.get(k, 0)
        diff = abs(actual - v)
        tol = max(abs(v) * 0.05, 1)  # Tolerance 5%
        status = "✅" if diff <= tol else "⚠️"
        print(f"   {k:25s} {status} actual={actual} | expected={v} | diff={diff:.2f}")

    print("\n" + "=" * 72)
    if all(abs(metrics.get(k, 0) - v) <= max(abs(v) * 0.05, 1)
           for k, v in expected_metrics.items()):
        print("✅ VERIFY THÀNH CÔNG — Config áp dụng đúng, kết quả khớp")
    else:
        print("⚠️  KẾT QUẢ KHÁC BIỆT — Kiểm tra lại code")
    print("=" * 72)


if __name__ == "__main__":
    main()
