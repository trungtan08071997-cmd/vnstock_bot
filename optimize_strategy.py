def compute_metrics(trades, position_size=0.10):
    """
    Tính metrics với POSITION SIZE CỐ ĐỊNH.
    
    Args:
        trades: list trades
        position_size: % vốn mỗi lệnh (mặc định 10%)
    
    Returns:
        dict metrics
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

    # ===== TOTAL RETURN - FIXED POSITION SIZE =====
    # Mỗi lệnh dùng position_size vốn → return thực = r × position_size
    # Total = sum(r × position_size) - KHÔNG compound
    per_trade_pnl = returns * position_size  # Đóng góp vào vốn tổng
    total_return = float(np.sum(per_trade_pnl))

    # ===== SHARPE (giữ nguyên) =====
    std = float(np.std(returns))
    sharpe = (np.mean(returns) / std * np.sqrt(252 / 10)) if std > 0 else 0

    # ===== MAX DRAWDOWN =====
    # Cumulative P&L (không compound)
    cumulative = np.cumsum(per_trade_pnl) + 100  # Bắt đầu từ 100
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
