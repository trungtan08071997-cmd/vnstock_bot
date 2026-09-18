import time
import pandas as pd
from lotusmarket import fetchers
from datetime import datetime, timedelta
import config

def get_stock_history(symbol, days=None):
    """Lấy dữ liệu OHLCV từ Entrade (nguồn ổn định cho GitHub Actions)."""
    if days is None:
        days = config.HISTORY_DAYS
    try:
        # Lấy lịch sử giá từ Entrade
        history = fetchers.entrade_history(symbol, days)
        if not history:
            return None
        df = pd.DataFrame([h.__dict__ for h in history])
        df = df.rename(columns={'tradingDate': 'time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
        return df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(days).reset_index(drop=True)
    except Exception as e:
        print(f"    ⚠️ Lỗi lấy dữ liệu {symbol}: {str(e)[:80]}")
        return None

def get_financial_ratios(symbol):
    """Lấy chỉ số tài chính từ KBS (P/E, P/B, ROE, EPS)."""
    try:
        # Lấy chỉ số cơ bản từ KBS
        quote = fetchers.kbs(symbol)
        if not quote:
            return None
        return {
            "symbol": symbol,
            "pe": getattr(quote, 'pe', None),
            "pb": getattr(quote, 'pb', None),
            "roe": getattr(quote, 'roe', None),
            "eps": getattr(quote, 'eps', None),
            "roa": getattr(quote, 'roa', None),
            "net_margin": getattr(quote, 'net_margin', None),
        }
    except Exception as e:
        print(f"    ⚠️ Lỗi lấy chỉ số {symbol}: {str(e)[:80]}")
        return None

def get_vnindex_history(days=None):
    """Lấy dữ liệu VN-Index."""
    return get_stock_history("VNINDEX", days)

def get_foreign_flow(symbol):
    """Lấy dữ liệu khối ngoại mua/bán ròng."""
    try:
        quote = fetchers.vps(symbol)
        if not quote:
            return {"symbol": symbol, "foreign_net": 0}
        return {"symbol": symbol, "foreign_net": getattr(quote, 'foreign_net_vol', 0)}
    except Exception:
        return {"symbol": symbol, "foreign_net": 0}
