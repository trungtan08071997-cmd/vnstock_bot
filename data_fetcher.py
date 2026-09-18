"""
Data fetcher dùng aipriceaction
"""
import pandas as pd
import config

try:
    from aipriceaction import AIPriceAction
    _USE_AIPA = True
except ImportError:
    print("⚠️ Chưa cài aipriceaction")
    _USE_AIPA = False


def _normalize_df(df):
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "time": "time", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })
    required = ["time", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return None
    df = df[required].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_stock_history(symbol, days=None):
    if not _USE_AIPA:
        return None
    if days is None:
        days = config.HISTORY_DAYS
    try:
        client = AIPriceAction()
        df = client.get_ohlcv(symbol, interval="1D", limit=days)
        df = _normalize_df(df)
        if df is not None and len(df) >= 50:
            return df
    except Exception as e:
        print(f"  ⚠️ {symbol}: {str(e)[:80]}")
    return None


def get_financial_ratios(symbol):
    """aipriceaction không hỗ trợ P/E, P/B, ROE"""
    return None


def get_vnindex_history(days=None):
    return get_stock_history("VNINDEX", days)


def get_foreign_flow(symbol):
    return {"symbol": symbol, "foreign_net": 0}