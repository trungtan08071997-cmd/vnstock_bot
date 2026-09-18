"""
Data fetcher - dùng lotusmarket, tự động detect tên cột
"""
import time
import pandas as pd
from datetime import datetime, timedelta
import config


def _normalize_dataframe(df, days):
    """
    Chuẩn hóa DataFrame từ lotusmarket về format chuẩn:
    time, open, high, low, close, volume
    """
    if df is None or df.empty:
        return None

    # In debug để biết cột thực tế
    print(f"      [DEBUG] Columns: {df.columns.tolist()}")

    # ===== TỰ ĐỘNG DETECT TÊN CỘT =====
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Cột thời gian
        if col_lower in ('time', 'date', 'tradingdate', 'trading_date',
                          'datetime', 'timestamp', 't'):
            rename_map[col] = 'time'
        # Cột giá mở cửa
        elif col_lower in ('open', 'o', 'openprice', 'open_price', 'giamo'):
            rename_map[col] = 'open'
        # Cột giá cao nhất
        elif col_lower in ('high', 'h', 'highprice', 'high_price', 'giacao'):
            rename_map[col] = 'high'
        # Cột giá thấp nhất
        elif col_lower in ('low', 'l', 'lowprice', 'low_price', 'giathap'):
            rename_map[col] = 'low'
        # Cột giá đóng cửa
        elif col_lower in ('close', 'c', 'closeprice', 'close_price', 'giadong'):
            rename_map[col] = 'close'
        # Cột khối lượng
        elif col_lower in ('volume', 'v', 'vol', 'nmvolume', 'khoiluong'):
            rename_map[col] = 'volume'

    df = df.rename(columns=rename_map)

    # ===== NẾU INDEX LÀ THỜI GIAN =====
    if 'time' not in df.columns:
        # Thử dùng index
        if df.index.name and 'date' in str(df.index.name).lower():
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'time'})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'time'})

    # ===== KIỂM TRA CỘT CẦN THIẾT =====
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"      [DEBUG] Missing: {missing}")
        return None

    # ===== FORMAT LẠI =====
    try:
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"      [DEBUG] Time parse error: {e}")
        return None

    df = df[required].tail(days).reset_index(drop=True)

    # Chuyển giá sang số
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Bỏ dòng NaN
    df = df.dropna(subset=['close']).reset_index(drop=True)

    if len(df) < 50:
        return None

    return df


# ============================================================
# LỊCH SỬ GIÁ
# ============================================================
def get_stock_history(symbol, days=None):
    """Lấy dữ liệu OHLCV từ lotusmarket."""
    if days is None:
        days = config.HISTORY_DAYS

    try:
        from lotusmarket import fetchers

        # Thử Entrade
        history = fetchers.entrade_history(symbol, days)

        if history is None:
            print(f"      [DEBUG] history is None cho {symbol}")
            return None

        print(f"      [DEBUG] Type: {type(history).__name__}")

        # Chuyển về DataFrame
        if isinstance(history, pd.DataFrame):
            df = history.copy()
        elif isinstance(history, list):
            if len(history) == 0:
                return None
            first = history[0]
            if hasattr(first, '_asdict'):
                df = pd.DataFrame([h._asdict() for h in history])
            elif hasattr(first, '__dict__'):
                df = pd.DataFrame([vars(h) for h in history])
            elif isinstance(first, dict):
                df = pd.DataFrame(history)
            else:
                print(f"      [DEBUG] Unknown item: {type(first).__name__}")
                return None
        else:
            print(f"      [DEBUG] Unknown type")
            return None

        result = _normalize_dataframe(df, days)
        if result is not None:
            print(f"      ✓ OK ({len(result)} phiên)")
            time.sleep(0.5)
        return result

    except Exception as e:
        print(f"      ❌ Lỗi {symbol}: {type(e).__name__}: {str(e)[:100]}")
        return None


# ============================================================
# CHỈ SỐ CƠ BẢN - Trả None để bot dùng fallback
# ============================================================
def get_financial_ratios(symbol):
    """
    Trả None để bot tự động dùng SCORING_WEIGHTS_NO_FUNDAMENTAL.
    Bot sẽ chấm điểm dựa trên technical + sentiment.
    """
    return {
        "symbol": symbol,
        "pe": None, "pb": None, "roe": None, "eps": None,
        "roa": None, "net_margin": None,
        "source": "unavailable",
    }


# ============================================================
# VNINDEX - Thử nhiều mã
# ============================================================
def get_vnindex_history(days=None):
    """Lấy VN-Index - thử nhiều mã có thể."""
    if days is None:
        days = config.HISTORY_DAYS

    for ticker in ["VNINDEX", "VN-INDEX", "VN30", "VN30INDEX"]:
        try:
            df = get_stock_history(ticker, days)
            if df is not None and len(df) >= 50:
                print(f"      ✓ VNINDEX ({ticker}) OK")
                return df
        except Exception:
            continue

    print("      ❌ Không lấy được VNINDEX")
    return None


def get_foreign_flow(symbol):
    return {"symbol": symbol, "foreign_net": 0}


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("🧪 TEST LOTUSMARKET")
    print("=" * 60)

    for sym in ["VCB", "HPG", "FPT"]:
        print(f"\n--- {sym} ---")
        df = get_stock_history(sym, 100)
        if df is not None:
            print(f"  ✓ {len(df)} phiên")
            print(f"  Phiên cuối: {df.iloc[-1].to_dict()}")
        else:
            print(f"  ❌ Fail")

    print("\n--- VNINDEX ---")
    vni = get_vnindex_history(100)
    if vni is not None:
        print(f"  ✓ VNINDEX: {vni.iloc[-1]['close']}")
