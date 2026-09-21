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
   # ===== HARDCODE RATIOS VN30 (cập nhật 2026) =====
VN30_RATIOS = {
    "ACB": {"pe": 7.2, "pb": 1.4, "roe": 22.5},
    "BCM": {"pe": 25.8, "pb": 3.2, "roe": 12.5},
    "BID": {"pe": 13.5, "pb": 2.1, "roe": 16.8},
    "BVH": {"pe": 15.2, "pb": 1.8, "roe": 12.2},
    "CTG": {"pe": 9.8, "pb": 1.7, "roe": 18.5},
    "FPT": {"pe": 24.5, "pb": 5.8, "roe": 25.3},
    "GAS": {"pe": 18.2, "pb": 3.1, "roe": 17.5},
    "GVR": {"pe": 28.5, "pb": 2.9, "roe": 10.5},
    "HDB": {"pe": 6.8, "pb": 1.3, "roe": 21.2},
    "HPG": {"pe": 12.5, "pb": 1.6, "roe": 14.2},
    "MBB": {"pe": 6.5, "pb": 1.3, "roe": 21.8},
    "MSN": {"pe": 22.5, "pb": 2.8, "roe": 12.5},
    "MWG": {"pe": 18.5, "pb": 3.2, "roe": 18.5},
    "PLX": {"pe": 16.2, "pb": 2.5, "roe": 15.8},
    "POW": {"pe": 14.5, "pb": 1.2, "roe": 8.5},
    "SAB": {"pe": 17.8, "pb": 3.5, "roe": 20.2},
    "SHB": {"pe": 5.8, "pb": 1.1, "roe": 18.2},
    "SSB": {"pe": 8.5, "pb": 1.5, "roe": 17.5},
    "SSI": {"pe": 14.2, "pb": 2.1, "roe": 15.2},
    "STB": {"pe": 8.2, "pb": 1.4, "roe": 16.5},
    "TCB": {"pe": 7.5, "pb": 1.4, "roe": 19.5},
    "TPB": {"pe": 6.8, "pb": 1.2, "roe": 18.5},
    "VCB": {"pe": 15.2, "pb": 2.8, "roe": 17.5},
    "VHM": {"pe": 8.5, "pb": 1.5, "roe": 18.2},
    "VIB": {"pe": 6.5, "pb": 1.3, "roe": 20.5},
    "VIC": {"pe": 32.5, "pb": 2.1, "roe": 6.5},
    "VJC": {"pe": 28.5, "pb": 4.2, "roe": 15.2},
    "VNM": {"pe": 16.5, "pb": 3.8, "roe": 24.2},
    "VPB": {"pe": 8.2, "pb": 1.5, "roe": 19.2},
    "VRE": {"pe": 12.5, "pb": 1.8, "roe": 14.5},
}


def get_financial_ratios(symbol):
    """Lấy chỉ số cơ bản - dùng HARDCODE cho VN30."""
    if symbol in VN30_RATIOS:
        r = VN30_RATIOS[symbol].copy()
        r["symbol"] = symbol
        r["eps"] = None
        r["roa"] = None
        r["net_margin"] = None
        r["source"] = "hardcoded"
        return r
    
    # Không phải VN30 → trả rỗng
    return {
        "symbol": symbol,
        "pe": None, "pb": None, "roe": None, "eps": None,
        "roa": None, "net_margin": None,
        "source": "unavailable",
    }


def _empty_ratio(symbol):
    """Trả về dict rỗng khi không lấy được."""
    return {
        "symbol": symbol,
        "pe": None, "pb": None, "roe": None, "eps": None,
        "roa": None, "net_margin": None,
        "source": "unavailable",
    }


# ============================================================
# VNINDEX - Thử TCBS → VNDirect → None
# ============================================================
def _tcbs_vnindex(days):
    """Lấy VNINDEX từ TCBS (type=index)."""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta

    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days * 2)).timestamp())

    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": "VNINDEX", "type": "index",
        "resolution": "D", "from": start_ts, "to": end_ts,
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        bars = r.json().get("data", [])
        if not bars:
            return None

        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["tradingDate"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"open": "open", "high": "high",
                                 "low": "low", "close": "close",
                                 "volume": "volume"})
        required = ["time", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        df = df[required].tail(days).reset_index(drop=True)
        print(f"      ✓ TCBS VNINDEX OK ({len(df)} phiên)")
        return df
    except Exception as e:
        print(f"      TCBS VNINDEX fail: {str(e)[:80]}")
        return None


def _vndirect_vnindex(days):
    """Lấy VNINDEX từ VNDirect endpoint index_prices."""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=days * 2)

    url = "https://finfo-api.vndirect.com.vn/v4/index_prices"
    params = {
        "q": f"code:VNINDEX~date:gte:{start.strftime('%Y-%m-%d')}"
              f"~date:lte:{end.strftime('%Y-%m-%d')}",
        "size": days + 50, "page": 1, "sort": "date",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        records = r.json().get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close",
            "nmVolume": "volume", "volume": "volume",
        })
        required = ["time", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        df = df[required].tail(days).reset_index(drop=True)
        print(f"      ✓ VNDirect INDEX OK ({len(df)} phiên)")
        return df
    except Exception as e:
        print(f"      VNDirect INDEX fail: {str(e)[:80]}")
        return None


def get_vnindex_history(days=None):
    """
    Lấy VN-Index - thử nhiều nguồn.
    Nếu tất cả fail → trả None (bot vẫn chạy tiếp).
    """
    if days is None:
        days = config.HISTORY_DAYS

    print("      📊 Đang lấy VNINDEX...")

    # Nguồn 1: TCBS
    df = _tcbs_vnindex(days)
    if df is not None and len(df) >= 50:
        return df

    # Nguồn 2: VNDirect
    df = _vndirect_vnindex(days)
    if df is not None and len(df) >= 50:
        return df

    # Nguồn 3: Fallback - trả None, không báo lỗi
    print("      ⚠️ Không lấy được VNINDEX - bỏ qua phần thị trường")
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
