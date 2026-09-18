"""
Data fetcher - dùng API công khai VNDirect/TCBS (hoạt động trên GitHub Actions)
- Không phụ thuộc vnstock
- Có cache + fallback nhiều nguồn
"""
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import config

CACHE_DIR = "output/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

REQUEST_DELAY = 1.0
MAX_RETRIES = 2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


# ============================================================
# CACHE
# ============================================================
def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key, max_age_hours=6):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - mtime > timedelta(hours=max_age_hours):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(key, data):
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


# ============================================================
# 1. VNDIRECT - Lịch sử giá
# ============================================================
def _vndirect_history(symbol, days):
    """Lấy OHLCV từ VNDirect."""
    end = datetime.now()
    start = end - timedelta(days=days * 2)
    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
    params = {
        "q": (f"code:{symbol}~date:gte:{start.strftime('%Y-%m-%d')}"
              f"~date:lte:{end.strftime('%Y-%m-%d')}"),
        "size": days + 50,
        "page": 1,
        "sort": "date",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        records = data.get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records)
        rename_map = {
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close",
            "nmVolume": "volume", "volume": "volume",
        }
        df = df.rename(columns=rename_map)
        cols = ["time", "open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                return None
        df = df[cols].tail(days).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"      VNDirect fail: {str(e)[:80]}")
        return None


# ============================================================
# 2. TCBS - Lịch sử giá (fallback)
# ============================================================
def _tcbs_history(symbol, days):
    """Lấy OHLCV từ TCBS public API."""
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days * 2)).timestamp())
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": symbol, "type": "stock",
        "resolution": "D", "from": start_ts, "to": end_ts,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        bars = data.get("data", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        if "tradingDate" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["tradingDate"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"open": "open", "high": "high",
                                 "low": "low", "close": "close",
                                 "volume": "volume"})
        cols = ["time", "open", "high", "low", "close", "volume"]
        df = df[cols].tail(days).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"      TCBS fail: {str(e)[:80]}")
        return None


# ============================================================
# MAIN: get_stock_history
# ============================================================
def get_stock_history(symbol, days=None):
    """
    Lấy dữ liệu OHLCV. Thử VNDirect → TCBS.
    """
    if days is None:
        days = config.HISTORY_DAYS

    cache_key = f"hist_{symbol}_{days}"
    cached = _load_cache(cache_key, max_age_hours=6)
    if cached:
        return pd.DataFrame(cached)

    for func, name in [(_vndirect_history, "VNDirect"),
                        (_tcbs_history, "TCBS")]:
        for attempt in range(MAX_RETRIES):
            df = func(symbol, days)
            if df is not None and len(df) >= 50:
                _save_cache(cache_key, df.to_dict("records"))
                print(f"      ✓ {name} OK ({len(df)} phiên)")
                time.sleep(REQUEST_DELAY)
                return df
            time.sleep(0.5)

    print(f"      ❌ Không lấy được giá {symbol}")
    return None


# ============================================================
# 3. VNDIRECT - Chỉ số cơ bản
# ============================================================
def _vndirect_ratios(symbol):
    """Lấy P/E, P/B, ROE từ VNDirect."""
    url = "https://finfo-api.vndirect.com.vn/v4/ratios/latest"
    params = {
        "filter": f"code:{symbol}",
        "order": "reportDate",
        "fields": ("code,reportDate,priceToEarnings,priceToBook,"
                   "roe,roa,netProfitMargin,eps"),
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        latest = data[0]
        return {
            "symbol": symbol,
            "pe": _safe_float(latest.get("priceToEarnings")),
            "pb": _safe_float(latest.get("priceToBook")),
            "roe": _safe_float(latest.get("roe")),
            "roa": _safe_float(latest.get("roa")),
            "eps": _safe_float(latest.get("eps")),
            "net_margin": _safe_float(latest.get("netProfitMargin")),
            "source": "vndirect",
        }
    except Exception as e:
        print(f"      VNDirect ratio fail: {str(e)[:80]}")
        return None


def get_financial_ratios(symbol):
    """Lấy chỉ số cơ bản."""
    cache_key = f"ratio_{symbol}"
    cached = _load_cache(cache_key, max_age_hours=24 * 7)
    if cached:
        return cached

    result = _vndirect_ratios(symbol)

    if result is None:
        result = {
            "symbol": symbol,
            "pe": None, "pb": None, "roe": None, "eps": None,
            "roa": None, "net_margin": None,
            "source": "unavailable",
        }

    _save_cache(cache_key, result)
    time.sleep(REQUEST_DELAY)
    return result


def _safe_float(val):
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


# ============================================================
# VNINDEX
# ============================================================
def get_vnindex_history(days=None):
    """Lấy VN-Index."""
    if days is None:
        days = config.HISTORY_DAYS
    # VNDirect dùng mã VNINDEX
    for sym in ["VNINDEX", "VN-INDEX"]:
        df = get_stock_history(sym, days)
        if df is not None:
            return df
    return None


def get_foreign_flow(symbol):
    return {"symbol": symbol, "foreign_net": 0}


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("🧪 TEST")
    df = get_stock_history("VCB", 100)
    if df is not None:
        print(f"VCB: {len(df)} phiên, close={df.iloc[-1]['close']}")
    r = get_financial_ratios("VCB")
    print(f"Ratios: {r}")
    vni = get_vnindex_history(100)
    if vni is not None:
        print(f"VNINDEX: close={vni.iloc[-1]['close']}")
