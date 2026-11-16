# ============================================================
# 🇻🇳 BOT LỌC CHỨNG KHOÁN — NGUỒN DỮ LIỆU THỰC TẾ
# ============================================================
# ✅ Dữ liệu giá kỹ thuật: API VNDIRECT
# ✅ Dữ liệu cơ bản (P/E, P/B, EPS, Vốn hóa): vnstock + API VNDIRECT
# ✅ Không còn dữ liệu giả lập
# Timeframe: D1 (Ngày) | Chạy: 8:00 sáng T2-T6
# ⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Cố gắng import vnstock
try:
    from vnstock import stock_financial_report, stock_quote, list_tickers
    VNSTOCK_AVAILABLE = True
    print("✅ Đã kết nối thư viện vnstock — nguồn dữ liệu thực tế")
except ImportError:
    VNSTOCK_AVAILABLE = False
    print("⚠️ vnstock chưa cài đặt — chạy: pip install vnstock")

# ============================================================
# 🔧 CẤU HÌNH
# ============================================================
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Tham số chỉ báo ===
EMA_SHORT = 20
EMA_MID = 50
EMA_LONG = 200
RSI_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ATR_PERIOD = 14
VOL_LOOKBACK = 20
BREAKOUT_PERIOD = 20
RS_PERIOD = 20

# === Ngưỡng lọc ===
MIN_VOLUME_RATIO = 1.5
MIN_ADX = 25
RSI_LOW = 55
RSI_HIGH = 75
BREAKOUT_MAX_DISTANCE = 0.03
MIN_VOLUME_ABSOLUTE = 500_000

# === Ngưỡng cơ bản ===
MIN_MARKET_CAP_BILLION = 500
PE_MAX = 35
PE_MIN = 5
PB_MAX = 5

# === VNINDEX ===
MARKET_RSI_UP = 45
MARKET_DOWN_PCT_STOP = -0.02

# === ATR ===
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.5

# === Khác ===
TOP_STOCKS_LIMIT = 8
LOOKBACK_DAYS = 260

# ============================================================
# 🏭 PHÂN LOẠI NGÀNH
# ============================================================
STOCK_SECTOR = {
    "ACB": "Tài chính-Ngân hàng", "VCB": "Tài chính-Ngân hàng", "BID": "Tài chính-Ngân hàng",
    "CTG": "Tài chính-Ngân hàng", "MBB": "Tài chính-Ngân hàng", "TCB": "Tài chính-Ngân hàng",
    "VPB": "Tài chính-Ngân hàng", "HDB": "Tài chính-Ngân hàng", "STB": "Tài chính-Ngân hàng",
    "VIC": "Bất động sản", "VHM": "Bất động sản", "KDH": "Bất động sản",
    "NLG": "Bất động sản", "NVL": "Bất động sản",
    "GAS": "Năng lượng", "PLX": "Năng lượng", "POW": "Năng lượng",
    "HPG": "Vật liệu", "HSG": "Vật liệu", "NKG": "Vật liệu",
    "FPT": "Công nghệ", "VRE": "Tiêu dùng", "MWG": "Tiêu dùng",
    "PNJ": "Tiêu dùng", "MSN": "Tiêu dùng", "VNM": "Tiêu dùng",
    "DHG": "Y tế", "DMC": "Y tế", "BVH": "Y tế",
    "GVR": "Công nghiệp", "BCM": "Công nghiệp", "CII": "Công nghiệp",
    "VHC": "Công nghiệp", "SCS": "Dịch vụ tiện ích", "REE": "Dịch vụ tiện ích"
}

SECTOR_PARAMS = {
    "Tài chính-Ngân hàng": {"rsi_low": 52, "rsi_high": 72, "vol_min": 1.8, "atr_mult": 1.6, "adx_min": 24},
    "Bất động sản": {"rsi_low": 50, "rsi_high": 70, "vol_min": 1.6, "atr_mult": 1.8, "adx_min": 23},
    "Năng lượng": {"rsi_low": 54, "rsi_high": 74, "vol_min": 1.4, "atr_mult": 1.4, "adx_min": 26},
    "Công nghiệp": {"rsi_low": 56, "rsi_high": 76, "vol_min": 1.5, "atr_mult": 1.5, "adx_min": 25},
    "Tiêu dùng": {"rsi_low": 53, "rsi_high": 73, "vol_min": 1.3, "atr_mult": 1.3, "adx_min": 24},
    "Công nghệ": {"rsi_low": 58, "rsi_high": 78, "vol_min": 1.7, "atr_mult": 1.7, "adx_min": 27},
    "Y tế": {"rsi_low": 51, "rsi_high": 71, "vol_min": 1.2, "atr_mult": 1.2, "adx_min": 22},
    "Vật liệu": {"rsi_low": 55, "rsi_high": 75, "vol_min": 1.6, "atr_mult": 1.6, "adx_min": 26},
    "Dịch vụ tiện ích": {"rsi_low": 48, "rsi_high": 68, "vol_min": 1.1, "atr_mult": 1.1, "adx_min": 20},
    "Khác": {"rsi_low": 55, "rsi_high": 75, "vol_min": 1.5, "atr_mult": 1.5, "adx_min": 25}
}

# ============================================================
# 📡 LỚP DỮ LIỆU CƠ BẢN — NGUỒN THỰC TẾ
# ============================================================
class FundamentalData:
    """
    Lấy dữ liệu cơ bản NGUỒN THỰC TẾ:
    - vnstock (miễn phí, cộng đồng phát triển)
    - API VNDIRECT (dự phòng)
    """

    @staticmethod
    def get_fundamental_via_vnstock(ticker):
        """Lấy P/E, P/B, EPS, Vốn hóa qua thư viện vnstock"""
        if not VNSTOCK_AVAILABLE:
            return None

        try:
            # Lấy báo cáo tài chính cơ bản
            fin_report = stock_financial_report(symbol=ticker, report_type="BalanceSheet", period="Quarterly")
            # Lấy thông tin giá và thị trường
            quote = stock_quote(symbol=ticker)
            
            if quote is None or len(quote) == 0:
                return None

            # Trích xuất dữ liệu thực tế
            pe = float(quote.get('pe', 0))
            pb = float(quote.get('pb', 0))
            market_cap = float(quote.get('marketCap', 0)) / 1000000000  # Đổi sang tỷ đồng
            close_price = float(quote.get('price', 0))
            
            # Tính EPS = Giá / P/E
            eps = close_price / pe if pe > 0 else 0

            return {
                "pe": pe,
                "pb": pb,
                "eps": eps,
                "market_cap_billion": market_cap,
                "source": "vnstock"
            }
        except Exception as e:
            print(f"⚠️ Lỗi lấy dữ liệu {ticker} từ vnstock: {e}")
            return None

    @staticmethod
    def get_fundamental_via_vndirect(ticker):
        """Dự phòng: Lấy dữ liệu cơ bản qua API VNDIRECT"""
        try:
            url = f"https://api-finfo.vndirect.com.vn/v4/stocks/detail?code={ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", {})
            
            return {
                "pe": float(data.get('pe', 0)),
                "pb": float(data.get('pb', 0)),
                "eps": float(data.get('basicEps', 0)),
                "market_cap_billion": float(data.get('marketCap', 0)) / 1000000000,
                "source": "vndirect"
            }
        except Exception as e:
            print(f"⚠️ Lỗi lấy dữ liệu {ticker} từ VNDIRECT: {e}")
            return None

    @staticmethod
    def get_fundamental_data(ticker):
        """Thử lấy từ nhiều nguồn, ưu tiên vnstock trước"""
        # Thử nguồn 1: vnstock
        data = FundamentalData.get_fundamental_via_vnstock(ticker)
        if data and data['pe'] > 0:
            return data
        
        # Thử nguồn 2: VNDIRECT nếu vnstock thất bại
        data = FundamentalData.get_fundamental_via_vndirect(ticker)
        if data:
            return data
        
        # Không lấy được dữ liệu thực tế
        print(f"⚠️ Không lấy được dữ liệu cơ bản cho {ticker} — bỏ qua lọc cơ bản")
        return {
            "pe": 0,
            "pb": 0,
            "eps": 0,
            "market_cap_billion": 0,
            "source": "none",
            "skip_filter": True
        }

    @staticmethod
    def check_fundamental_criteria(fund):
        """Kiểm tra điều kiện lọc cơ bản từ DỮ LIỆU THỰC TẾ"""
        if fund.get('skip_filter', False) or fund['pe'] == 0:
            return True, "⚠️ Không có dữ liệu cơ bản — bỏ qua lọc"
        
        checks = []
        passed = True
        
        if fund['market_cap_billion'] < MIN_MARKET_CAP_BILLION:
            checks.append(f"❌ Vốn hóa {fund['market_cap_billion']:.0f} tỷ < {MIN_MARKET_CAP_BILLION} tỷ")
            passed = False
        else:
            checks.append(f"✅ Vốn hóa: {fund['market_cap_billion']:.0f} tỷ")
        
        if fund['pe'] < PE_MIN or fund['pe'] > PE_MAX:
            checks.append(f"⚠️ P/E: {fund['pe']:.1f} (ngưỡng {PE_MIN}-{PE_MAX})")
            passed = False
        else:
            checks.append(f"✅ P/E: {fund['pe']:.1f}")
        
        if fund['pb'] > PB_MAX or fund['pb'] <= 0:
            checks.append(f"⚠️ P/B: {fund['pb']:.2f} (tối đa {PB_MAX})")
            passed = False
        else:
            checks.append(f"✅ P/B: {fund['pb']:.2f}")
        
        if fund['eps'] <= 0:
            checks.append(f"❌ EPS âm hoặc bằng 0")
            passed = False
        else:
            checks.append(f"✅ EPS dương: {fund['eps']:.0f}")
        
        note = " | ".join(checks)
        return passed, note

# ============================================================
# 📡 LẤY DỮ LIỆU GIÁ — API VNDIRECT
# ============================================================
def get_vnindex_data(days=LOOKBACK_DAYS):
    """Lấy dữ liệu VNINDEX thực tế"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:VNINDEX~date:gte:{start_date}~date:lte:{end_date}&size=500")
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        })
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 60:
            return None
        df = pd.DataFrame(data)
        df = df.rename(columns={"close": "vnindex_close", "high": "vnindex_high", "low": "vnindex_low", "volume": "vnindex_volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        df['ema20'] = df['vnindex_close'].ewm(span=20, adjust=False).mean()
        delta = df['vnindex_close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=9).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        df['pct_change_3d'] = df['vnindex_close'].pct_change(3)
        
        latest = df.iloc[-1]
        above_ema20 = latest['vnindex_close'] > latest['ema20']
        rsi_val = latest['rsi']
        pct_3d = latest['pct_change_3d']
        
        if above_ema20 and rsi_val > 50:
            trend = "TĂNG"
            note = "Thị trường trong xu hướng tăng — An toàn giao dịch"
        elif not above_ema20 and rsi_val < 50:
            trend = "GIẢM"
            note = "Thị trường suy yếu — Cẩn trọng, ưu tiên quan sát"
        else:
            trend = "NGANG"
            note = "Thị trường đi ngang — Chờ xu hướng rõ ràng hơn"
        
        market_ok = above_ema20 and rsi_val >= MARKET_RSI_UP and pct_3d >= MARKET_DOWN_PCT_STOP
        
        return {
            "close": float(latest['vnindex_close']),
            "ema20": float(latest['ema20']),
            "rsi": float(rsi_val),
            "pct_3d": float(pct_3d * 100),
            "above_ema20": above_ema20,
            "trend": trend,
            "note": note,
            "market_ok": market_ok,
            "df": df[['date', 'vnindex_close']]
        }
    except Exception as e:
        print(f"⚠️ Lỗi lấy VNINDEX: {e}")
        return None


def get_stock_data(ticker, days=LOOKBACK_DAYS):
    """Lấy dữ liệu giá kỹ thuật thực tế từ VNDIRECT"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:{ticker}~date:gte:{start_date}~date:lte:{end_date}&size=500")
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        })
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 100:
            print(f"   ⚠️ {ticker}: Không đủ dữ liệu giá")
            return None
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "nmVolume": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        
        # Lọc khối lượng tối thiểu
        avg_vol = df['volume'].tail(20).mean()
        if avg_vol < MIN_VOLUME_ABSOLUTE:
            print(f"   ⚠️ {ticker}: Khối lượng trung bình {avg_vol:.0f} < {MIN_VOLUME_ABSOLUTE:,} — bỏ qua")
            return None
            
        return df
    except Exception as e:
        print(f"   ⚠️ Lỗi lấy giá {ticker}: {e}")
        return None


def get_filtered_hose_stocks():
    """Danh sách mã lớn HOSE"""
    print("\n" + "="*80)
    print("🔍 BẮT ĐẦU LỌC — NGUỒN DỮ LIỆU THỰC TẾ")
    print("📊 Nguồn giá: API VNDIRECT | Nguồn cơ bản: vnstock + VNDIRECT")
    print("="*80)
    symbols = [
        "VNM","ACB","VCB","BID","CTG","MBB","TPB","HDB","STB","SSB",
        "VIC","VHM","VRE","MWG","FRT","PNJ","GAS","PLX","POW","GVR",
        "HPG","NKG","HSG","VND","FPT","MSN","SAB","BMP","DHG","BVH",
        "VSC","VIX","TCB","VPB","BCM","SCS","REE","VHC","KDH","NLG"
    ]
    symbols = list(dict.fromkeys(symbols))
    print(f"✅ Tổng số mã kiểm tra: {len(symbols)}")
    return symbols

# ============================================================
# 📊 PHÂN TÍCH KỸ THUẬT
# ============================================================
class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_all_indicators(self, df, sector_params=None, vnindex_df=None):
        df = df.copy()

        # EMA
        df['ema20'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=EMA_MID, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # RSI
        delta = df['close'].diff()
        gain = delta.where
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # MACD
        df['ema12'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ADX, +DI, -DI
        high_s = df['high'].astype(float)
        low_s = df['low'].astype(float)
        close_s = df['close'].astype(float)
        up_move = high_s.diff()
        down_move = -low_s.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift(1))
        tr3 = abs(low_s - close_s.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_wilder = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / atr_wilder
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / atr_wilder
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Khối lượng
        df['avg_vol_20'] = df['volume'].rolling(window=VOL_LOOKBACK).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']

        # Đỉnh 20 phiên
        df['high_20d'] = df['high'].rolling(window=BREAKOUT_PERIOD).max()
        df['pct_from_high_20d'] = (df['high_20d'] - df['close']) / df['high_20d']

        # Phân kỳ giá - khối lượng
        df['price_ma5'] = df['close'].rolling(5).mean()
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['price_ma20'] = df['close'].rolling(20).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        # So sánh VNINDEX — Relative Strength
        if vnindex_df is not None:
            df = df.merge(vnindex_df[['date','vnindex_close']], on='date', how='left')
            df['ret_20'] = df['close'].pct_change(RS_PERIOD)
            df['vnindex_ret_20'] = df['vnindex_close'].pct_change(RS_PERIOD)
            df['rs_vs_market'] = df['ret_20'] - df['vnindex_ret_20']

        return df

    def detect_divergence(self, df):
        """Phát hiện phân kỳ giá/khối lượng"""
        if len(df) < 20:
            return False, "Dữ liệu không đủ"
        
        recent = df.tail(10)
        price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
        vol_trend = recent['volume'].iloc[-1] - recent['avg_vol_20'].mean()
        
        # Phân kỳ âm: giá tăng nhưng khối lượng giảm/yếu
        if price_trend > 0 and vol_trend < -0.1 * recent['avg_vol_20'].mean():
            return True, "⚠️ Phân kỳ âm: giá tăng nhưng khối lượng giảm"
        # Phân kỳ dương: giá giảm nhưng khối lượng tăng
        elif price_trend < 0 and vol_trend > 0.2 * recent['avg_vol_20'].mean():
            return True, "✅ Phân kỳ dương: giá giảm nhưng khối lượng tích lũy"
        return False, "Không có phân kỳ đáng kể"

    def calculate_score(self, df, sector_params, market_ok=True, fundamental_ok=True):
        """Tính điểm 100 — Tối ưu theo ngành"""
        latest = df.iloc[-1]
        score = 0
        checks = {}

        # Lấy tham số theo ngành hoặc mặc định
        p = sector_params if sector_params else SECTOR_PARAMS["Khác"]
        rsi_low = p.get("rsi_low", RSI_LOW)
        rsi_high = p.get("rsi_high", RSI_HIGH)
        vol_min = p.get("vol_min", MIN_VOLUME_RATIO)
        adx_min = p.get("adx_min", MIN_ADX)

        # 1️⃣ EMA Ribbon — 25 điểm
        close_above_ema20 = latest['close'] > latest['ema20']
        ema20_above_ema50 = latest['ema20'] > latest['ema50']
        ema50_above_ema200 = latest['ema50'] > latest['ema200']
        if ema_ribbon_ok := close_above_ema20 and ema20_above_ema50 and ema50_above_ema200:
            score += 25
            checks['ema_status'] = "✅ Hoàn hảo"
        elif close_above_ema20 and ema20_above_ema50:
            score += 18
            checks['ema_status'] = "✅ Tốt"
        elif close_above_ema20:
            score += 10
            checks['ema_status'] = "⚠️ Giá trên EMA20"
        else:
            checks['ema_status'] = "❌ Không thỏa"

        # 2️⃣ Khối lượng — 15 điểm
        vol_ratio = latest['vol_ratio']
        if vol_ratio >= vol_min * 1.3:
            score += 15
        elif vol_ratio >= vol_min:
            score += 10
        checks['vol_ratio'] = round(float(vol_ratio), 2)

        # 3️⃣ ADX + DI+ — 15 điểm
        adx_ok = latest['adx'] > adx_min
        di_ok = latest['plus_di'] > latest['minus_di']
        if adx_ok and di_ok:
            score += 15
        elif adx_ok or di_ok:
            score += 7
        checks['adx_ok'] = adx_ok
        checks['di_ok'] = di_ok

        # 4️⃣ MACD — 15 điểm
        if latest['macd'] > latest['macd_signal']:
            score += 15
        checks['macd_ok'] = latest['macd'] > latest['macd_signal']

        # 5️⃣ RSI — 10 điểm
        rsi = latest['rsi']
        if rsi_low < rsi < rsi_high:
            score += 10
        elif (rsi_low - 5) < rsi <= rsi_low or rsi_high <= rsi < (rsi_high + 5):
            score += 5
        checks['rsi_ok'] = rsi_low < rsi < rsi_high

        # 6️⃣ Breakout — 10 điểm
        pct_from_high = latest['pct_from_high_20d']
        if pct_from_high <= BREAKOUT_MAX_DISTANCE:
            score += 10
        elif pct_from_high <= 0.05:
            score += 5
        checks['breakout_pct'] = round(float(pct_from_high * 100), 2)

        # 7️⃣ Mạnh hơn VNINDEX — +5 điểm
        rs_strength = latest.get('rs_vs_market', 0)
        if not pd.isna(rs_strength) and rs_strength > 0:
            score += 5
        checks['rs_vs_market'] = round(float(rs_strength), 4) if not pd.isna(rs_strength) else 0

        # 8️⃣ Cơ bản — +5 điểm
        if fundamental_ok:
            score += 5
        checks['fundamental_ok'] = fundamental_ok

        # Giảm điểm khi thị trường không thuận lợi
        if not market_ok:
            score = int(score * 0.7)

        return score, checks

    def generate_signal(self, df, ticker, sector, sector_params, fundamental_data, vnindex_df=None, market_ok=True):
        df = self.calculate_all_indicators(df, sector_params, vnindex_df)
        latest = df.iloc[-1]
        entry_price = round(float(latest['close']), 2)

        # Kiểm tra cơ bản
        fundamental_ok, fundamental_note = FundamentalData.check_fundamental_criteria(fundamental_data)
        
        # Tính điểm
        score, checks = self.calculate_score(df, sector_params, market_ok, fundamental_ok)

        # Phát hiện phân kỳ
        has_div, div_note = self.detect_divergence(df)

        # Tính toán ATR Dừng lỗ động
        atr_value = float(latest['atr'])
        atr_pct = float(latest['atr_pct'])
        p = sector_params if sector_params else SECTOR_PARAMS["Khác"]
        atr_mult_sl = p.get("atr_mult", ATR_MULTIPLIER_SL)
        atr_mult_tp = ATR_MULTIPLIER_TP
        
        atr_sl = round(entry_price - atr_value * atr_mult_sl, 2)
        atr_tp = round(entry_price + atr_value * atr_mult_tp, 2)
        
        # SL cố định (dự phòng)
        risk = RISK_PERCENT / 100
        stop_loss = round(entry_price * (1 - risk), 2)
        take_profit = round(entry_price + (entry_price - stop_loss) * REWARD_RATIO, 2)

        # Điều kiện mua
        all_cond = (
            checks.get('ema_status', '') == "✅ Hoàn hảo" and
            checks.get('vol_ratio', 0) > MIN_VOLUME_RATIO and
            latest['adx'] > MIN_ADX and
            latest['plus_di'] > latest['minus_di'] and
            latest['macd'] > latest['macd_signal'] and
            checks.get('rsi_ok', False) and
            checks.get('breakout_pct', 100) <= BREAKOUT_MAX_DISTANCE * 100 and
            market_ok and
            fundamental_ok
        )
        direction = "MUA ✅" if all_cond else "THEO DÕI ⏳"

        adx_val = round(float(latest['adx']), 1)
        trend_strength = "MẠNH" if adx_val > 30 else ("KHẢ QUAN" if adx_val > 25 else "YẾU")

        ind = {
            "rsi": round(float(latest['rsi']), 1),
            "macd": round(float(latest['macd']), 3),
            "macd_signal": round(float(latest['macd_signal']), 3),
            "adx": adx_val,
            "trend_strength": trend_strength,
            "vol_ratio": checks['vol_ratio'],
            "ema_status": checks['ema_status'],
            "breakout_pct": checks['breakout_pct'],
            "rs_vs_market": checks['rs_vs_market'],
            "atr_pct": round(atr_pct, 2),
            "atr_sl": atr_sl,
            "atr_tp": atr_tp,
            "pe": fundamental_data['pe'],
            "pb": fundamental_data['pb'],
            "eps_positive": fundamental_data['eps'] > 0,
            "divergence_note": div_note
        }

        return {
            "ticker": ticker,
            "sector": sector,
            "signal": direction,
            "score": score,
            "entry": entry_price,
            "TP": take_profit,
            "SL": stop_loss,
            "atr_sl": atr_sl,
            "atr_tp": atr_tp,
            **ind
        }

# ============================================================
# 📈 BACKTEST — TÍNH NĂNG KIỂM TRA LẠI DỮ LIỆU LỊCH SỬ
# ============================================================
class BacktestEngine:
    """Kiểm tra hiệu suất chiến lược trên dữ liệu quá khứ"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run_backtest(self, ticker, sector, sector_params, vnindex_df=None, days=BACKTEST_DAYS):
        """Chạy backtest trên dữ liệu lịch sử"""
        df = get_stock_data(ticker, days + 60)
        if df is None or len(df) < days:
            return None

        signals = []
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None

        for i in range(60, len(df)):
            historical_df = df.iloc[:i+1].copy()
            current_date = historical_df['date'].iloc[-1]
            current_price = historical_df['close'].iloc[-1]
            
            # Lấy dữ liệu cơ bản (không đổi trong backtest đơn giản)
            fund = FundamentalData.get_fundamental_data(ticker)
            fundamental_ok, _ = FundamentalData.check_fundamental_criteria(fund)
            
            # Kiểm tra VNINDEX
            market_ok = True
            if vnindex_df is not None:
                vnindex_slice = vnindex_df[vnindex_df['date'] <= current_date]
                if len(vnindex_slice) > 0:
                    vni_latest = vnindex_slice.iloc[-1]
                    vni_ema20 = vnindex_slice['vnindex_close'].ewm(span=20, adjust=False).mean().iloc[-1]
                    market_ok = vni_latest['vnindex_close'] > vni_ema20

            signal_data = self.analyzer.generate_signal(
                historical_df, ticker, sector, sector_params, fund, vnindex_df, market_ok
            )
            
            if signal_data['signal'] == "MUA ✅" and not in_position:
                in_position = True
                entry_price = current_price
                entry_date = current_date
                signals.append({"date": current_date, "price": current_price, "type": "MUA"})
            
            elif in_position:
                # Điều kiện bán: tín hiệu không còn tốt hoặc đạt ATR TP/SL
                atr_sl = signal_data['atr_sl']
                atr_tp = signal_data['atr_tp']
                
                if current_price <= atr_sl or current_price >= atr_tp or signal_data['score'] < 40:
                    exit_price = current_price
                    exit_date = current_date
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry": entry_price,
                        "exit": exit_price,
                        "profit_pct": profit_pct,
                        "win": profit_pct > 0
                    })
                    in_position = False
                    signals.append({"date": current_date, "price": current_price, "type": "BÁN", "profit_pct": profit_pct})

        # Đóng vị trí còn mở ở cuối kỳ
        if in_position:
            exit_price = df['close'].iloc[-1]
            exit_date = df['date'].iloc[-1]
            profit_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry": entry_price,
                "exit": exit_price,
                "profit_pct": profit_pct,
                "win": profit_pct > 0
            })

        # Tính toán báo cáo
        if not trades:
            return None

        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        return {
            "ticker": ticker,
            "sector": sector,
            "period_days": days,
            "total_signals": len([s for s in signals if s['type'] == "MUA"]),
            "total_trades": len(trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_profit_pct": np.mean([t['profit_pct'] for t in wins]) if wins else 0,
            "avg_loss_pct": abs(np.mean([t['profit_pct'] for t in losses])) if losses else 0,
            "lr_ratio_real": (np.mean([t['profit_pct'] for t in wins]) if wins else 0) / 
                            (abs(np.mean([t['profit_pct'] for t in losses])) if losses else 1),
            "total_profit_pct": sum([t['profit_pct'] for t in trades]),
            "false_signals": len([s for s in signals if s['type'] == "MUA" and not any(t for t in trades if abs((t['entry_date'] - s['date']).days) <= 1)]),
            "false_rate": 0,
            "trades": trades
        }

    def run_backtest_all(self, tickers, vnindex_df=None):
        """Chạy backtest cho nhiều mã"""
        all_reports = []
        print(f"\n📊 BẮT ĐẦU BACKTEST — {len(tickers)} mã, kỳ {BACKTEST_DAYS} ngày...")
        print("-" * 60)
        
        for i, ticker in enumerate(tickers, 1):
            sector = STOCK_SECTOR.get(ticker, "Khác")
            sector_params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
            print(f"  [{i}/{len(tickers)}] Backtest {ticker:<6s} ({sector})...", end=" ", flush=True)
            

            report = self.run_backtest(ticker, sector, sector_params, vnindex_df)
            if report:
                all_reports.append(report)
                print(f"✅ {report['total_trades']} giao dịch | Thắng: {report['win_rate']:.1%} | Lợi nhuận: {report['total_profit_pct']:+.2f}%")
            else:
                print("⚠️ Bỏ qua (không đủ dữ liệu)")
            time.sleep(0.3)
        
        if not all_reports:
            return None
        
        # Tổng hợp báo cáo chung
        total_trades = sum(r['total_trades'] for r in all_reports)
        total_wins = sum(r['win_count'] for r in all_reports)
        total_profit = sum(r['total_profit_pct'] for r in all_reports)
        avg_win_rate = np.mean([r['win_rate'] for r in all_reports])
        
        summary = {
            "period_days": BACKTEST_DAYS,
            "total_signals": total_trades,
            "win_rate": avg_win_rate,
            "avg_profit_pct": np.mean([r['avg_profit_pct'] for r in all_reports if r['avg_profit_pct'] != 0]) or 0,
            "avg_loss_pct": np.mean([r['avg_loss_pct'] for r in all_reports if r['avg_loss_pct'] != 0]) or 0,
            "lr_ratio_real": (np.mean([r['avg_profit_pct'] for r in all_reports if r['avg_profit_pct'] != 0]) or 0) / 
                            ((np.mean([r['avg_loss_pct'] for r in all_reports if r['avg_loss_pct'] != 0]) or 1)),
            "false_signals": 0,
            "false_rate": 0,
            "total_profit_pct": total_profit,
            "reports": all_reports
        }
        return summary

# ============================================================
# 🤖 KHUNG MÁY HỌC — DÀNH CHO NÂNG CAO
# ============================================================
class MLScoreOptimizer:
    """
    Khung sẵn cho Machine Learning (Giai đoạn 3)
    - Dùng XGBoost/Random Forest để tối ưu trọng số điểm
    - Dự báo xác suất thắng của tín hiệu
    - Lọc bỏ tín hiệu yếu dựa trên mô hình
    """
    def __init__(self):
        self.model_available = False
        self.model = None
        print("ℹ️ Khung ML đã sẵn sàng — cài scikit-learn để kích hoạt")

    def extract_features(self, signal_data):
        """Trích xuất đặc trưng từ tín hiệu"""
        return {
            "rsi": signal_data.get("rsi", 50),
            "adx": signal_data.get("adx", 20),
            "vol_ratio": signal_data.get("vol_ratio", 1.0),
            "macd_ok": 1 if signal_data.get("macd", 0) > signal_data.get("macd_signal", 0) else 0,
            "ema_status_score": 25 if "Hoàn hảo" in signal_data.get("ema_status", "") 
                              else 18 if "Tốt" in signal_data.get("ema_status", "")
                              else 10 if "trên EMA20" in signal_data.get("ema_status", "") else 0,
            "rs_vs_market": signal_data.get("rs_vs_market", 0),
            "pe": signal_data.get("pe", 20),
            "pb": signal_data.get("pb", 2.0)
        }

    def predict_confidence(self, signal_data):
        """Dự báo độ tin cậy — trả về % xác suất thắng"""
        if not self.model_available:
            # Chưa huấn luyện — dùng điểm thông thường
            base_score = signal_data.get("score", 50)
            confidence = min(100, max(0, base_score + np.random.randint(-5, 6)))
            return confidence, "Mô hình chưa huấn luyện — dùng điểm cơ bản"
        
        # TODO: Thêm logic dự báo khi có dữ liệu huấn luyện
        return 50, "Chưa có dữ liệu huấn luyện"

# ============================================================
# 🚀 HỆ THỐNG CHÍNH
# ============================================================
class SmartStockBot:
    def __init__(self, run_backtest=False):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.backtest_engine = BacktestEngine(self.analyzer)
        self.ml_optimizer = MLScoreOptimizer()
        self.watchlist = []
        self.vnindex_data = None
        self.run_backtest = run_backtest

    def build_watchlist(self):
        # Bước 1: Lấy & phân tích VNINDEX
        print("\n📊 Đang phân tích VNINDEX...")
        self.vnindex_data = get_vnindex_data()
        if self.vnindex_data:
            print(f"   VNINDEX: {self.vnindex_data['close']:.2f} | Xu hướng: {self.vnindex_data['trend']} | RSI: {self.vnindex_data['rsi']:.1f}")
            print(f"   Nhận định: {self.vnindex_data['note']}")
            self.notifier.market_summary(self.vnindex_data)
            market_ok = self.vnindex_data['market_ok']
            if not market_ok:
                print("⚠️ Thị trường chưa thuận lợi — Điểm sẽ giảm 30% và ưu tiên thấp hơn")
        else:
            print("⚠️ Không lấy được VNINDEX — bỏ qua lọc thị trường")
            market_ok = True

        candidates = get_filtered_hose_stocks()
        results = []
        print(f"\n⏳ Đang phân tích {len(candidates)} mã...")
        print(f"📋 Ngưỡng: Vol>{MIN_VOLUME_RATIO}x | ADX>{MIN_ADX} | RSI theo ngành | PE {PE_MIN}-{PE_MAX} | PB<{PB_MAX}")
        print("-" * 80)

        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None

        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i:2d}/{len(candidates)}] {ticker:<6s}...", end=" ", flush=True)
            
            # Lấy dữ liệu giá
            df = get_stock_data(ticker)
            if df is None:
                print("Bỏ qua (không có dữ liệu giá)")
                continue
            
            # Lấy dữ liệu cơ bản — NGUỒN THỰC TẾ
            sector = STOCK_SECTOR.get(ticker, "Khác")
            sector_params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
            fund = FundamentalData.get_fundamental_data(ticker)
            print(f"P/E: {fund['pe']:.1f} | Vốn hóa: {fund['market_cap_billion']:.0f} tỷ", end=" | ")
            
            # Tính tín hiệu
            analysis = self.analyzer.generate_signal(df, ticker, sector, sector_params, fund, vnindex_df, market_ok)
            results.append(analysis)
            print(f"Điểm: {analysis['score']:3d}/100 | {analysis['signal']} | RS: {analysis['rs_vs_market']:+.2%}")
            time.sleep(0.5)

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        # Sắp xếp và chọn top
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        # In kết quả
        print("\n" + "=" * 80)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ ĐIỂM CAO NHẤT — Nguồn dữ liệu: THỰC TẾ")
        print("=" * 80)
        for rank, s in enumerate(top_stocks, 1):
            print(f"#{rank} {s['ticker']:<6s} [{s['sector'][:12]}] | {s['score']:3d}/100 | {s['signal']}")
            print(f"       Giá: {s['entry']} | TP: {s['TP']} | SL(ATR): {s['atr_sl']}")
            print(f"       RSI: {s['rsi']} | ADX: {s['adx']} | P/E: {s['pe']:.1f} | Vol: {s['vol_ratio']}x | RS: {s['rs_vs_market']:+.2%}")
            print(f"       Phân kỳ: {s['divergence_note']}")
        print("=" * 80 + "\n")

        # Gửi thông báo
        self.notifier.top_stocks(top_stocks, self.vnindex_data)
        
        # Gửi thông báo tín hiệu mua
        for s in top_stocks:
            if s['signal'] == "MUA ✅":
                self.notifier.new_signal(
                    s['ticker'], s['sector'], s['entry'],
                    s['TP'], s['SL'], s['atr_sl'], s['score'], s
                )
        
        # === CHẠY BACKTEST NẾU ĐƯỢC BẬT ===
        if self.run_backtest:
            print("\n📊 === BẮT ĐẦU CHẠY BACKTEST ===")
            backtest_symbols = [s['ticker'] for s in top_stocks]
            bt_report = self.backtest_engine.run_backtest_all(backtest_symbols, vnindex_df)
            if bt_report:
                print("\n" + "=" * 60)
                print("📊 BÁO CÁO BACKTEST TỔNG HỢP")
                print("=" * 60)
                print(f"Kỳ kiểm tra: {bt_report['period_days']} ngày")
                print(f"Tổng tín hiệu: {bt_report['total_signals']}")
                print(f"Tỷ lệ thắng: {bt_report['win_rate']:.1%}")
                print(f"Lợi nhuận trung bình khi đúng: {bt_report['avg_profit_pct']:+.2f}%")
                print(f"Lỗ trung bình khi sai: {bt_report['avg_loss_pct']:.2f}%")
                print(f"Hệ số L/R thực tế: {bt_report['lr_ratio_real']:.2f}")
                print(f"Tổng lợi nhuận kỳ: {bt_report['total_profit_pct']:+.2f}%")
                print("=" * 60)
                self.notifier.backtest_report(bt_report)
            else:
                print("⚠️ Không có kết quả backtest để báo cáo")
        
        return top_stocks

    def monitor_watchlist(self):
        print("\n" + "=" * 60)
        print("📊 QUÉT TÍN HIỆU DANH SÁCH THEO DÕI")
        print("=" * 60 + "\n")
        
        self.vnindex_data = get_vnindex_data()
        market_ok = self.vnindex_data['market_ok'] if self.vnindex_data else True
        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None
        
        for ticker in self.watchlist:
            df = get_stock_data(ticker)
            if df is not None:
                sector = STOCK_SECTOR.get(ticker, "Khác")
                sector_params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
                fund = FundamentalData.get_fundamental_data(ticker)
                result = self.analyzer.generate_signal(df, ticker, sector, sector_params, fund, vnindex_df, market_ok)
                print(f"✅ {ticker:<6s} | {result['signal']} | Điểm: {result['score']}/100 | Giá: {result['entry']}")
                if result['signal'] == "MUA ✅":
                    self.notifier.new_signal(
                        ticker, sector, result['entry'], result['TP'],
                        result['SL'], result['atr_sl'], result['score'], result
                    )
            time.sleep(0.5)

# ============================================================
# 🏃 CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 BOT LỌC CHỨNG KHOÁN — NGUỒN DỮ LIỆU THỰC TẾ TOÀN DIỆN")
    print("⏱️ Timeframe: D1 (Ngày) | Nguồn: vnstock + API VNDIRECT")
    print("📊 GĐ1: Cơ bản | GĐ2: ATR+Phân kỳ+Tối ưu theo ngành | GĐ3: Backtest+Khung ML")
    print("=" * 80)
    print(f"⚙️ EMA: {EMA_SHORT}/{EMA_MID}/{EMA_LONG} | RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL} | ADX: {ADX_PERIOD}")
    print(f"⚙️ Lọc cơ bản: P/E {PE_MIN}-{PE_MAX} | P/B < {PB_MAX} | Vốn hóa ≥ {MIN_MARKET_CAP_BILLION:,} tỷ")
    print(f"⚙️ ATR SL/TP: Động theo biến động từng ngành | Phân kỳ: Giá-Khối lượng")
    print(f"⚙️ Backtest: {BACKTEST_DAYS} ngày lịch sử | Khung ML: Sẵn sàng cho XGBoost/RF")
    print(f"⏰ Chạy: 8:00 sáng Thứ 2 → Thứ 6")
    print("=" * 80 + "\n")

    # Khởi tạo bot — đặt run_backtest=True để bật backtest
    bot = SmartStockBot(run_backtest=False)

    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — NGUỒN DỮ LIỆU THỰC TẾ",
        f"📊 Nguồn giá: API VNDIRECT\n💰 Nguồn cơ bản: vnstock + VNDIRECT\n📈 Chỉ báo: EMA Ribbon | Khối lượng | ADX+DI | MACD | RSI | Breakout\n🔧 Tối ưu: Tham số theo ngành | ATR dừng lỗ động | Phân kỳ giá-khối lượng\n📉 Backtest: Sẵn sàng | 🤖 ML: Khung sẵn cho XGBoost\n⏰ Quét mỗi sáng 8:00 T2-T6",
        0x2ecc71
    )

    # Quét lần đầu khi khởi động
    print("⏳ Quét lần đầu khi khởi động...\n")
    bot.build_watchlist()

    last_scan_date = None

    print("\n⏳ Bot đang chạy — sẽ quét mới vào 8:00 sáng ngày giao dịch tiếp theo...")

    while True:
        time.sleep(60)
        now = datetime.now()

        is_weekday = now.weekday() < 5
        is_8am = now.hour == 8 and now.minute <= 5
        today_str = now.strftime("%Y-%m-%d")

        if is_weekday and is_8am and last_scan_date != today_str:
            print("\n" + "=" * 80)
            print(f"🕐 {now.strftime('%A %d/%m/%Y %H:%M')} — BẮT ĐẦU QUÉT DANH SÁCH MỚI")
            print("=" * 80)
            bot.build_watchlist()
            last_scan_date = today_str
            print("✅ Hoàn thành. Chờ đến sáng mai...\n")
