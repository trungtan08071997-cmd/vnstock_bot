# ============================================================
# 🇻🇳 BOT LỌC CHỨNG KHOÁN — CHỈ DÙNG API VNDIRECT
# ============================================================
# ✅ Nguồn duy nhất: API VNDIRECT (api-finfo.vndirect.com.vn)
# ✅ Không cần vnstock, không cần thư viện bên thứ ba
# ✅ Dữ liệu giá + P/E + P/B + EPS + Vốn hóa đều từ VNDIRECT
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

# === Ngưỡng lọc kỹ thuật ===
MIN_VOLUME_RATIO = 1.5
MIN_ADX = 25
RSI_LOW = 55
RSI_HIGH = 75
BREAKOUT_MAX_DISTANCE = 0.03
MIN_VOLUME_ABSOLUTE = 500_000

# === Ngưỡng lọc cơ bản ===
MIN_MARKET_CAP_BILLION = 500
PE_MAX = 35
PE_MIN = 5
PB_MAX = 5

# === VNINDEX lọc thị trường ===
MARKET_RSI_UP = 45
MARKET_DOWN_PCT_STOP = -0.02

# === ATR Dừng lỗ động ===
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.5

# === Khác ===
TOP_STOCKS_LIMIT = 8
LOOKBACK_DAYS = 260
BACKTEST_DAYS = 180
RISK_PERCENT = 2.5
REWARD_RATIO = 1.5

# ============================================================
# 🏭 PHÂN LOẠI NGÀNH & THAM SỐ THEO NGÀNH
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
# 📡 LỚP DỮ LIỆU CƠ BẢN — CHỈ DÙNG API VNDIRECT
# ============================================================
class FundamentalData:
    """Lấy P/E, P/B, EPS, Vốn hóa từ API VNDIRECT duy nhất"""

    @staticmethod
    def get_fundamental_data(ticker):
        """Gọi API VNDIRECT lấy thông tin cơ bản"""
        try:
            url = f"https://api-finfo.vndirect.com.vn/v4/stocks/detail?code={ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Origin": "https://finfo.vndirect.com.vn",
                "Referer": "https://finfo.vndirect.com.vn/"
            }
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            
            pe = float(data.get('pe', 0))
            pb = float(data.get('pb', 0))
            eps = float(data.get('basicEps', 0))
            market_cap = float(data.get('marketCap', 0)) / 1000000000  # Đổi sang tỷ đồng
            
            return {
                "pe": pe,
                "pb": pb,
                "eps": eps,
                "market_cap_billion": market_cap,
                "source": "vndirect",
                "skip_filter": False
            }
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Lỗi kết nối lấy cơ bản {ticker}: {e}")
        except (ValueError, TypeError) as e:
            print(f"⚠️ Dữ liệu cơ bản không hợp lệ {ticker}: {e}")
        
        # Trả về dữ liệu mặc định nếu lỗi
        print(f"⚠️ Không lấy được dữ liệu cơ bản {ticker} — bỏ qua lọc cơ bản")
        return {
            "pe": 0, "pb": 0, "eps": 0, "market_cap_billion": 0,
            "source": "none", "skip_filter": True
        }

    @staticmethod
    def check_fundamental_criteria(fund):
        """Kiểm tra điều kiện lọc cơ bản"""
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
    """Lấy dữ liệu VNINDEX"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:VNINDEX~date:gte:{start_date}~date:lte:{end_date}&size=500")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 60:
            print("⚠️ Dữ liệu VNINDEX không đủ")
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
    """Lấy dữ liệu giá OHLCV từ API VNDIRECT"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:{ticker}~date:gte:{start_date}~date:lte:{end_date}&size=500")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        resp = requests.get(url, timeout=30, headers=headers)
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
    print("🔍 BẮT ĐẦU LỌC — NGUỒN DỮ LIỆU: API VNDIRECT")
    print("📊 Giá + P/E + P/B + EPS + Vốn hóa — Tất cả từ VNDIRECT")
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
# 📡 LỚP GỬI THÔNG BÁO DISCORD
# ============================================================
class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url and webhook_url.startswith("http"))
        if not self.enabled:
            print("⚠️ Chưa cấu hình DISCORD_WEBHOOK_URL — thông báo sẽ bị tắt")

    def send_embed(self, title, description, color=0x3498db):
        if not self.enabled:
            print(f"[THÔNG BÁO] {title}\n{description}\n")
            return False
        embed = {
            "title": title,
            "description": description + f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "color": color,
            "footer": {"text": "📢 Bot VNDIRECT | GĐ1: Cơ bản | GĐ2: ATR+Phân kỳ | Timeframe: D1"}
        }
        try:
            resp = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            return resp.status_code == 204
        except Exception as e:
            print(f"⚠️ Lỗi Discord: {e}")
            return False

    def market_summary(self, vnindex_info):
        status_emoji = "🟢" if vnindex_info['trend'] == "TĂNG" else ("🟡" if vnindex_info['trend'] == "NGANG" else "🔴")
        desc = f"""
{status_emoji} **VNINDEX:** {vnindex_info['close']:.2f}
📊 Xu hướng: {vnindex_info['trend']} | RSI: {vnindex_info['rsi']:.1f}
📉 Thay đổi 3 phiên: {vnindex_info['pct_3d']:+.2f}%
💡 Nhận định: {vnindex_info['note']}
        """
        return self.send_embed("📊 TỔNG QUAN THỊ TRƯỜNG — VNINDEX", desc, 0x3498db)

    def new_signal(self, ticker, sector, entry, tp, sl, atr_sl, score, ind):
        emoji = "🟢" if score >= 60 else ("🟡" if score >= 45 else "⏳")
        title = f"{emoji} TÍN HIỆU — {ticker} | {sector}"
        divergence = ind.get('divergence_note', 'Không có phân kỳ')
        desc = f"""
📌 **Mã:** {ticker} | 🏭 Ngành: {sector}
📊 **Điểm:** {score}/100 | ⏱️ Khung: D1 (Ngày)
💰 **Giá vào:** {entry}
🎯 **TP cố định:** {tp} | 🎯 **TP(ATR):** {ind.get('atr_tp', tp)}
🛑 **SL cố định:** {sl} | 🛑 **SL(ATR):** {atr_sl}
📈 **RSI:** {ind['rsi']} | **ADX:** {ind['adx']} | **ATR%:** {ind['atr_pct']:.2f}%
📉 **MACD:** {ind['macd']}/{ind['macd_signal']} | **Vol:** {ind['vol_ratio']}x
🔍 **P/E:** {ind['pe']:.1f} | **P/B:** {ind['pb']:.2f} | **EPS:** {'+' if ind['eps_positive'] else '-'}
📊 **Phân kỳ:** {divergence}
🆚 **RS{RS_PERIOD}:** {ind['rs_vs_market']:+.2%}
📡 **Nguồn dữ liệu:** VNDIRECT
        """
        return self.send_embed(title, desc, 0x2ecc71 if score >= 60 else 0xf39c12)

    def watchlist_update(self, count, stocks_list):

desc = f"""
✅ Đã cập nhật danh sách theo dõi — {count} mã
{stocks_list}
⏰ Quét lại lúc 8:00 sáng mai
        """
        return self.send_embed("📋 DANH SÁCH THEO DÕI ĐÃ CẬP NHẬT", desc, 0x9b59b6)

    def top_stocks(self, top_stocks, vnindex_data):
        list_text = ""
        for rank, s in enumerate(top_stocks, 1):
            signal_icon = "✅" if "MUA" in s['signal'] else "⏳"
            list_text += f"#{rank} {signal_icon} **{s['ticker']}** — {s['score']}/100 — {s['signal']}\n"
        
        market_note = ""
        if vnindex_data:
            market_note = f"\n📊 Thị trường: {vnindex_data['trend']} | RSI: {vnindex_data['rsi']:.1f}"
        
        desc = f"""
🏆 Top {len(top_stocks)} mã có điểm cao nhất hôm nay:

{list_text}
{market_note}
📡 Nguồn dữ liệu: API VNDIRECT
⏰ Chạy mỗi sáng 8:00 T2-T6
⚠️ **Không phải lời khuyên đầu tư — chỉ tham khảo phân tích kỹ thuật**
        """
        return self.send_embed("🏆 TOP MÃ LỌC — " + datetime.now().strftime("%d/%m/%Y"), desc, 0xf1c40f)

    def backtest_report(self, report):
        if not report:
            return False
        win_rate = report.get('win_rate', 0) * 100
        total_profit = report.get('total_profit_pct', 0)
        emoji = "🟢" if total_profit > 0 else "🔴"
        desc = f"""
{emoji} **BÁO CÁO BACKTEST — {report.get('period_days', 0)} ngày lịch sử**

📊 Tổng tín hiệu: {report.get('total_signals', 0)}
✅ Tỷ lệ thắng: {win_rate:.1f}%
💰 Lợi nhuận trung bình khi đúng: {report.get('avg_profit_pct', 0):+.2f}%
📉 Lỗ trung bình khi sai: {report.get('avg_loss_pct', 0):.2f}%
⚖️ Hệ số L/R: {report.get('lr_ratio_real', 0):.2f}
💵 Tổng lợi nhuận kỳ: {total_profit:+.2f}%
        """
        return self.send_embed("📊 BÁO CÁO BACKTEST", desc, 0x1abc9c if total_profit > 0 else 0xe74c3c)

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

        # So sánh VNINDEX — Relative Strength
        if vnindex_df is not None:
            df = df.merge(vnindex_df[['date','vnindex_close']], on='date', how='left')
            df['ret_20'] = df['close'].pct_change(RS_PERIOD)
            df['vnindex_ret_20'] = df['vnindex_close'].pct_change(RS_PERIOD)
            df['rs_vs_market'] = df['ret_20'] - df['vnindex_ret_20']

        return df

    def detect_divergence(self, df):
        """Phát hiện phân kỳ giá-khối lượng đơn giản"""
        if len(df) < 20:
            return False, "Dữ liệu không đủ"
        
        recent = df.tail(10)
        price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
        vol_trend = recent['volume'].iloc[-1] - recent['avg_vol_20'].mean()
        
        if price_trend > 0 and vol_trend < -0.1 * recent['avg_vol_20'].mean():
            return True, "⚠️ Phân kỳ âm: giá tăng nhưng khối lượng giảm"
        elif price_trend < 0 and vol_trend > 0.2 * recent['avg_vol_20'].mean():
            return True, "✅ Phân kỳ dương: giá giảm nhưng khối lượng tích lũy"
        return False, "Không có phân kỳ đáng kể"

    def calculate_score(self, df, sector_params, market_ok=True, fundamental_ok=True):
        """Tính điểm 100 theo ngành"""
        latest = df.iloc[-1]
        score = 0
        checks = {}

        p = sector_params if sector_params else SECTOR_PARAMS["Khác"]
        rsi_low = p.get("rsi_low", RSI_LOW)
        rsi_high = p.get("rsi_high", RSI_HIGH)
        vol_min = p.get("vol_min", MIN_VOLUME_RATIO)
        adx_min = p.get("adx_min", MIN_ADX)

        # EMA Ribbon — 25 điểm
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

        # Khối lượng — 15 điểm
        vol_ratio = latest['vol_ratio']
        if vol_ratio >= vol_min * 1.3:
            score += 15
        elif vol_ratio >= vol_min:
            score += 10
        checks['vol_ratio'] = round(float(vol_ratio), 2)

        # ADX + DI+ — 15 điểm
        adx_ok = latest['adx'] > adx_min
        di_ok = latest['plus_di'] > latest['minus_di']
        if adx_ok and di_ok:
            score += 15
        elif adx_ok or di_ok:
            score += 7
        checks['adx_ok'] = adx_ok
        checks['di_ok'] = di_ok

        # MACD — 15 điểm
        if latest['macd'] > latest['macd_signal']:
            score += 15
        checks['macd_ok'] = latest['macd'] > latest['macd_signal']

        # RSI — 10 điểm
        rsi = latest['rsi']
        if rsi_low < rsi < rsi_high:
            score += 10
        elif (rsi_low - 5) < rsi <= rsi_low or rsi_high <= rsi < (rsi_high + 5):
            score += 5
        checks['rsi_ok'] = rsi_low < rsi < rsi_high

        # Breakout — 10 điểm
        pct_from_high = latest['pct_from_high_20d']
        if pct_from_high <= BREAKOUT_MAX_DISTANCE:
            score += 10
        elif pct_from_high <= 0.05:
            score += 5
        checks['breakout_pct'] = round(float(pct_from_high * 100), 2)

        # Mạnh hơn VNINDEX — +5 điểm
        rs_strength = latest.get('rs_vs_market', 0)
        if not pd.isna(rs_strength) and rs_strength > 0:
            score += 5
        checks['rs_vs_market'] = round(float(rs_strength), 4) if not pd.isna(rs_strength) else 0

        # Cơ bản — +5 điểm
        if fundamental_ok:
            score += 5
        checks['fundamental_ok'] = fundamental_ok

        if not market_ok:
            score = int(score * 0.7)

        return score, checks

    def generate_signal(self, df, ticker, sector, sector_params, fundamental_data, vnindex_df=None, market_ok=True):
        df = self.calculate_all_indicators(df, sector_params, vnindex_df)
        latest = df.iloc[-1]
        entry_price = round(float(latest['close']), 2)

        fundamental_ok, fundamental_note = FundamentalData.check_fundamental_criteria(fundamental_data)
        score, checks = self.calculate_score(df, sector_params, market_ok, fundamental_ok)
        has_div, div_note = self.detect_divergence(df)

        # ATR Dừng lỗ động
        atr_value = float(latest['atr'])
        atr_pct = float(latest['atr_pct'])
        p = sector_params if sector_params else SECTOR_PARAMS["Khác"]
        atr_mult_sl = p.get("atr_mult", ATR_MULTIPLIER_SL)
        
        atr_sl = round(entry_price - atr_value * atr_mult_sl, 2)
        atr_tp = round(entry_price + atr_value * ATR_MULTIPLIER_TP, 2)
        
        # SL/TP cố định dự phòng
        stop_loss = round(entry_price * (1 - RISK_PERCENT/100), 2)
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
# 📈 BACKTEST
# ============================================================
class BacktestEngine:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run_backtest(self, ticker, sector, sector_params, vnindex_df=None, days=BACKTEST_DAYS):
        df = get_stock_data(ticker, days + 60)
        if df is None or len(df) < days:
            return None

        trades = []
        in_position = False
        entry_price = entry_date = None

        for i in range(60, len(df)):
            historical_df = df.iloc[:i+1].copy()
            current_date = historical_df['date'].iloc[-1]
            current_price = historical_df['close'].iloc[-1]
            
            fund = FundamentalData.get_fundamental_data(ticker)
            fundamental_ok, _ = FundamentalData.check_fundamental_criteria(fund)
            
            market_ok = True
            if vnindex_df is not None:
                vni_slice = vnindex_df[vnindex_df['date'] <= current_date]
                if len(vni_slice) > 0:
                    vni_ema20 = vni_slice['vnindex_close'].ewm(span=20, adjust=False).mean().iloc[-1]
                    market_ok = vni_slice['vnindex_close'].iloc[-1] > vni_ema20

            signal_data = self.analyzer.generate_signal(
                historical_df, ticker, sector, sector_params, fund, vnindex_df, market_ok
            )
            
            if signal_data['signal'] == "MUA ✅" and not in_position:
                in_position = True
                entry_price = current_price
                entry_date = current_date
            elif in_position:
                if (current_price <= signal_data['atr_sl'] or 
                    current_price >= signal_data['atr_tp'] or 
                    signal_data['score'] < 40):
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    trades.append({
                        "entry_date": entry_date, "exit_date": current_date,
                        "entry": entry_price, "exit": current_price,
                        "profit_pct": profit_pct, "win": profit_pct > 0
                    })
                    in_position = False
        
        if in_position and entry_price:
            profit_pct = (df['close'].iloc[-1] - entry_price) / entry_price * 100
            trades.append({
                "entry_date": entry_date, "exit_date": df['date'].iloc[-1],
                "entry": entry_price, "exit": df['close'].iloc[-1],
                "profit_pct": profit_pct, "win": profit_pct > 0
            })

        if not trades:
            return None

        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        return {
            "ticker": ticker,
            "sector": sector,
            "period_days": days,
            "total_trades": len(trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(trades),
            "avg_profit_pct": np.mean([t['profit_pct'] for t in wins]) if wins else 0,
            "avg_loss_pct": abs(np.mean([t['profit_pct'] for t in losses])) if losses else 0,
            "lr_ratio_real": (np.mean([t['profit_pct'] for t in wins]) if wins else 0) / 
                            (abs(np.mean([t['profit_pct'] for t in losses])) if losses else 1),

            "total_profit_pct": sum([t['profit_pct'] for t in trades]),
            "trades": trades
        }

    def run_backtest_all(self, tickers, vnindex_df=None):
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
        
        total_trades = sum(r['total_trades'] for r in all_reports)
        total_wins = sum(r['win_count'] for r in all_reports)
        total_profit = sum(r['total_profit_pct'] for r in all_reports)
        avg_win_rate = np.mean([r['win_rate'] for r in all_reports])
        
        return {
            "period_days": BACKTEST_DAYS,
            "total_signals": total_trades,
            "win_rate": avg_win_rate,
            "avg_profit_pct": np.mean([r['avg_profit_pct'] for r in all_reports if r['avg_profit_pct'] != 0]) or 0,
            "avg_loss_pct": np.mean([r['avg_loss_pct'] for r in all_reports if r['avg_loss_pct'] != 0]) or 0,
            "lr_ratio_real": (np.mean([r['avg_profit_pct'] for r in all_reports if r['avg_profit_pct'] != 0]) or 0) / 
                            ((np.mean([r['avg_loss_pct'] for r in all_reports if r['avg_loss_pct'] != 0]) or 1)),
            "total_profit_pct": total_profit,
            "reports": all_reports
        }

# ============================================================
# 🚀 HỆ THỐNG CHÍNH
# ============================================================
class SmartStockBot:
    def __init__(self, run_backtest=False):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.backtest_engine = BacktestEngine(self.analyzer)
        self.watchlist = []
        self.vnindex_data = None
        self.run_backtest = run_backtest

    def build_watchlist(self):
        print("\n📊 Đang phân tích VNINDEX...")
        self.vnindex_data = get_vnindex_data()
        market_ok = True
        if self.vnindex_data:
            print(f"   VNINDEX: {self.vnindex_data['close']:.2f} | Xu hướng: {self.vnindex_data['trend']} | RSI: {self.vnindex_data['rsi']:.1f}")
            print(f"   Nhận định: {self.vnindex_data['note']}")
            self.notifier.market_summary(self.vnindex_data)
            market_ok = self.vnindex_data['market_ok']
            if not market_ok:
                print("⚠️ Thị trường chưa thuận lợi — Điểm sẽ giảm 30% và ưu tiên thấp hơn")
        else:
            print("⚠️ Không lấy được VNINDEX — bỏ qua lọc thị trường")

        candidates = get_filtered_hose_stocks()
        results = []
        print(f"\n⏳ Đang phân tích {len(candidates)} mã...")
        print(f"📋 Ngưỡng: Vol>{MIN_VOLUME_RATIO}x | ADX>{MIN_ADX} | RSI theo ngành | PE {PE_MIN}-{PE_MAX} | PB<{PB_MAX}")
        print("-" * 80)

        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None

        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i:2d}/{len(candidates)}] {ticker:<6s}...", end=" ", flush=True)
            
            df = get_stock_data(ticker)
            if df is None:
                print("Bỏ qua (không có dữ liệu giá)")
                continue
            
            sector = STOCK_SECTOR.get(ticker, "Khác")
            sector_params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
            fund = FundamentalData.get_fundamental_data(ticker)
            print(f"P/E: {fund['pe']:.1f} | Vốn hóa: {fund['market_cap_billion']:.0f} tỷ", end=" | ")
            
            analysis = self.analyzer.generate_signal(df, ticker, sector, sector_params, fund, vnindex_df, market_ok)
            results.append(analysis)
            print(f"Điểm: {analysis['score']:3d}/100 | {analysis['signal']} | RS: {analysis['rs_vs_market']:+.2%}")
            time.sleep(0.5)

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        print("\n" + "=" * 80)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ ĐIỂM CAO NHẤT — Nguồn dữ liệu: VNDIRECT")
        print("=" * 80)
        for rank, s in enumerate(top_stocks, 1):
            print(f"#{rank} {s['ticker']:<6s} [{s['sector'][:12]}] | {s['score']:3d}/100 | {s['signal']}")
            print(f"       Giá: {s['entry']} | TP: {s['TP']} | SL(ATR): {s['atr_sl']}")
            print(f"       RSI: {s['rsi']} | ADX: {s['adx']} | P/E: {s['pe']:.1f} | Vol: {s['vol_ratio']}x | RS: {s['rs_vs_market']:+.2%}")
            print(f"       Phân kỳ: {s['divergence_note']}")
        print("=" * 80 + "\n")

        self.notifier.top_stocks(top_stocks, self.vnindex_data)
        
        for s in top_stocks:
            if s['signal'] == "MUA ✅":
                self.notifier.new_signal(
                    s['ticker'], s['sector'], s['entry'],
                    s['TP'], s['SL'], s['atr_sl'], s['score'], s
                )
        
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
    print("🚀 BOT LỌC CHỨNG KHOÁN — NGUỒN DỮ LIỆU: API VNDIRECT")
    print("⏱️ Timeframe: D1 (Ngày) | Chạy: 8:00 sáng Thứ 2 → Thứ 6")
    print("📊 GĐ1: Cơ bản | GĐ2: ATR+Phân kỳ+Tối ưu theo ngành | GĐ3: Backtest")
    print("=" * 80)
    print(f"⚙️ EMA: {EMA_SHORT}/{EMA_MID}/{EMA_LONG} | RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL} | ADX: {ADX_PERIOD}")
    print(f"⚙️ Lọc cơ bản: P/E {PE_MIN}-{PE_MAX} | P/B < {PB_MAX} | Vốn hóa ≥ {MIN_MARKET_CAP_BILLION:,} tỷ")
    print(f"⚙️ ATR SL/TP: Động theo biến động từng ngành | Phân kỳ: Giá-Khối lượng")
    print(f"⚙️ Backtest: {BACKTEST_DAYS} ngày lịch sử")
    print(f"⏰ Chạy: 8:00 sáng Thứ 2 → Thứ 6")
    print("=" * 80 + "\n")

    bot = SmartStockBot(run_backtest=False)

    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — NGUỒN DỮ LIỆU: API VNDIRECT",
        f"📊 Nguồn giá + cơ bản: api-finfo.vndirect.com.vn\n📈 Chỉ báo: EMA Ribbon | Khối lượng | ADX+DI | MACD | RSI | Breakout\n🔧 Tối ưu: Tham số theo ngành | ATR dừng lỗ động | Phân kỳ giá-khối lượng\n📉 Backtest: Sẵn sàng\n⏰ Quét mỗi sáng 8:00 T2-T6\n⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ",
        0x2ecc71
    )

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

                             
