# ============================================================
# 🇻🇳 BOT LỌC CHỨNG KHOÁN — PHIÊN BẢN 2.0 HOÀN CHỈNH
# ✅ Chạy lần đầu không lỗi
# ✅ Dữ liệu từ API VNDIRECT
# ✅ Random Forest — nhẹ & ổn định
# ✅ Quản lý rủi ro, phân kỳ, đa khung thời gian
# ⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time
import json
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings('ignore')

# === Machine Learning — Chỉ dùng Random Forest, không cần XGBoost ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
XGB_AVAILABLE = False

# ============================================================
# 🔧 CẤU HÌNH
# ============================================================
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Tham số kỹ thuật ===
EMA_SHORT, EMA_MID, EMA_LONG = 20, 50, 200
RSI_PERIOD = 9
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ADX_PERIOD, ATR_PERIOD = 14, 14
VOL_LOOKBACK = 20

# === Ngưỡng lọc ===
MIN_VOLUME_RATIO = 1.5
MIN_ADX = 22
PE_MIN, PE_MAX_DEFAULT = 5, 35
PB_MAX = 5
MIN_MARKET_CAP_BILLION = 500
MARKET_RSI_OK = 45

# === Quản lý vốn ===
TOTAL_CAPITAL = 100_000_000
MAX_POSITION_PERCENT = 15
RISK_PER_TRADE_PERCENT = 2
MAX_DAILY_LOSS_PERCENT = 5

# === Chi phí giao dịch ===
COMMISSION_RATE = 0.15 / 100
SLIPPAGE_RATE = 0.10 / 100
STAMP_DUTY_RATE = 0.10 / 100

# === Cấu hình chạy ===
BACKTEST_DAYS = 180
TOP_STOCKS_LIMIT = 8
TRAIN_MODEL = True      # ✅ Tự huấn luyện lần đầu
OPTIMIZE_PARAMS = False

# === File lưu ===
PE_INDUSTRY_FILE = "pe_industry_data.json"
MODEL_FILE = "ml_model.pkl"

# ============================================================
# 📋 DANH SÁCH NGÀNH & P/E TRUNG BÌNH
# ============================================================
STOCK_SECTOR = {
    "ACB": "Tài chính-Ngân hàng", "VCB": "Tài chính-Ngân hàng",
    "BID": "Tài chính-Ngân hàng", "CTG": "Tài chính-Ngân hàng",
    "MBB": "Tài chính-Ngân hàng", "TCB": "Tài chính-Ngân hàng",
    "VPB": "Tài chính-Ngân hàng", "HDB": "Tài chính-Ngân hàng",
    "STB": "Tài chính-Ngân hàng",
    "VIC": "Bất động sản", "VHM": "Bất động sản",
    "KDH": "Bất động sản", "NLG": "Bất động sản",
    "VRE": "Bất động sản",
    "GAS": "Năng lượng", "PLX": "Năng lượng", "POW": "Năng lượng",
    "HPG": "Vật liệu", "HSG": "Vật liệu", "NKG": "Vật liệu",
    "FPT": "Công nghệ", "MWG": "Tiêu dùng", "PNJ": "Tiêu dùng",
    "MSN": "Tiêu dùng", "VNM": "Tiêu dùng",
    "DHG": "Y tế", "BVH": "Y tế",
    "GVR": "Công nghiệp", "BCM": "Công nghiệp",
    "CII": "Công nghiệp", "VHC": "Công nghiệp",
    "SCS": "Tiện ích", "REE": "Tiện ích"
}

PE_INDUSTRY_AVG = {
    "Tài chính-Ngân hàng": {"pe_avg": 12.5, "pe_min": 8, "pe_max": 18},
    "Bất động sản": {"pe_avg": 18.0, "pe_min": 10, "pe_max": 30},
    "Năng lượng": {"pe_avg": 14.0, "pe_min": 8, "pe_max": 22},
    "Vật liệu": {"pe_avg": 16.0, "pe_min": 9, "pe_max": 28},
    "Công nghệ": {"pe_avg": 25.0, "pe_min": 15, "pe_max": 45},
    "Tiêu dùng": {"pe_avg": 20.0, "pe_min": 12, "pe_max": 35},
    "Y tế": {"pe_avg": 22.0, "pe_min": 14, "pe_max": 38},
    "Công nghiệp": {"pe_avg": 15.0, "pe_min": 9, "pe_max": 25},
    "Tiện ích": {"pe_avg": 13.0, "pe_min": 8, "pe_max": 20},
    "Khác": {"pe_avg": 18.0, "pe_min": 8, "pe_max": 35}
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
    "Tiện ích": {"rsi_low": 48, "rsi_high": 68, "vol_min": 1.1, "atr_mult": 1.1, "adx_min": 20},
    "Khác": {"rsi_low": 55, "rsi_high": 75, "vol_min": 1.5, "atr_mult": 1.5, "adx_min": 25}
}

# ============================================================
# 📡 LẤY DỮ LIỆU TỪ API VNDIRECT
# ============================================================
def get_vnindex_data(days=260):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date&q=code:VNINDEX~date:gte:{start_date}~date:lte:{end_date}&size=500"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 60:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(9).mean()
        loss = -delta.where(delta < 0, 0).rolling(9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        latest = df.iloc[-1]
        above_ema20 = latest['close'] > latest['ema20']
        rsi_val = latest['rsi']
        if above_ema20 and rsi_val > 50:
            trend, note = "TĂNG", "Thị trường thuận lợi"
        elif not above_ema20 and rsi_val < 50:
            trend, note = "GIẢM", "Thị trường suy yếu — cẩn trọng"
        else:
            trend, note = "NGANG", "Thị trường đi ngang"
        market_ok = above_ema20 and rsi_val >= MARKET_RSI_OK
        return {
            "close": float(latest['close']),
            "rsi": float(rsi_val),
            "trend": trend,
            "note": note,
            "market_ok": market_ok,
            "df": df[['date', 'close']]
        }
    except Exception as e:
        print(f"⚠️ Lỗi VNINDEX: {e}")
        return None


def get_stock_data(ticker, days=260):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{ticker}~date:gte:{start_date}~date:lte:{end_date}&size=500"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 100:
            return None
        df = pd.DataFrame(data)
        df = df.rename(columns={"nmVolume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df
    except Exception as e:
        return None


def get_filtered_stocks():
    return list(STOCK_SECTOR.keys())

# ============================================================
# 🏭 QUẢN LÝ P/E NGÀNH
# ============================================================
class IndustryPEManager:
    def __init__(self):
        self.data = PE_INDUSTRY_AVG.copy()
        self.last_update = None
        self.load_saved_data()
    
    def load_saved_data(self):
        if os.path.exists(PE_INDUSTRY_FILE):
            try:
                with open(PE_INDUSTRY_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.data.update(saved.get('data', {}))
                    self.last_update = saved.get('updated')
            except: pass
    
    def get_pe_range(self, sector):
        info = self.data.get(sector, self.data["Khác"])
        return info["pe_min"], info["pe_max"], info["pe_avg"]

pe_manager = IndustryPEManager()

# ============================================================
# 📡 DỮ LIỆU CƠ BẢN
# ============================================================
class FundamentalData:
    @staticmethod
    def get_fundamental_data(ticker):
        try:
            url = f"https://api-finfo.vndirect.com.vn/v4/stocks/detail?code={ticker}"
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
            resp = requests.get(url, timeout=15, headers=headers)
            data = resp.json().get("data", {})
            return {
                "pe": float(data.get('pe', 0) or 0),
                "pb": float(data.get('pb', 0) or 0),
                "market_cap_billion": float(data.get('marketCap', 0) or 0) / 1000000000
            }
        except:
            return {"pe": 0, "pb": 0, "market_cap_billion": 0}

# ============================================================
# 📊 PHÂN KỲ GIÁ - CHỈ BÁO
# ============================================================
class DivergenceDetector:
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def find_peaks(self, series, order=5):
        peaks = []
        for i in range(order, len(series) - order):
            if series.iloc[i] > series.iloc[i-order:i].max() and series.iloc[i] > series.iloc[i+1:i+order+1].max():
                peaks.append((i, series.iloc[i]))
        return peaks
    
    def find_troughs(self, series, order=5):
        troughs = []
        for i in range(order, len(series) - order):
            if series.iloc[i] < series.iloc[i-order:i].min() and series.iloc[i] < series.iloc[i+1:i+order+1].min():
                troughs.append((i, series.iloc[i]))
        return troughs
    
    def check_divergence(self, price, indicator, is_bullish=True):
        result = {"found": False, "type": None, "strength": 0, "note": ""}
        if is_bullish:
            p_t = self.find_troughs(price)
            i_t = self.find_troughs(indicator)
            if len(p_t) >= 2 and len(i_t) >= 2:
                if p_t[-1][1] < p_t[-2][1] and i_t[-1][1] > i_t[-2][1]:
                    result = {"found": True, "type": "DƯƠNG", "strength": 75, "note": "✅ Phân kỳ dương — tiềm năng phục hồi"}
        else:
            p_p = self.find_peaks(price)
            i_p = self.find_peaks(indicator)
            if len(p_p) >= 2 and len(i_p) >= 2:
                if p_p[-1][1] > p_p[-2][1] and i_p[-1][1] < i_p[-2][1]:
                    result = {"found": True, "type": "ÂM", "strength": 75, "note": "⚠️ Phân kỳ âm — cẩn trọng điều chỉnh"}
        return result
    
    def detect_all(self, df):
        df = df.tail(self.lookback).copy()
        rsi_bull = self.check_divergence(df['close'], df['rsi'], True)
        rsi_bear = self.check_divergence(df['close'], df['rsi'], False)
        macd_bull = self.check_divergence(df['close'], df['macd'], True)
        macd_bear = self.check_divergence(df['close'], df['macd'], False)
        return {
            "rsi_bullish": rsi_bull, "rsi_bearish": rsi_bear,
            "macd_bullish": macd_bull, "macd_bearish": macd_bear,
            "has_bullish": rsi_bull['found'] or macd_bull['found'],
            "has_bearish": rsi_bear['found'] or macd_bear['found']
        }

# ============================================================
# 📅 XÁC NHẬN ĐA KHUNG THỜI GIAN
# ============================================================
class MultiTimeframeConfirm:
    def resample_weekly(self, df_daily):
        df = df_daily.copy()
        df['week'] = df['date'].dt.to_period('W')
        return df.groupby('week').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).reset_index(drop=True)
    
    def calculate_trend_score(self, df):
        if len(df) < 30: return 0
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        latest = df.iloc[-1]
        score = 0
        if latest['close'] > latest['ema20']: score += 1
        if latest['ema20'] > latest['ema50']: score += 1
        return score
    
    def get_confirmation(self, df_daily):
        w1 = self.resample_weekly(df_daily)
        d_score = self.calculate_trend_score(df_daily)
        w_score = self.calculate_trend_score(w1)
        confidence = int(((d_score/2)*0.5 + (w_score/2)*0.5) * 100)
        strong = d_score >= 1 and w_score >= 1
        level = "MẠNH ✅" if strong else ("KHẢ QUAN ⚡" if confidence >= 40 else "TRÁNH ❌")
        return {"confidence": confidence, "level": level, "strong_bullish": strong}

# ============================================================
# 🤖 MÁY HỌC — RANDOM FOREST
# ============================================================
class StockMLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_accuracy = 0
        self.load_model()
    
    def prepare_features(self, df):
        df = df.copy()
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
        df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(9).mean()
        loss = -delta.where(delta < 0, 0).rolling(9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['avg_vol_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']
        tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['future_return_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return_5d'] >= 0.02).astype(int)
        df = df.dropna()
        features = ['return_1d', 'return_5d', 'price_above_ema20', 'price_above_ema50', 'rsi', 'macd', 'vol_ratio', 'atr']
        self.feature_names = features
        return df, features
    
    def train(self, historical_data_dict):
        print("\n🤖 === BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ML ===")
        all_X, all_y = [], []
        
        for ticker, df in historical_data_dict.items():
            try:
                df_prep, features = self.prepare_features(df)
                if len(df_prep) > 30:
                    all_X.append(df_prep[features].values)
                    all_y.append(df_prep['target'].values)
            except Exception as e:
                print(f"⚠️ Bỏ qua {ticker}: {e}")
        
        if not all_X:
            print("❌ Không có dữ liệu để huấn luyện")
            return False
        
        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, shuffle=False, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("📈 Sử dụng mô hình: Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=20, random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True
        print(f"✅ Huấn luyện hoàn tất! Độ chính xác: {self.model_accuracy:.2%}")
        self.save_model()
        return True
    
    def predict_probability(self, df):
        if not self.is_trained or self.model is None:
            return 50.0, "Chưa huấn luyện"
        try:
            df_prep, features = self.prepare_features(df)
            if len(df_prep) < 1:
                return 50.0, "Không đủ dữ liệu"
            latest = df_prep[features].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest)
            prob = self.model.predict_proba(latest_scaled)[0, 1] * 100
            note = "✅ Cao" if prob >= 70 else ("⚡ Khả quan" if prob >= 55 else "⏳ Thấp")
            return round(prob, 1), note
        except Exception as e:
            print(f"⚠️ Lỗi dự báo: {e}")
            return 50.0, "Lỗi"
    
    def save_model(self):
        try:
            import pickle
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'model': self.model, 'scaler': self.scaler,
                    'features': self.feature_names, 'accuracy': self.model_accuracy
                }, f)
            print(f"💾 Đã lưu mô hình")
        except Exception as e:
            print(f"⚠️ Không lưu được: {e}")
    
    def load_model(self):
        if not os.path.exists(MODEL_FILE):
            return False
        try:
            import pickle
            with open(MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['features']
                self.model_accuracy = data.get('accuracy', 0)
                self.is_trained = True
                print(f"✅ Tải mô hình thành công | Độ chính xác: {self.model_accuracy:.2%}")
            return True
        except Exception as e:
            print(f"⚠️ Không tải được mô hình: {e}")
            return False

# ============================================================
# 📡 THÔNG BÁO DISCORD
# ============================================================
class DiscordNotifier:
    def __init__(self, webhook_url):
        self.url = webhook_url
        self.enabled = bool(webhook_url and webhook_url.startswith("http"))
    
    def send_embed(self, title, desc, color=0x3498db):
        full_desc = desc + f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"\n📢 [{title}]\n{full_desc}\n")
        if not self.enabled:
            return True
        try:
            requests.post(self.url, json={
                "embeds": [{"title": title, "description": full_desc, "color": color}]
            }, timeout=10)
            return True
        except Exception as e:
            print(f"⚠️ Lỗi gửi Discord: {e}")
            return False
    
    def market_summary(self, vnindex):
        emoji = "🟢" if vnindex['trend'] == "TĂNG" else ("🟡" if vnindex['trend'] == "NGANG" else "🔴")
        return self.send_embed("📊 TỔNG QUAN THỊ TRƯỜNG",
            f"{emoji} VNINDEX: {vnindex['close']:.2f}\n📈 Xu hướng: {vnindex['trend']}\n💡 {vnindex['note']}", 0x3498db)
    
    def top_stocks(self, stocks, vnindex):
        text = ""
        for i, s in enumerate(stocks, 1):
            icon = "✅" if "MUA" in s['signal'] else "⏳"
            text += f"#{i} {icon} {s['ticker']} — {s['score']}/100 — {s['signal']}\n"
        market_note = f"\n📊 VNINDEX: {vnindex['trend']}" if vnindex else ""
        return self.send_embed("🏆 TOP MÃ HÔM NAY", f"{text}{market_note}\n⚠️ Chỉ tham khảo — không phải lời khuyên đầu tư", 0xf1c40f)

# ============================================================
# 📊 PHÂN TÍCH KỸ THUẬT
# ============================================================
class TechnicalAnalyzer:
    def __init__(self):
        self.divergence = DivergenceDetector()
        self.timeframe = MultiTimeframeConfirm()
        self.ml = StockMLPredictor()
    
    def calculate_indicators(self, df):
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(9).mean()
        loss = -delta.where(delta < 0, 0).rolling(9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_wilder = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_wilder
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_wilder
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        df['avg_vol_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']
        return df
    
    def generate_signal(self, df, ticker, sector, fund_data, market_ok=True):
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        entry = round(float(latest['close']), 2)
        params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
        
        pe_min_sec, pe_max_sec, _ = pe_manager.get_pe_range(sector)
        pe_ok = True
        if fund_data['pe'] > 0:
            pe_ok = pe_min_sec <= fund_data['pe'] <= pe_max_sec
        
        div = self.divergence.detect_all(df)
        mtf = self.timeframe.get_confirmation(df)
        ml_prob, ml_note = self.ml.predict_probability(df)
        
        score = 0
        if latest['close'] > latest['ema20'] and latest['ema20'] > latest['ema50']:
            score += 20
        if latest['vol_ratio'] >= params['vol_min']:
            score += 15
        if latest['adx'] > params['adx_min'] and latest['plus_di'] > latest['minus_di']:
            score += 15
        if params['rsi_low'] < latest['rsi'] < params['rsi_high']:
            score += 10
        if latest['macd'] > latest['macd_signal']:
            score += 10
        if div['has_bullish']:
            score += 10
        if mtf['strong_bullish']:
            score += 15
        if pe_ok:
            score += 5
        
        score = int(max(0, min(100, score)))
        if not market_ok:
            score = int(score * 0.7)
        
        atr_val = float(latest['atr'])
        sl_atr = round(entry - atr_val * params['atr_mult'], 2)
        tp_atr = round(entry + atr_val * 2.5, 2)
        
        signal = "✅ MUA" if (score >= 65 and mtf['confidence'] >= 40 and ml_prob >= 55 and market_ok and pe_ok) else \
                 "⚡ XEM XÉT" if score >= 50 else "⏳ THEO DÕI"
        
        return {
            "ticker": ticker, "sector": sector, "signal": signal, "score": score,
            "entry": entry, "SL_atr": sl_atr, "TP_atr": tp_atr,
            "rsi": round(float(latest['rsi']), 1), "adx": round(float(latest['adx']), 1),
            "vol_ratio": round(float(latest['vol_ratio']), 2), "ml_prob": ml_prob, "ml_note": ml_note
        }

# ============================================================
# 🚀 BOT CHÍNH
# ============================================================
class SmartStockBot:
    def __init__(self):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
    
    def train_ml_model(self):
        print("\n🤖 === THU THẬP DỮ LIỆU HUẤN LUYỆN ===")
        tickers = list(STOCK_SECTOR.keys())
        historical = {}
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ")
            df = get_stock_data(ticker, days=BACKTEST_DAYS + 120)
            if df is not None and len(df) >= 100:
                historical[ticker] = df
                print(f"✅ {len(df)} phiên")
            else:
                print("⚠️ Bỏ qua")
            time.sleep(0.3)
        if historical:
            self.analyzer.ml.train(historical)
        else:
            print("❌ Không đủ dữ liệu huấn luyện")
    
    def run_scan(self):
        print("\n" + "=" * 80)
        print("📊 === BẮT ĐẦU QUÉT THỊ TRƯỜNG ===")
        print("=" * 80)
        
        vnindex = get_vnindex_data()
        market_ok = vnindex['market_ok'] if vnindex else True
        if vnindex:
            print(f"   VNINDEX: {vnindex['close']:.2f} | RSI: {vnindex['rsi']:.1f} | {vnindex['trend']}")
            print(f"   Thị trường {'THUẬN LỢI ✅' if market_ok else 'KHÔNG THUẬN LỢI ⚠️ — Giảm điểm tín hiệu 30%'}")
            self.notifier.market_summary(vnindex)
        
        candidates = get_filtered_stocks()
        results = []
        print(f"\n⏳ Đang phân tích {len(candidates)} mã...")
        
        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i:2d}/{len(candidates)}] {ticker:<6s}...", end=" ", flush=True)
            df = get_stock_data(ticker)
            if df is None:
                print("❌ Không có dữ liệu")
                continue
            sector = STOCK_SECTOR.get(ticker, "Khác")
            fund = FundamentalData.get_fundamental_data(ticker)
            sig = self.analyzer.generate_signal(df, ticker, sector, fund, market_ok)
            results.append(sig)
            print(f"Điểm: {sig['score']:3d}/100 | {sig['signal']} | ML: {sig['ml_prob']:.0f}%")
            time.sleep(0.3)
        
        if not results:
            print("❌ Không có kết quả phân tích!")
            return []
        
        results.sort(key=lambda x: x['score'], reverse=True)
        top = results[:TOP_STOCKS_LIMIT]
        
        print("\n" + "=" * 80)
        print(f"🏆 TOP {len(top)} MÃ HÀNG ĐẦU")
        print("=" * 80)
        for rank, s in enumerate(top, 1):
            print(f"#{rank} {s['ticker']:<6s} | {s['sector'][:18]} | Điểm: {s['score']}/100 | {s['signal']}")
            print(f"       Giá: {s['entry']} | SL: {s['SL_atr']} | TP: {s['TP_atr']}")
        print("=" * 80 + "\n")
        
        self.notifier.top_stocks(top, vnindex)
        return top

# ============================================================
# 🏃 KHỞI ĐỘNG
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 BOT CHỨNG KHOÁN VNDIRECT — PHIÊN BẢN HOÀN CHỈNH")
    print("=" * 80)
    print("📊 Nguồn dữ liệu: api-finfo.vndirect.com.vn")
    print("🧠 ML: Random Forest | Phân kỳ RSI/MACD | P/E ngành động")
    print("⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ")
    print("=" * 80 + "\n")
    
    bot = SmartStockBot()
    
    if TRAIN_MODEL:
        bot.train_ml_model()
    
    bot.notifier.send_embed("✅ BOT ĐÃ ONLINE",
        f"⏰ Bắt đầu lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"🤖 Mô hình ML: {'Đã huấn luyện' if bot.analyzer.ml.is_trained else 'Chưa huấn luyện'}\n"
        f"📊 Số mã theo dõi: {len(STOCK_SECTOR)}\n"
        f"⚠️ KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ — Chỉ tham khảo phân tích kỹ thuật", 0x2ecc71)
    
    print("⏳ Quét lần đầu...\n")
    bot.run_scan()
    
    last_scan = None
    print("\n⏳ Bot đang chạy — sẽ quét mới vào 8:00 sáng Thứ 2 đến Thứ 6")
    
    while True:
        time.sleep(60)
        now = datetime.now()
        is_weekday = now.weekday() < 5
        is_8am = now.hour == 8 and now.minute <= 5
        today = now.strftime("%Y-%m-%d")
        
        if is_weekday and is_8am and last_scan != today:
            print("\n" + "=" * 80)
            print(f"🕐 {now.strftime('%A %d/%m/%Y %H:%M')} — QUÉT MỚI")
            print("=" * 80)
            bot.run_scan()
            last_scan = today
            print("✅ Hoàn thành. Chờ đến sáng mai...\n")


       
