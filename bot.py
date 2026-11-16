# ============================================================
# 🇻🇳 BOT LỌC CHỨNG KHOÁN — PHIÊN BẢN 2.0 NÂNG CẤP
# ============================================================
# ✅ Phân kỳ RSI/MACD nâng cao
# ✅ P/E trung bình ngành động
# ✅ Quản lý vốn & vị trí thông minh
# ✅ Backtest với slippage + hoa hồng thực tế
# ✅ Xác nhận đa khung thời gian (W1/M1)
# ✅ ML: XGBoost + Random Forest dự báo xác suất
# ✅ Grid Search tối ưu tham số tự động
# Nguồn dữ liệu: API VNDIRECT
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

# === ML ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost chưa cài đặt — chỉ dùng Random Forest")

# ============================================================
# 🔧 CẤU HÌNH NÂNG CAO
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

# === Ngưỡng lọc ===
MIN_VOLUME_RATIO = 1.5
MIN_ADX = 25
PE_MIN = 5
PE_MAX_DEFAULT = 35
PB_MAX = 5
MIN_MARKET_CAP_BILLION = 500

# === Quản lý vốn ===
TOTAL_CAPITAL = 100_000_000  # 100 triệu vốn ảo
MAX_POSITION_PERCENT = 15     # Tối đa 15% vốn/1 mã
MAX_TOTAL_EXPOSURE = 60       # Tối đa 60% tổng vốn đầu tư
RISK_PER_TRADE_PERCENT = 2    # Rủi ro tối đa 2% vốn/giao dịch
MAX_DAILY_LOSS_PERCENT = 5    # Dừng nếu lỗ 5%/ngày

# === Chi phí giao dịch thực tế ===
COMMISSION_RATE = 0.15 / 100  # Hoa hồng 0.15% mỗi chiều
Slippage_RATE = 0.10 / 100     # Độ trượt giá 0.10%
STAMP_DUTY_RATE = 0.10 / 100   # Thuế chuyển nhượng 0.1% khi bán

# === Backtest ===
BACKTEST_DAYS = 180
TOP_STOCKS_LIMIT = 8

# === File lưu dữ liệu & mô hình ===
PE_INDUSTRY_FILE = "pe_industry_data.json"
MODEL_FILE = "ml_model.json"
HISTORY_FILE = "signal_history.csv"

# ============================================================
# 🏭 DANH SÁCH NGÀNH & DỮ LIỆU P/E TRUNG BÌNH NGÀNH
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

# P/E trung bình ngành — sẽ được cập nhật động
PE_INDUSTRY_AVG = {
    "Tài chính-Ngân hàng": {"pe_avg": 12.5, "pe_min": 8, "pe_max": 18},
    "Bất động sản": {"pe_avg": 18.0, "pe_min": 10, "pe_max": 30},
    "Năng lượng": {"pe_avg": 14.0, "pe_min": 8, "pe_max": 22},
    "Vật liệu": {"pe_avg": 16.0, "pe_min": 9, "pe_max": 28},
    "Công nghệ": {"pe_avg": 25.0, "pe_min": 15, "pe_max": 45},
    "Tiêu dùng": {"pe_avg": 20.0, "pe_min": 12, "pe_max": 35},
    "Y tế": {"pe_avg": 22.0, "pe_min": 14, "pe_max": 38},
    "Công nghiệp": {"pe_avg": 15.0, "pe_min": 9, "pe_max": 25},
    "Dịch vụ tiện ích": {"pe_avg": 13.0, "pe_min": 8, "pe_max": 20},
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
    "Dịch vụ tiện ích": {"rsi_low": 48, "rsi_high": 68, "vol_min": 1.1, "atr_mult": 1.1, "adx_min": 20},
    "Khác": {"rsi_low": 55, "rsi_high": 75, "vol_min": 1.5, "atr_mult": 1.5, "adx_min": 25}
}

# ============================================================
# 📡 QUẢN LÝ DỮ LIỆU P/E NGÀNH ĐỘNG
# ============================================================
class IndustryPEManager:
    """Quản lý và cập nhật P/E trung bình ngành từ dữ liệu thực tế"""
    
    def __init__(self):
        self.data = PE_INDUSTRY_AVG.copy()
        self.last_update = None
        self.load_saved_data()
    
    def load_saved_data(self):
        """Tải dữ liệu đã lưu nếu có"""
        if os.path.exists(PE_INDUSTRY_FILE):
            try:
                with open(PE_INDUSTRY_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.data.update(saved.get('data', {}))
                    self.last_update = saved.get('updated')
                    print(f"✅ Tải P/E ngành: Cập nhật cuối: {self.last_update}")
            except Exception as e:
                print(f"⚠️ Không tải được P/E ngành: {e}")
    
    def save_data(self):
        """Lưu dữ liệu ra file"""
        try:
            with open(PE_INDUSTRY_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'data': self.data,
                    'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Không lưu được P/E ngành: {e}")
    
    def update_from_market(self):
        """Cập nhật P/E trung bình ngành từ dữ liệu thực tế"""
        print("\n🔄 Đang cập nhật P/E trung bình ngành...")
        industry_pe_collect = {}
        
        for ticker, sector in STOCK_SECTOR.items():
            try:
                url = f"https://api-finfo.vndirect.com.vn/v4/stocks/detail?code={ticker}"
                headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
                resp = requests.get(url, timeout=10, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    pe = float(data.get('pe', 0))
                    if pe > 0 and pe < 200:
                        if sector not in industry_pe_collect:
                            industry_pe_collect[sector] = []
                        industry_pe_collect[sector].append(pe)
                time.sleep(0.1)
            except:
                pass
        
        # Tính trung bình
        updated_count = 0
        for sector, pes in industry_pe_collect.items():
            if len(pes) >= 3:
                avg_pe = np.mean(pes)
                min_pe = max(5, np.percentile(pes, 10))
                max_pe = np.percentile(pes, 90)
                self.data[sector] = {
                    "pe_avg": round(avg_pe, 1),
                    "pe_min": round(min_pe, 1),
                    "pe_max": round(max_pe, 1)
                }
                updated_count += 1
                print(f"   {sector:<20s}: P/E trung bình = {avg_pe:.1f} (khoảng {min_pe:.1f}-{max_pe:.1f})")
        
        if updated_count > 0:
            self.save_data()
            self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ Cập nhật thành công {updated_count} ngành\n")
    
    def get_pe_range(self, sector):
        """Lấy khoảng P/E cho ngành"""
        info = self.data.get(sector, self.data["Khác"])
        return info["pe_min"], info["pe_max"], info["pe_avg"]

# Khởi tạo toàn cục
pe_manager = IndustryPEManager()

# ============================================================
# 📡 QUẢN LÝ VỐN & RỦI RO
# ============================================================
class RiskManager:
    """Quản lý vốn, vị trí và giới hạn rủi ro"""
    
    def __init__(self, total_capital=TOTAL_CAPITAL):
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.positions = {}  # ticker -> {shares, entry_price, value, risk_amount}
        self.daily_pnl = 0
        self.daily_start_equity = total_capital
        self.trade_log = []
    
    def reset_daily(self):
        """Reset mỗi ngày giao dịch mới"""
        self.daily_pnl = 0
        self.daily_start_equity = self.calculate_total_equity()
    
    def calculate_total_equity(self):
        """Tính tổng tài sản hiện tại"""
        positions_value = sum(p.get('value', 0) for p in self.positions.values())
        return self.available_capital + positions_value
    
    def check_daily_loss_limit(self):
        """Kiểm tra giới hạn lỗ hàng ngày"""
        current_equity = self.calculate_total_equity()
        loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity * 100
        if loss_pct >= MAX_DAILY_LOSS_PERCENT:
            print(f"⚠️ ĐÃ ĐẠT GIỚI HẠN LỖ HÀNG NGÀY: {loss_pct:.2f}% — DỪNG MỞ VỊ TRÍ MỚI!")
            return True
        return False
    
    def calculate_position_size(self, entry_price, stop_loss_price, confidence=1.0):
        """
        Tính khối lượng tối ưu dựa trên rủi ro
        Công thức: Vốn rủi ro = 2% vốn / (Giá vào - Giá dừng lỗ)
        """
        if entry_price <= stop_loss_price:
            return 0, "Giá dừng lỗ không hợp lệ"
        
        risk_per_share = entry_price - stop_loss_price
        max_risk_amount = self.total_capital * (RISK_PER_TRADE_PERCENT / 100) * confidence
        
        # Giới hạn theo % vốn
        max_value_by_percent = self.total_capital * (MAX_POSITION_PERCENT / 100)
        
        shares_by_risk = int(max_risk_amount / risk_per_share)
        shares_by_capital = int(max_value_by_percent / entry_price)
        
        shares = min(shares_by_risk, shares_by_capital)
        
        # Kiểm tra tổng vị trí
        current_exposure = sum(p.get('value', 0) for p in self.positions.values())
        new_position_value = shares * entry_price
        if current_exposure + new_position_value > self.total_capital * (MAX_TOTAL_EXPOSURE / 100):
            max_allowed_value = self.total_capital * (MAX_TOTAL_EXPOSURE / 100) - current_exposure
            shares = int(max_allowed_value / entry_price)
        
        if shares <= 0:
            return 0, "Không đủ vốn hoặc rủi ro quá cao"
        
        return shares, f"Khối lượng: {shares:,} | Giá trị: {shares*entry_price:,.0f}đ | Rủi ro tối đa: {shares*risk_per_share:,.0f}đ"
    
    def calculate_transaction_cost(self, price, shares, is_buy=True):
        """Tính tổng chi phí giao dịch thực tế"""
        value = price * shares
        commission = value * COMMISSION_RATE
        slippage_cost = value * Slippage_RATE
        stamp_duty = value * STAMP_DUTY_RATE if not is_buy else 0
        
        total_cost = commission + slippage_cost + stamp_duty
        net_price = price * (1 - Slippage_RATE) if is_buy else price * (1 + Slippage_RATE)
        
        return {
            "gross_value": value,
            "commission": commission,
            "slippage": slippage_cost,
            "stamp_duty": stamp_duty,
            "total_cost": total_cost,
            "net_price": net_price
        }
    
    def open_position(self, ticker, entry_price, stop_loss, confidence=1.0):
        """Mở vị trí mới với quản lý rủi ro"""
        if self.check_daily_loss_limit():
            return False, "Đã đạt giới hạn lỗ hàng ngày"
        
        shares, note = self.calculate_position_size(entry_price, stop_loss, confidence)
        if shares == 0:
            return False, note
        
        cost = self.calculate_transaction_cost(entry_price, shares, is_buy=True)
        total_outlay = cost['gross_value'] + cost['total_cost']
        
        if total_outlay > self.available_capital:
            return False, "Không đủ vốn khả dụng"
        
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "value": cost['gross_value'],
            "risk_amount": shares * (entry_price - stop_loss),
            "entry_time": datetime.now(),
            "cost_detail": cost
        }
        
        self.available_capital -= total_outlay
        return True, f"{note}\n💵 Chi phí giao dịch: {cost['total_cost']:,.0f}đ (HH: {cost['commission']:,.0f} + Trượt: {cost['slippage']:,.0f})"
    
    def close_position(self, ticker, exit_price):
        """Đóng vị trí và tính kết quả"""
        if ticker not in self.positions:
            return 0, 0, "Không có vị trí"
        
        pos = self.positions[ticker]
        cost = self.calculate_transaction_cost(exit_price, pos['shares'], is_buy=False)
        gross_revenue = cost['gross_value']
        net_revenue = gross_revenue - cost['total_cost']
        
        entry_value = pos['shares'] * pos['entry_price']
        profit = net_revenue - entry_value
        profit_pct = (profit / entry_value) * 100
        
        self.available_capital += net_revenue
        self.daily_pnl += profit
        
        self.trade_log.append({
            "ticker": ticker,
            "entry_price": pos['entry_price'],
            "exit_price": exit_price,
            "shares": pos['shares'],
            "profit": profit,
            "profit_pct": profit_pct,
            "duration_days": (datetime.now() - pos['entry_time']).days,
            "commission_buy": pos['cost_detail']['commission'],
            "commission_sell": cost['commission'],
            "slippage_buy": pos['cost_detail']['slippage'],
            "slippage_sell": cost['slippage'],
            "stamp_duty": cost['stamp_duty']
        })
        
        del self.positions[ticker]
        return profit, profit_pct, f"Lợi nhuận: {profit:,.0f}đ ({profit_pct:+.2f}%)"

# ============================================================
# 📊 PHÂN KỲ NÂNG CAO — RSI & MACD
# ============================================================
class DivergenceDetector:
    """Phát hiện phân kỳ giá-RSI và giá-MACD nâng cao"""
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def find_peaks(self, series, order=5):
        """Tìm đỉnh cục bộ"""
        peaks = []
        for i in range(order, len(series) - order):
            if series.iloc[i] > series.iloc[i-order:i].max() and series.iloc[i] > series.iloc[i+1:i+order+1].max():
                peaks.append((i, series.iloc[i]))
        return peaks
    
    def find_troughs(self, series, order=5):
        """Tìm đáy cục bộ"""
        troughs = []
        for i in range(order, len(series) - order):
            if series.iloc[i] < series.iloc[i-order:i].min() and series.iloc[i] < series.iloc[i+1:i+order+1].min():
                troughs.append((i, series.iloc[i]))
        return troughs
    
    def check_divergence(self, price, indicator, is_bullish=True):
        """
        Kiểm tra phân kỳ
        - Phân kỳ dương: Giá giảm sâu hơn → Chỉ báo giảm ít hơn → Tín hiệu MUA
        - Phân kỳ âm: Giá tăng cao hơn → Chỉ báo tăng ít hơn → Tín hiệu BÁN
        """
        peaks_price = self.find_peaks(price)
        peaks_ind = self.find_peaks(indicator)
        troughs_price = self.find_troughs(price)
        troughs_ind = self.find_troughs(indicator)
        
        result = {"found": False, "type": None, "strength": 0, "note": ""}
        
        if is_bullish and len(troughs_price) >= 2 and len(troughs_ind) >= 2:
            # Phân kỳ dương: Giá tạo đáy thấp hơn, chỉ báo tạo đáy cao hơn
            p1_idx, p1_val = troughs_price[-2]
            p2_idx, p2_val = troughs_price[-1]
            i1_idx, i1_val = troughs_ind[-2]
            i2_idx, i2_val = troughs_ind[-1]
            
            if p2_val < p1_val and i2_val > i1_val and p2_idx > p1_idx and i2_idx > i1_idx:
                price_drop = (p1_val - p2_val) / p1_val
                ind_rise = (i2_val - i1_val) / (abs(i1_val) + 1e-9)
                result = {
                    "found": True,
                    "type": "DƯƠNG",
                    "strength": min(100, (price_drop * 100) + (ind_rise * 50)),
                    "note": f"✅ Phân kỳ dương: Giá giảm {price_drop*100:.1f}% nhưng chỉ báo tăng → Tiềm năng phục hồi"
                }
        
        elif not is_bullish and len(peaks_price) >= 2 and len(peaks_ind) >= 2:
            # Phân kỳ âm: Giá tạo đỉnh cao hơn, chỉ báo tạo đỉnh thấp hơn
            p1_idx, p1_val = peaks_price[-2]
            p2_idx, p2_val = peaks_price[-1]
            i1_idx, i1_val = peaks_ind[-2]
            i2_idx, i2_val = peaks_ind[-1]
            
            if p2_val > p1_val and i2_val < i1_val and p2_idx > p1_idx and i2_idx > i1_idx:
                price_rise = (p2_val - p1_val) / p1_val
                ind_drop = (i1_val - i2_val) / (abs(i1_val) + 1e-9)
                result = {
                    "found": True,
                    "type": "ÂM",
                    "strength": min(100, (price_rise * 100) + (ind_drop * 50)),
                    "note": f"⚠️ Phân kỳ âm: Giá tăng {price_rise*100:.1f}% nhưng chỉ báo suy yếu → Cẩn trọng điều chỉnh"
                }
        
        return result
    
    def detect_all(self, df):
        """Phát hiện tất cả loại phân kỳ"""
        df = df.tail(self.lookback).copy()
        
        rsi_bull = self.check_divergence(df['close'], df['rsi'], is_bullish=True)
        rsi_bear = self.check_divergence(df['close'], df['rsi'], is_bullish=False)
        macd_bull = self.check_divergence(df['close'], df['macd'], is_bullish=True)
        macd_bear = self.check_divergence(df['close'], df['macd'], is_bullish=False)
        
        return {
            "rsi_bullish": rsi_bull,
            "rsi_bearish": rsi_bear,
            "macd_bullish": macd_bull,
            "macd_bearish": macd_bear,
            "has_bullish": rsi_bull['found'] or macd_bull['found'],
            "has_bearish": rsi_bear['found'] or macd_bear['found'],
            "strongest_bullish": max([r for r in [rsi_bull, macd_bull] if r['found']], 
                                    key=lambda x: x['strength'], default=None),
            "strongest_bearish": max([r for r in [rsi_bear, macd_bear] if r['found']], 
                                    key=lambda x: x['strength'], default=None)
        }

# ============================================================
# 📡 XÁC NHẬN ĐA KHUNG THỜI GIAN (W1/M1)
# ============================================================
class MultiTimeframeConfirm:
    """Xác nhận tín hiệu từ khung Ngày → Tuần → Tháng để lọc tín hiệu giả"""
    
    def __init__(self):
        pass
    
    def resample_weekly(self, df_daily):
        """Chuyển dữ liệu D1 → W1"""
        df = df_daily.copy()
        df['week'] = df['date'].dt.to_period('W')
        weekly = df.groupby('week').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'date': 'last'
        }).reset_index(drop=True)
        return weekly
    
    def resample_monthly(self, df_daily):
        """Chuyển dữ liệu D1 → M1"""
        df = df_daily.copy()
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'date': 'last'
        }).reset_index(drop=True)
        return monthly
    
    def calculate_trend_score(self, df):
        """Tính điểm xu hướng cho khung thời gian"""
        if len(df) < 30:
            return 0, "không đủ dữ liệu"
        
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        latest = df.iloc[-1]
        score = 0
        notes = []
        
        if latest['close'] > latest['ema20']:
            score += 1
            notes.append("Giá trên EMA20")
        if latest['ema20'] > latest['ema50']:
            score += 1
            notes.append("EMA20 trên EMA50")
        
        return score, ", ".join(notes)
    
    def get_confirmation(self, df_daily):
        """Lấy xác nhận từ W1 và M1"""
        w1 = self.resample_weekly(df_daily)
        m1 = self.resample_monthly(df_daily)
        
        d_score, d_note = self.calculate_trend_score(df_daily)
        w_score, w_note = self.calculate_trend_score(w1)
        m_score, m_note = self.calculate_trend_score(m1)
        
        total_score = d_score + w_score * 1.5 + m_score * 2.0
        max_score = 2 + 3 + 4
        confidence_pct = int((total_score / max_score) * 100)
        
        # Xác nhận mạnh khi ít nhất 2 khung đồng thuận
        strong_bullish = (d_score >= 1 and w_score >= 1 and m_score >= 1)
        weak_bullish = (d_score >= 1 and (w_score >= 1 or m_score >= 1))
        
        if strong_bullish:
            level = "MẠNH ✅"
        elif weak_bullish:
            level = "KHẢ QUAN ⚡"
        elif confidence_pct >= 40:
            level = "YẾU ⏳"
        else:
            level = "TRÁNH ❌"
        
        return {
            "daily_score": d_score,
            "weekly_score": w_score,
            "monthly_score": m_score,
            "confidence": confidence_pct,
            "level": level,
            "strong_bullish": strong_bullish,
            "weak_bullish": weak_bullish,
            "notes": {
                "D1": d_note,
                "W1": w_note,
                "M1": m_note
            }
        }

# ============================================================
# 🤖 MÁY HỌC — XGBOOST + RANDOM FOREST
# ============================================================
class StockMLPredictor:
    """Huấn luyện và dự báo xác suất tăng giá bằng Machine Learning"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_accuracy = 0
        self.load_model()
    
    def prepare_features(self, df):
        """Tạo tập đặc trưng từ dữ liệu giá"""
        df = df.copy()
        
        # Đặc trưng cơ bản
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # EMA
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
        df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(9).mean()
        loss = -delta.where(delta < 0, 0).rolling(9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Khối lượng
        df['avg_vol_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Mục tiêu: Giá tăng >= 2% trong 5 phiên tới
        df['future_return_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return_5d'] >= 0.02).astype(int)
        
        # Loại bỏ dòng thiếu dữ liệu
        df = df.dropna()
        
        features = [
            'return_1d', 'return_5d', 'return_20d', 'volatility_20d',
            'price_above_ema20', 'price_above_ema50',
            'rsi', 'macd', 'macd_hist', 'vol_ratio', 'atr'
        ]
        
        self.feature_names = features
        return df, features
    
    def train(self, historical_data_dict):
        """Huấn luyện mô hình trên nhiều mã cổ phiếu"""
        print("\n🤖 === BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ML ===")
        all_X = []
        all_y = []
        
        for ticker, df in historical_data_dict.items():
            try:
                df_prep, features = self.prepare_features(df)
                if len(df_prep) > 30:
                    X = df_prep[features].values
                    y = df_prep['target'].values
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                print(f"⚠️ Bỏ qua {ticker}: {e}")
        
        if not all_X:
            print("❌ Không có dữ liệu để huấn luyện")
            return False
        
        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        
        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Chuẩn hóa
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Chọn mô hình
        if self.model_type == 'xgboost' and XGB_AVAILABLE:
            print("📈 Sử dụng mô hình: XGBoost")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            print("📈 Sử dụng mô hình: Random Forest")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                random_state=42
            )
        
        # Huấn luyện
        self.model.fit(X_train_scaled, y_train)
        
        # Đánh giá
        y_pred = self.model.predict(X_test_scaled)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True
        
        print(f"✅ Huấn luyện hoàn tất! Độ chính xác: {self.model_accuracy:.2%}")
        
        # Tầm quan trọng đặc trưng
        if hasattr(self.model, 'feature_importances_'):
            importance = sorted(zip(self.feature_names, self.model.feature_importances_), 
                               key=lambda x: x[1], reverse=True)
            print("📊 Tầm quan trọng chỉ báo:")
            for name, imp in importance[:5]:
                print(f"   {name:<20s}: {imp:.1%}")
        
        self.save_model()
        return True
    
    def predict_probability(self, df):
        """Dự báo xác suất tín hiệu mua hợp lệ"""
        if not self.is_trained or self.model is None:
            return 50.0, "Mô hình chưa huấn luyện"
        
        try:
            df_prep, features = self.prepare_features(df)
            if len(df_prep) < 1:
                return 50.0, "Dữ liệu không đủ"
            
            latest = df_prep[features].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest)
            prob = self.model.predict_proba(latest_scaled)[0, 1] * 100
            
            note = "Xác suất cao ✅" if prob >= 70 else ("Khả quan ⚡" if prob >= 55 else "Thấp ⏳")
            return round(prob, 1), note
        except Exception as e:
            print(f"⚠️ Lỗi dự báo ML: {e}")
            return 50.0, "Lỗi dự báo"
    
    def save_model(self):
        """Lưu mô hình cơ bản"""
        try:
            import pickle
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'accuracy': self.model_accuracy,
                    'trained_at':
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
            print(f"💾 Mô hình đã lưu: Độ chính xác {self.model_accuracy:.2%}")
        except Exception as e:
            print(f"⚠️ Không lưu được mô hình: {e}")
    
    def load_model(self):
        """Tải mô hình đã huấn luyện"""
        if not os.path.exists(MODEL_FILE):
            return False
        try:
            import pickle
            with open(MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.model_accuracy = data.get('accuracy', 0)
                self.is_trained = True
                print(f"✅ Tải mô hình thành công | Độ chính xác: {self.model_accuracy:.2%}")
            return True
        except Exception as e:
            print(f"⚠️ Không tải được mô hình: {e}")
            return False

# ============================================================
# 🔍 TỐI ƯU THAM SỐ — GRID SEARCH
# ============================================================
class ParameterOptimizer:
    """Tìm bộ tham số tốt nhất bằng Grid Search"""
    
    def __init__(self, analyzer, backtest_engine):
        self.analyzer = analyzer
        self.backtest = backtest_engine
        self.best_params = None
        self.best_score = -float('inf')
        self.results = []
    
    def define_search_space(self):
        """Không gian tìm kiếm tham số"""
        return {
            'rsi_low': [48, 50, 52, 55],
            'rsi_high': [68, 70, 72, 75],
            'min_vol_ratio': [1.3, 1.5, 1.7],
            'min_adx': [22, 25, 28],
            'atr_mult_sl': [1.4, 1.5, 1.6, 1.8],
            'reward_ratio': [1.5, 2.0, 2.5]
        }
    
    def evaluate_params(self, params, test_tickers, vnindex_df):
        """Đánh giá một bộ tham số"""
        global RSI_LOW, RSI_HIGH, MIN_VOLUME_RATIO, MIN_ADX, ATR_MULTIPLIER_SL, REWARD_RATIO
        
        # Lưu tham số cũ
        old = {
            'rsi_low': RSI_LOW, 'rsi_high': RSI_HIGH,
            'vol_min': MIN_VOLUME_RATIO, 'adx_min': MIN_ADX,
            'atr_sl': ATR_MULTIPLIER_SL, 'rr': REWARD_RATIO
        }
        
        # Cập nhật tham số mới
        RSI_LOW = params['rsi_low']
        RSI_HIGH = params['rsi_high']
        MIN_VOLUME_RATIO = params['min_vol_ratio']
        MIN_ADX = params['min_adx']
        ATR_MULTIPLIER_SL = params['atr_mult_sl']
        REWARD_RATIO = params['reward_ratio']
        
        # Chạy backtest
        report = self.backtest.run_backtest_all(test_tickers, vnindex_df)
        
        # Khôi phục tham số cũ
        RSI_LOW = old['rsi_low']
        RSI_HIGH = old['rsi_high']
        MIN_VOLUME_RATIO = old['vol_min']
        MIN_ADX = old['adx_min']
        ATR_MULTIPLIER_SL = old['atr_sl']
        REWARD_RATIO = old['rr']
        
        if not report:
            return -999, None
        
        # Hàm mục tiêu: Tỷ lệ thắng * Lợi nhuận - độ biến động
        score = report['win_rate'] * report['total_profit_pct']
        return score, report
    
    def run_grid_search(self, test_tickers, vnindex_df, max_combinations=50):
        """Thực hiện tìm kiếm tham số tối ưu"""
        print("\n" + "=" * 70)
        print("🔍 BẮT ĐẦU TỐI ƯU THAM SỐ — GRID SEARCH")
        print("=" * 70)
        
        space = self.define_search_space()
        from itertools import product
        
        keys = list(space.keys())
        values = list(space.values())
        combinations = list(product(*values))
        
        # Giới hạn số lượng tổ hợp để không quá lâu
        if len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)
        
        print(f"📊 Tổng số bộ tham số cần kiểm tra: {len(combinations)}")
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            score, report = self.evaluate_params(params, test_tickers, vnindex_df)
            
            result = {
                'params': params,
                'score': score,
                'win_rate': report['win_rate'] if report else 0,
                'profit': report['total_profit_pct'] if report else 0
            }
            self.results.append(result)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"  ✅ [{i}/{len(combinations)}] MỚI TỐI ƯU | Score: {score:.2f} | Win%: {result['win_rate']:.1%} | Lợi nhuận: {result['profit']:+.2f}%")
            else:
                if i % 10 == 0:
                    print(f"  ⏳ [{i}/{len(combinations)}] Đang kiểm tra...")
            
            time.sleep(0.2)
        
        # Sắp xếp kết quả
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n" + "=" * 70)
        print("🏆 KẾT QUẢ TỐI ƯU THAM SỐ — TOP 5 BỐ THAM SỐ")
        print("=" * 70)
        for rank, r in enumerate(self.results[:5], 1):
            print(f"#{rank} Score: {r['score']:.2f} | Win%: {r['win_rate']:.1%} | Lợi nhuận: {r['profit']:+.2f}%")
            print(f"   {r['params']}")
        
        print("\n🏆 BỘ THAM SỐ TỐI ƯU NHẤT ĐƯỢC CHỌN:")
        print(f"   {self.best_params}")
        print("=" * 70 + "\n")
        
        return self.best_params

# ============================================================
# 📊 LỚP PHÂN TÍCH KỸ THUẬT NÂNG CAO
# ============================================================
class TechnicalAnalyzerV2:
    def __init__(self):
        self.divergence = DivergenceDetector()
        self.timeframe = MultiTimeframeConfirm()
        self.risk = RiskManager()
        self.ml = StockMLPredictor(model_type='xgboost' if XGB_AVAILABLE else 'rf')
    
    def calculate_indicators(self, df):
        df = df.copy()
        
        # EMA
        df['ema20'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=EMA_MID, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
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
        
        # ADX
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
        tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_wilder = tr_series.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / atr_wilder
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / atr_wilder
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Khối lượng
        df['avg_vol_20'] = df['volume'].rolling(window=VOL_LOOKBACK).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']
        
        # Đỉnh giá
        df['high_20d'] = df['high'].rolling(window=BREAKOUT_PERIOD).max()
        df['pct_from_high_20d'] = (df['high_20d'] - df['close']) / df['high_20d']
        
        return df
    
    def generate_signal(self, df, ticker, sector, fundamental_data, vnindex_df=None, market_ok=True):
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        entry = round(float(latest['close']), 2)
        
        # === 1. P/E ngành động ===
        pe_min_sec, pe_max_sec, pe_avg_sec = pe_manager.get_pe_range(sector)
        pe_ok = True
        pe_note = ""
        if fundamental_data['pe'] > 0:
            if pe_min_sec <= fundamental_data['pe'] <= pe_max_sec:
                pe_note = f"✅ P/E {fundamental_data['pe']:.1f} trong khoảng ngành {pe_min_sec:.1f}-{pe_max_sec:.1f} (TB: {pe_avg_sec:.1f})"
            else:
                pe_ok = False
                pe_note = f"⚠️ P/E {fundamental_data['pe']:.1f} ngoài khoảng ngành {pe_min_sec:.1f}-{pe_max_sec:.1f}"
        else:
            pe_note = "⚠️ Không có dữ liệu P/E"
        
        # === 2. Phân kỳ RSI/MACD ===
        div = self.divergence.detect_all(df)
        
        # === 3. Xác nhận đa khung thời gian ===
        mtf = self.timeframe.get_confirmation(df)
        
        # === 4. Dự báo ML ===
        ml_prob, ml_note = self.ml.predict_probability(df)
        
        # === 5. Tính điểm tổng hợp ===
        sector_params = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Khác"])
        rsi_low = sector_params.get("rsi_low", RSI_LOW)
        rsi_high = sector_params.get("rsi_high", RSI_HIGH)
        vol_min = sector_params.get("vol_min", MIN_VOLUME_RATIO)
        adx_min = sector_params.get("adx_min", MIN_ADX)
        atr_mult_sl = sector_params.get("atr_mult", 1.5)
        
        score = 0
        checks = {}
        
        # EMA Ribbon — 20 điểm
        ema_ok = latest['ema20'] > latest['ema50'] and latest['ema50'] > latest['ema200']
        price_ema_ok = latest['close'] > latest['ema20']
        if ema_ok and price_ema_ok:
            score += 20
            checks['ema'] = "✅ Hoàn hảo"
        elif price_ema_ok:
            score += 10
            checks['ema'] = "✅ Tốt"
        else:
            checks['ema'] = "❌ Yếu"
        
        # Khối lượng — 15 điểm
        if latest['vol_ratio'] >= vol_min * 1.2:
            score += 15
        elif latest['vol_ratio'] >= vol_min:
            score += 10
        checks['vol_ratio'] = round(float(latest['vol_ratio']), 2)
        
        # ADX + DI — 15 điểm
        if latest['adx'] > adx_min and latest['plus_di'] > latest['minus_di']:
            score += 15
        elif latest['adx'] > adx_min * 0.8:
            score += 7
        checks['adx'] = round(float(latest['adx']), 1)
        
        # RSI — 10 điểm
        if rsi_low < latest['rsi'] < rsi_high:
            score += 10
        checks['rsi'] = round(float(latest['rsi']), 1)
        
        # MACD — 10 điểm
        if latest['macd'] > latest['macd_signal']:
            score += 10
        checks['macd'] = "✅ Tăng" if latest['macd'] > latest['macd_signal'] else "❌ Giảm"
        
        # Phân kỳ dương — +10 điểm
        if div['has_bullish']:
            score += 10
            checks['divergence'] = div['strongest_bullish']['note']
        elif div['has_bearish']:
            score -= 10
            checks['divergence'] = div['strongest_bearish']['note']
        else:
            checks['divergence'] = "Không có phân kỳ đáng kể"
        
        # Xác nhận đa khung — +15 điểm
        if mtf['strong_bullish']:
            score += 15
        elif mtf['weak_bullish']:
            score += 8
        checks['mtf'] = mtf['level']
        
        # P/E ngành — +5 điểm
        if pe_ok:
            score += 5
        checks['pe_ok'] = pe_ok
        
        # Xác suất ML — điều chỉnh điểm
        ml_bonus = (ml_prob - 50) / 5  # +/- 10 điểm tối đa
        score = max(0, min(100, score + ml_bonus))
        checks['ml_prob'] = ml_prob
        
        # Điều chỉnh theo thị trường
        if not market_ok:
            score = int(score * 0.7)
        
        # === 6. Tính SL/TP động theo ATR ===
        atr_val = float(latest['atr'])
        sl_atr = round(entry - atr_val * atr_mult_sl, 2)
        tp_atr = round(entry + atr_val * 2.5, 2)
        sl_fixed = round(entry * 0.975, 2)
        tp_fixed = round(entry * 1.0625, 2)
        
        # === 7. Tín hiệu cuối cùng ===
        confidence = score / 100 * mtf['confidence'] / 100 * (ml_prob / 100)
        signal = "MUA ✅" if (score >= 65 and mtf['weak_bullish'] and ml_prob >= 55 and market_ok and pe_ok) else \
                 "XEM XÉT ⚡" if score >= 50 else "THEO DÕI ⏳"
        
        return {
            "ticker": ticker,
            "sector": sector,
            "signal": signal,
            "score": score,
            "confidence": round(confidence * 100, 1),
            "entry": entry,
            "SL_atr": sl_atr,
            "TP_atr": tp_atr,
            "SL_fixed": sl_fixed,
            "TP_fixed": tp_fixed,
            "rsi": checks['rsi'],
            "adx": checks['adx'],
            "vol_ratio": checks['vol_ratio'],
            "ema_status": checks['ema'],
            "macd_status": checks['macd'],
            "divergence": checks['divergence'],
            "mtf_status": checks['mtf'],
            "pe_note": pe_note,
            "pe_ok": pe_ok,
            "ml_prob": ml_prob,
            "ml_note": ml_note,
            "checks": checks
        }

# ============================================================
# 🚀 CHƯƠNG TRÌNH CHÍNH
# ============================================================
class SmartStockBotV2:
    def __init__(self, run_ml_train=False, run_optimize=False):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzerV2()
        self.risk = self.analyzer.risk
        self.watchlist = []
        self.vnindex_data = None
        
        # Cập nhật P/E ngành
        pe_manager.update_from_market()
        
        # Huấn luyện ML nếu yêu cầu
        if run_ml_train:
            self.train_ml_model()
        
        # Tối ưu tham số nếu yêu cầu
        self.run_optimize = run_optimize
    
    def train_ml_model(self):
        """Thu thập dữ liệu và huấn luyện mô hình ML"""
        print("\n🤖 === BẮT ĐẦU THU THẬP DỮ LIỆU HUẤN LUYỆN ===")
        tickers = list(set(list(STOCK_SECTOR.keys())))
        historical_data = {}
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Tải dữ liệu {ticker}...", end=" ")
            df = get_stock_data(ticker, days=BACKTEST_DAYS + 120)
            if df is not None and len(df) >= 100:
                historical_data[ticker] = df
                print(f"✅ {len(df)} phiên")
            else:
                print("⚠️ Bỏ qua")
            time.sleep(0.3)
        
        if historical_data:
            self.analyzer.ml.train(historical_data)
        else:
            print("❌ Không có dữ liệu đủ để huấn luyện")
    
    def build_watchlist(self):
        print("\n📊 === PHÂN TÍCH VNINDEX ===")
        self.vnindex_data = get_vnindex_data()
        market_ok = True
        if self.vnindex_data:
            print(f"   VNINDEX: {self.vnindex_data['close']:.2f} | Xu hướng: {self.vnindex_data['trend']} | RSI: {self.vnindex_data['rsi']:.1f}")
            market_ok = self.vnindex_data['market_ok']
            if not market_ok:
                print("⚠️ Thị trường không thuận lợi — điểm tín hiệu giảm 30%")
            self.notifier.market_summary(self.vnindex_data)
        
        candidates = get_filtered_hose_stocks()
        results = []
        print(f"\n⏳ Đang phân tích {len(candidates)} mã...")
        print("-" * 90)
        
        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None
        
        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i:2d}/{len(candidates)}] {ticker:<6s}...", end=" ", flush=True)
            
            df = get_stock_data(ticker)
            if df is None:
                print("❌ Không có dữ liệu")
                continue
            
            sector = STOCK_SECTOR.get(ticker, "Khác")
            fund = FundamentalData.get_fundamental_data(ticker)
            
            sig = self.analyzer.generate_signal(df, ticker, sector, fund, vnindex_df, market_ok)
            results.append(sig)
            
            print(f"Điểm: {sig['score']:3d}/100 | {sig['signal']} | ML: {sig['ml_prob']:.0f}% | {sig['mtf_status']}")
            print(f"       {sig['pe_note']}")
            print(f"       Phân kỳ: {sig['divergence']}")
            print(f"       Giá: {sig['entry']} | SL: {sig['SL_atr']} | TP: {sig['TP_atr']}")
            time.sleep(0.5)
        
        if not results:
            print("❌ Không có kết quả phân tích!")
            return []
        
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]
        
        print("\n" + "=" * 90)
        print(f"🏆 TOP {len(top_stocks)} MÃ — Sắp xếp theo Độ tin cậy tổng hợp")
        print("=" * 90)
        for rank, s in enumerate(top_stocks, 1):
            print(f"#{rank} {s['ticker']:<6s} [{s['sector'][:15]}] | Điểm: {s['score']}/100 | Độ tin cậy: {s['confidence']}%")
            print(f"       Tín hiệu: {s['signal']} | Xác suất ML: {s['ml_prob']:.0f}% | Khung: {s['mtf_status']}")
            print(f"       Giá vào: {s['entry']} | SL(ATR): {s['SL_atr']} | TP(ATR): {s['TP_atr']}")
            print(f"       P/E: {s['pe_note']}")
        print("=" * 90 + "\n")
        
        # Gửi thông báo
        self.notifier.top_stocks(top_stocks, self.vnindex_data)
        
        # Tính toán quản lý vốn cho từng tín hiệu MUA
        for s in top_stocks:
            if s['signal'] == "MUA ✅":
                shares, note = self.risk.calculate_position_size(
                    s['entry'], s['SL_atr'], confidence=s['confidence']/100
                )
                s['shares_suggest'] = shares
                s['position_note'] = note
                self.notifier.new_signal(
                    s['ticker'], s['sector'], s['entry'],
                    s['TP_atr'], s['SL_fixed'], s['SL_atr'],
                    s['score'], s
                )
        
        # Tối ưu tham số nếu yêu cầu
        if self.run_optimize and len(top_stocks) >= 3 and vnindex_df is not None:
            print("\n🔄 Chạy tối ưu tham số...")
            opt = ParameterOptimizer(self.analyzer, self.analyzer.backtest)
            best = opt.run_grid_search([s['ticker'] for s in top_stocks[:5]], vnindex_df, max_combinations=30)
            if best:
                print(f"✅ Tham số tối ưu có sẵn để áp dụng: {best}")
        
        return top_stocks

# ============================================================
# 🏃 KHỞI ĐỘNG CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("🚀 BOT CHỨNG KHOÁN — PHIÊN BẢN 2.0 NÂNG CẤP HOÀN CHỈNH")
    print("=" * 90)
    print("📊 Nguồn dữ liệu: API VNDIRECT")
    print("🤖 Tính năng: Phân kỳ RSI/MACD | P/E ngành động | Quản lý vốn thông minh")
    print("✅ Chi phí thực tế (hoa hồng + trượt giá) | Xác nhận đa khung D1/W1/M1")
    print("🧠 ML: XGBoost/RF dự báo xác suất | Grid Search tối ưu tham số")
    print("⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ")
    print("=" * 90 + "\n")
    
    # Chọn chế độ chạy
    TRAIN_MODEL = True      # Đặt =True để huấn luyện lại mô hình ML
    OPTIMIZE_PARAMS = False   # Đặt =True để chạy tối ưu tham số
    
    bot = SmartStockBotV2(run_ml_train=TRAIN_MODEL, run_optimize=OPTIMIZE_PARAMS)
    
    bot.notifier.send_embed(
        "✅ BOT V2.0 ĐÃ ONLINE — NÂNG CẤP HOÀN TẤT",
        f"""📊 Phân kỳ RSI/MACD: Đã kích hoạt
🏭 P/E ngành động: Đã cập nhật
💰 Quản lý vốn: {TOTAL_CAPITAL:,.0f}đ | Max {MAX_POSITION_PERCENT}%/mã | Rủi ro {RISK_PER_TRADE_PERCENT}%/giao dịch
💱 Chi phí thực tế: HH {COMMISSION_RATE*100:.2f}% + Trượt {Slippage_RATE*100:.2f}% + Thuế {STAMP_DUTY_RATE*100:.2f}%
⏳ Xác nhận đa khung D1/W1/M1: Đã bật
🧠 Mô hình ML: {'Đã tải' if bot.analyzer.ml.is_trained else 'Chưa huấn luyện'}
🔍 Tối ưu tham số: {'Bật' if OPTIMIZE_PARAMS else 'Tắt'}
⏰ Quét: 8:00 sáng T2-T6
⚠️ KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ — Chỉ tham khảo phân tích kỹ thuật""",
        0x2ecc71
    )
    
    print("⏳ Quét lần đầu...\n")
    bot.build_watchlist()
    
    last_scan_date = None
    print("\n⏳ Bot đang chạy nền — sẽ quét mới vào 8:00 sáng các ngày giao dịch...")
    
    while True:
        time.sleep(60)
        now = datetime.now()
        is_weekday = now.weekday() < 5
        is_8am = now.hour == 8 and now.minute <= 5
        today_str = now.strftime("%Y-%m-%d")
        
        if is_weekday and is_8am and last_scan_date != today_str:
            print("\n" + "=" * 90)
            print(f"🕐 {now.strftime('%A %d/%m/%Y %H:%M')} — QUÉT DANH SÁCH MỚI")
            print("=" * 90)
            bot.build_watchlist()
            last_scan_date = today_str
            print("✅ Hoàn thành. Chờ đến sáng mai...\n")


