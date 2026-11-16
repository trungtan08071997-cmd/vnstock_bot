# ============================================================
# 🇻🇳 BOT LỌC TỰ ĐỘNG & PHÂN TÍCH — CHỌN 5-10 MÃ TỐT NHẤT HOSE
# Lọc theo: Vốn hóa + Thanh khoản + Tín hiệu kỹ thuật
# ⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time
from dotenv import load_dotenv
from vnstock.stock import stock_historical_data
from vnstock.listing import listing_companies

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🔧 CẤU HÌNH
# ============================================================
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Tham số chỉ báo — Tối ưu cho thị trường VN ===
RSI_PERIOD = 9
MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 7
SMA_SHORT = 10
SMA_LONG = 40

# === Tiêu chí LỌC SÀN — Tùy chỉnh theo nhu cầu ===
MIN_MARKET_CAP_BN = 1000    # Vốn hóa tối thiểu: 1.000 tỷ đồng
MIN_VOLUME_MIL = 500        # Thanh khoản trung bình tối thiểu: 500 triệu/phiên
TOP_STOCKS_LIMIT = 8        # Chọn ra N mã tốt nhất để theo dõi

# === Quản trị rủi ro ===
RISK_PERCENT = 2.5
REWARD_RATIO = 1.5

# === Số ngày dữ liệu lịch sử cần lấy ===
LOOKBACK_DAYS = 120

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
            "footer": {"text": "📢 Hệ thống lọc tự động — Thị trường Việt Nam"}
        }
        try:
            resp = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            return resp.status_code == 204
        except Exception as e:
            print(f"⚠️ Lỗi Discord: {e}")
            return False

    def new_signal(self, ticker, direction, entry, tp, sl, score):
        emoji = "🟢" if direction == "MUA" else "🔴"
        color = 0x2ecc71 if direction == "MUA" else 0xe67e22
        title = f"{emoji} TÍN HIỆU MỚI — {direction} {ticker}"
        desc = f"""
📌 **Mã:** {ticker}
📊 **Điểm tín hiệu:** {score}/100
💰 **Giá vào:** {entry}
🎯 **Mức TP:** {tp}
🛑 **Mức SL:** {sl}
📐 **Tỷ lệ L/R:** 1 : {REWARD_RATIO}
        """
        return self.send_embed(title, desc, color)

    def watchlist_update(self, count, stocks_list):
        desc = f"""
✅ Đã lọc xong sàn HOSE
📊 Tiêu chí: Vốn hóa ≥ {MIN_MARKET_CAP_BN} tỷ | Thanh khoản ≥ {MIN_VOLUME_MIL}tr
📋 Số mã đủ điều kiện: {count}
📈 Danh sách theo dõi mới: {', '.join(stocks_list)}
        """
        return self.send_embed("📋 CẬP NHẬT DANH SÁCH THEO DÕI", desc, 0x9b59b6)

# ============================================================
# 📊 LỚP PHÂN TÍCH KỸ THUẬT & TÍNH ĐIỂM
# ============================================================
class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_all_indicators(self, df):
        df = df.copy()
        df[f'sma{SMA_SHORT}'] = df['close'].rolling(window=SMA_SHORT).mean()
        df[f'sma{SMA_LONG}'] = df['close'].rolling(window=SMA_LONG).mean()
        df['ema12'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()

        df['bb_mid'] = df['close'].rolling(window=18).mean()
        bb_std = df['close'].rolling(window=18).std()
        df['bb_upper'] = df['bb_mid'] + 1.8 * bb_std
        df['bb_lower'] = df['bb_mid'] - 1.8 * bb_std

        return df

    def calculate_score(self, df):
        """Tính điểm tổng hợp 0-100 — càng cao càng tốt"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0

        # 1. Xu hướng giá trên SMA (25 điểm)
        if latest['close'] > latest[f'sma{SMA_SHORT}']: score += 12
        if latest[f'sma{SMA_SHORT}'] > latest[f'sma{SMA_LONG}']: score += 13

        # 2. RSI hợp lý (25 điểm)
        if 45 < latest['rsi'] < 70:
            score += 25
        elif 35 < latest['rsi'] < 75:
            score += 15

        # 3. MACD (25 điểm)
        if latest['macd'] > latest['macd_signal']:
            score += 15
            if prev['macd'] <= prev['macd_signal']:
                score += 10  # Vừa cắt lên

        # 4. Vị trí so với Bollinger Bands (15 điểm)
        if latest['close'] > latest['bb_mid']:
            score += 15

        # 5. Khối lượng tăng (10 điểm)
        avg_vol = df['volume'].tail(10).mean()
        if latest['volume'] > avg_vol * 1.1:
            score += 10

        return score

    def generate_signal(self, df, ticker):
        df = self.calculate_all_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        entry_price = round(latest['close'], 2)
        score = self.calculate_score(df)

        sell_signal = (
            entry_price < latest[f'sma{SMA_SHORT}'] and
            30 < latest['rsi'] < 50 and
            latest['macd'] < latest['macd_signal'] and
            prev['macd'] >= prev['macd_signal'] and
            entry_price < prev['close']
        )

        buy_signal = (
            entry_price > latest[f'sma{SMA_SHORT}'] and
            50 < latest['rsi'] < 70 and
            latest['macd'] > latest['macd_signal'] and
            prev['macd'] <= prev['macd_signal'] and
            entry_price > prev['close']
        )

        risk = RISK_PERCENT / 100

        if sell_signal:
            direction = "BÁN"
            stop_loss = round(entry_price * (1 + risk), 2)
            risk_amount = stop_loss - entry_price
            take_profit = round(entry_price - risk_amount * REWARD_RATIO, 2)
        elif buy_signal:
            direction = "MUA"
            stop_loss = round(entry_price * (1 - risk), 2)
            risk_amount = entry_price - stop_loss
            take_profit = round(entry_price + risk_amount * REWARD_RATIO, 2)
        else:
            direction = "KHÔNG CÓ"
            stop_loss = round(entry_price * 0.975, 2)
            take_profit = round(entry_price * 1.0375, 2)

        return {
            "ticker": ticker,
            "signal": direction,
            "score": score,
            "entry": entry_price,
            "TP": take_profit,
            "SL": stop_loss
        }

# ============================================================
# 🔍 LỌC DANH SÁCH MÃ THEO VỐN HÓA & THANH KHOẢN
# ============================================================
def get_filtered_hose_stocks():
    """
    Bước 1: Lấy toàn bộ danh sách HOSE
    Bước 2: Lọc theo vốn hóa và thanh khoản
    Bước 3: Trả về danh sách mã đủ điều kiện
    """
    try:
        print("\n" + "="*60)
        print("🔍 ĐANG LỌC DANH SÁCH SÀN HOSE...")
        print(f"📋 Tiêu chí: Vốn hóa ≥ {MIN_MARKET_CAP_BN} tỷ | Thanh khoản ≥ {MIN_VOLUME_MIL}tr/phiên")
        print("="*60)

        # Lấy danh sách toàn bộ
        df_all = listing_companies()
        
        # Lọc chỉ lấy HOSE
        df_hose = df_all[df_all['exchange'] == 'HOSE'].copy()
        print(f"✅ Tổng số mã trên HOSE: {len(df_hose)}")

        # Lọc theo vốn hóa & thanh khoản (nếu có cột dữ liệu)
        if 'market_cap' in df_hose.columns:
            df_hose = df_hose[df_hose['market_cap'] >= MIN_MARKET_CAP_BN]
        if 'volume' in df_hose.columns:
            df_hose = df_hose[df_hose['avg_volume'] >= MIN_VOLUME_MIL * 1000]

        symbols = df_hose['symbol'].tolist()
        print(f"✅ Số mã đủ điều kiện lọc: {len(symbols)}")
        
        # Giới hạn số lượng để không chạy quá lâu
        if len(symbols) > 150:
            symbols = symbols[:150]
            print(f"ℹ️ Giới hạn quét 150 mã đầu để tối ưu tốc độ")

        return symbols

    except Exception as e:
        print(f"⚠️ Lỗi lấy danh sách: {e}")
        print("ℹ️ Sử dụng danh sách dự phòng...")
        return ["VNM", "ACB", "MWG", "BMP", "DHG", "MBB", "HDB", "GAS", "VIC", "TPB"]

# ============================================================
# 📥 LẤY DỮ LIỆU GIÁ TỪ VNSTOCK
# ============================================================
def get_stock_data(ticker, days=LOOKBACK_DAYS):
    """Lấy dữ liệu giá THẬT từ sàn HOSE"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = stock_historical_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1D"
        )
        
        if df is None or len(df) < 50:
            return None
            
        return df
    except Exception as e:
        return None

# ============================================================
# 🚀 HỆ THỐNG CHÍNH
# ============================================================
class SmartStockBot:
    def __init__(self):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.watchlist = []
        self.positions = {}

    def build_watchlist(self):
        """Lọc và chọn ra TOP mã tốt nhất"""
        candidates = get_filtered_hose_stocks()
        results = []

        print(f"\n⏳ Đang phân tích {len(candidates)} mã... (có thể mất vài phút)")
        
        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i}/{len(candidates)}] {ticker}...", end=" ")
            df = get_stock_data(ticker)
            
            if df is not None:
                analysis = self.analyzer.generate_signal(df, ticker)
                results.append(analysis)
                print(f"Điểm: {analysis['score']} | {analysis['signal']}")
            else:
                print("Bỏ qua")
            
            time.sleep(0.8)  # Tránh bị chặn API

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        # Sắp xếp theo điểm tín hiệu giảm dần
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Chọn TOP mã có điểm cao nhất
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        # In ra danh sách TOP
        print("\n" + "="*60)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ TÍN HIỆU TỐT NHẤT")
        print("="*60)
        for rank, stock in enumerate(top_stocks, 1):
            print(f"#{rank} {stock['ticker']:6s} | Điểm: {stock['score']:3d}/100 | {stock['signal']} | Giá: {stock['entry']}")
        print("="*60 + "\n")

        # Gửi thông báo Discord
        self.notifier.watchlist_update(len(results), self.watchlist)
        
        return top_stocks

    def monitor_watchlist(self):
        """Theo dõi danh sách đã chọn"""
        print("\n" + "="*60)
        print("📊 QUÉT TÍN HIỆU DANH SÁCH THEO DÕI")
        print("="*60 + "\n")

        for ticker in self.watchlist:
            df = get_stock_data(ticker)
            if df is not None:
                result = self.analyzer.generate_signal(df, ticker)
                
                if result['signal'] in ['MUA', 'BÁN']:
                    print(f"✅ {ticker}: {result['signal']} | Vào: {result['entry']} | TP: {result['TP']} | SL: {result['SL']}")
                    self.notifier.new_signal(
                        ticker, result['signal'],
                        result['entry'], result['TP'], result['SL'],
                        result['score']
                    )
                else:
                    print(f"⏳ {ticker}: Điểm {result['score']}/100 — Chờ tín hiệu")
            time.sleep(0.5)

# ============================================================
# 🏃 CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BOT LỌC TỰ ĐỘNG HOSE — CHỌN TOP MÃ TỐT NHẤT")
    print("="*60)
    print(f"⚙️  Vốn hóa tối thiểu: {MIN_MARKET_CAP_BN} tỷ")
    print(f"⚙️  Thanh khoản tối thiểu: {MIN_VOLUME_MIL}tr/phiên")
    print(f"⚙️  Số mã chọn ra: TOP {TOP_STOCKS_LIMIT}")
    print("="*60 + "\n")

    # Khởi tạo Bot
    bot = SmartStockBot()

    # Gửi xác nhận online
    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — LỌC TỰ ĐỘNG HOSE",
        f"📋 Tiêu chí: Vốn hóa ≥ {MIN_MARKET_CAP_BN} tỷ\n📊 Thanh khoản ≥ {MIN_VOLUME_MIL}tr\n🎯 Sẽ chọn TOP {TOP_STOCKS_LIMIT} mã tốt nhất",
        0x2ecc71
    )

    # Bước 1: Lọc & chọn danh sách tốt nhất — chạy 1 lần lúc khởi động
    top_stocks = bot.build_watchlist()

    # Bước 2: Theo dõi liên tục danh sách đã chọn
    print("\n⏳ Bot đang theo dõi danh sách... Quét lại mỗi giờ")
    last_monitor = datetime.now()
    
    while True:
        time.sleep(60)  # Kiểm tra mỗi phút
        elapsed = (datetime.now() - last_monitor).total_seconds()
        
        if elapsed >= 3600:  # Quét lại mỗi giờ
            print("\n" + "-"*60)
            print(f"🕐 CẬP NHẬT TÍN HIỆU — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*60)
            bot.monitor_watchlist()
            last_monitor = datetime.now()
