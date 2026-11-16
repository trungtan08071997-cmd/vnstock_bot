# ============================================================
# HỆ THỐNG GIAO DỊCH — TỐI ƯU CHO THỊ TRƯỜNG VIỆT NAM
# Triển khai trên Render + GitHub
# ⚠️ CHỈ DÙNG CHO HỌC TẬP — KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🔧 CẤU HÌNH — Đọc từ biến môi trường (an toàn trên Render)
# ============================================================
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Tham số chỉ báo — TỐI ƯU CHO THỊ TRƯỜNG VN ===
RSI_PERIOD = 9
MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 7
SMA_SHORT = 10
SMA_LONG = 40
BB_PERIOD = 18
BB_STD = 1.8

# === Quản trị rủi ro ===
RISK_PERCENT = 2.5
REWARD_RATIO = 1.5
MARKET_ALERT_THRESHOLD = 3.0

# === Danh mục theo dõi ===
WATCHLIST = ["VNM", "ACB", "VIC", "MWG", "FRT"]

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
            "footer": {"text": "📢 Hệ thống theo dõi — Thị trường Việt Nam"}
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

    def new_signal(self, ticker, direction, entry, tp, sl):
        emoji = "🟢" if direction == "MUA" else "🔴"
        color = 0x2ecc71 if direction == "MUA" else 0xe67e22
        title = f"{emoji} TÍN HIỆU MỚI — {direction} {ticker}"
        desc = f"""
📌 **Mã:** {ticker}
📊 **Hướng:** {direction}
💰 **Giá vào:** {entry}
🎯 **Mức TP:** {tp}
🛑 **Mức SL:** {sl}
📐 **Tỷ lệ L/R:** 1 : {REWARD_RATIO}
        """
        return self.send_embed(title, desc, color)

    def sl_triggered(self, ticker, direction, entry, sl, current, loss_pct):
        title = "🚨 🔴 KÍCH HOẠT DỪNG LỖ — ĐÓNG VỊ THẾ NGAY!"
        desc = f"""
🔴 **Mã:** {ticker}
📌 **Hướng:** {direction}
💰 **Giá vào:** {entry}
🛑 **Mức SL:** {sl}
📊 **Giá hiện tại:** {current}
📉 **Thua lỗ:** {abs(loss_pct):.2f}%

> ⚠️ **HÀNH ĐỘNG:** Đóng vị thế ngay! Không đỡ đỉnh!
        """
        return self.send_embed(title, desc, 0xe74c3c)

    def tp_hit(self, ticker, direction, entry, tp, current, profit_pct):
        title = "✅ 🟢 ĐẠT MỤC TIÊU — CHỐT LỜI!"
        desc = f"""
🟢 **Mã:** {ticker}
📌 **Hướng:** {direction}
💰 **Giá vào:** {entry}
🎯 **Mức TP:** {tp}
📊 **Giá hiện tại:** {current}
📈 **Lợi nhuận:** {abs(profit_pct):.2f}%

> 🎉 **HÀNH ĐỘNG:** Chốt lời hoặc di chuyển SL lên giá vào lệnh
        """
        return self.send_embed(title, desc, 0x2ecc71)

    def market_alert(self, index_name, change_pct, current_value):
        if change_pct < 0:
            title = "📉 ⚠️ THỊ TRƯỜNG GIẢM MẠNH!"
            color = 0xe74c3c
            desc = f"""
🔻 **{index_name}:** {current_value:.2f}
📊 **Thay đổi:** {change_pct:.2f}%

> 🔔 Kiểm tra ngay danh mục! Hạn chế mở lệnh mới.
            """
        else:
            title = "📈 ⚠️ THỊ TRƯỜNG TĂNG MẠNH!"
            color = 0x2ecc71
            desc = f"""
🔺 **{index_name}:** {current_value:.2f}
📊 **Thay đổi:** +{change_pct:.2f}%

> 🔔 Cẩn trọng chốt lời một phần vị thế đang có lãi.
            """
        return self.send_embed(title, desc, color)

# ============================================================
# 📊 LỚP PHÂN TÍCH KỸ THUẬT & TÍN HIỆU
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

        df['bb_mid'] = df['close'].rolling(window=BB_PERIOD).mean()
        bb_std = df['close'].rolling(window=BB_PERIOD).std()
        df['bb_upper'] = df['bb_mid'] + BB_STD * bb_std
        df['bb_lower'] = df['bb_mid'] - BB_STD * bb_std

        return df

    def generate_signal(self, df, ticker):
        df = self.calculate_all_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        entry_price = round(latest['close'], 2)

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
            return {"ticker": ticker, "signal": "KHÔNG CÓ", "entry": entry_price, "note": "Chờ tín hiệu rõ hơn"}

        return {"ticker": ticker, "signal": direction, "entry": entry_price, "TP": take_profit, "SL": stop_loss}

# ============================================================
# 🚀 HỆ THỐNG CHÍNH
# ============================================================
class TradingBot:
    def __init__(self):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.positions = {}
        self.market_ref = {}

    def scan_ticker(self, df, ticker, send_alert=True):
        result = self.analyzer.generate_signal(df, ticker)
        if result['signal'] in ['MUA', 'BÁN']:
            self.positions[ticker] = {
                "direction": result['signal'],
                "entry": result['entry'],
                "SL": result['SL'],
                "TP": result['TP']
            }
            if send_alert:
                self.notifier.new_signal(ticker, result['signal'], result['entry'], result['TP'], result['SL'])
            print(f"✅ {ticker}: {result['signal']} | Vào: {result['entry']} | TP: {result['TP']} | SL: {result['SL']}")
        else:
            print(f"⏳ {ticker}: {result['note']}")
        return result

    def scan_watchlist(self, data_dict):
        print("\n" + "="*60)
        print("📊 QUÉT TÍN HIỆU DANH SÁCH THEO DÕI")
        print("="*60 + "\n")
        for ticker, df in data_dict.items():
            self.scan_ticker(df, ticker)

    def check_price(self, ticker, current_price):
        if ticker not in self.positions:
            return f"ℹ️ Không có vị thế {ticker}"
        pos = self.positions[ticker]
        entry = pos['entry']
        sl = pos['SL']
        tp = pos['TP']
        direction = pos['direction']

        if direction == "BÁN" and current_price >= sl:
            loss_pct = (current_price - entry) / entry * 100
            self.notifier.sl_triggered(ticker, direction, entry, sl, current_price, loss_pct)
            del self.positions[ticker]
            return f"🔴 {ticker}: DỪNG LỖ"
        if direction == "MUA" and current_price <= sl:
            loss_pct = (entry - current_price) / entry * 100
            self.notifier.sl_triggered(ticker, direction, entry, sl, current_price, loss_pct)
            del self.positions[ticker]
            return f"🔴 {ticker}: DỪNG LỖ"
        if direction == "BÁN" and current_price <= tp:
            profit_pct = (entry - current_price) / entry * 100
            self.notifier.tp_hit(ticker, direction, entry, tp, current_price, profit_pct)
            del self.positions[ticker]
            return f"🟢 {ticker}: CHỐT LỜI"
        if direction == "MUA" and current_price >= tp:
            profit_pct = (current_price - entry) / entry * 100
            self.notifier.tp_hit(ticker, direction, entry, tp, current_price, profit_pct)
            del self.positions[ticker]
            return f"🟢 {ticker}: CHỐT LỜI"
        return f"✅ {ticker} an toàn | Giá: {current_price} | SL: {sl} | TP: {tp}"

    def monitor_index(self, index_name, current_price, ref_price=None):
        if ref_price is None:
            if index_name not in self.market_ref:
                self.market_ref[index_name] = current_price
                return f"📌 Lưu tham chiếu {index_name}: {current_price:.2f}"
            ref_price = self.market_ref[index_name]
        change_pct = (current_price - ref_price) / ref_price * 100
        if abs(change_pct) >= MARKET_ALERT_THRESHOLD:
            self.notifier.market_alert(index_name, change_pct, current_price)
            return f"⚠️ {index_name} thay đổi {change_pct:.2f}% — Đã gửi Discord"
        return f"✅ {index_name}: {current_price:.2f} ({change_pct:+.2f}%)"

# ============================================================
# 🧪 DỮ LIỆU MẪU — THAY BẰNG DỮ LIỆU THỰC
# ============================================================
def create_sample_data(ticker, days=90):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(hash(ticker) % 10000)
    base = np.random.uniform(30, 150)
    prices = base + np.cumsum(np.random.randn(len(dates)) * 0.8)
    return pd.DataFrame({
        'open': prices + np.random.randn(len(dates)) * 0.2,
        'high': prices + abs(np.random.randn(len(dates))) * 0.6,
        'low': prices - abs(np.random.randn(len(dates))) * 0.6,
        'close': prices,
        'volume': np.random.randint(500000, 3000000, size=len(dates))
    }, index=dates)

# ============================================================
# 🏃 CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 KHỞI ĐỘNG BOT THEO DÕI THỊ TRƯỜNG")
    print("="*60)
    print(f"📋 Danh sách theo dõi: {', '.join(WATCHLIST)}")
    print(f"⚙️  RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}")
    print(f"🛑 Rủi ro: {RISK_PERCENT}% | Tỷ lệ L/R: 1:{REWARD_RATIO}")
    print("="*60 + "\n")

    # Tạo dữ liệu mẫu
    def create_sample_data(ticker, days=90):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(hash(ticker) % 10000)
        base = np.random.uniform(30, 150)
        prices = base + np.cumsum(np.random.randn(len(dates)) * 0.8)
        return pd.DataFrame({
            'open': prices + np.random.randn(len(dates)) * 0.2,
            'high': prices + abs(np.random.randn(len(dates))) * 0.6,
            'low': prices - abs(np.random.randn(len(dates))) * 0.6,
            'close': prices,
            'volume': np.random.randint(500000, 3000000, size=len(dates))
        }, index=dates)

    # Khởi tạo dữ liệu mẫu cho toàn bộ danh sách
    market_data = {ticker: create_sample_data(ticker) for ticker in WATCHLIST}

    # === Khởi tạo Bot ===
    bot = TradingBot()

    # Gửi tin nhắn xác nhận bot online
    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE",
        f"Bot đang chạy trên Render\n📋 Theo dõi: {', '.join(WATCHLIST)}",
        0x2ecc71
    )

    # === Bước 1: Quét tín hiệu toàn bộ danh sách ===
    bot.scan_watchlist(market_data)

    # === Bước 2: Theo dõi VN-Index ===
    print("\n" + "-"*60)
    print("📉 THEO DÕI BIẾN ĐỘNG THỊ TRƯỜNG")
    print("-"*60)
    print(bot.monitor_index("VN-Index", 1245.50))

    # === Vòng lặp giữ bot luôn chạy ===
    print("\n⏳ Bot đang theo dõi... Nhấn Ctrl+C để dừng")
    import time
    while True:
        time.sleep(60)  # Nghỉ 60 giây

