# ============================================================
# 🇻🇳 BOT LỌC TỰ ĐỘNG & PHÂN TÍCH — CHỌN TOP MÃ TỐT NHẤT HOSE
# Lấy giá THẬT qua API TCBS công khai (không cần vnstock)
# Thông báo Discord: Tín hiệu | Cập nhật danh sách | SL/TP
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
# 🔧 CẤU HÌNH — TÙY CHỈNH THEO NHU CẦU
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

# === Số mã chọn ra để theo dõi ===
TOP_STOCKS_LIMIT = 8

# === Quản trị rủi ro ===
RISK_PERCENT = 2.5
REWARD_RATIO = 1.5

# === Số ngày dữ liệu lịch sử ===
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
            resp = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
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
📋 Số mã phân tích: {count}
📈 Danh sách theo dõi mới: {', '.join(stocks_list)}
        """
        return self.send_embed("📋 CẬP NHẬT DANH SÁCH THEO DÕI", desc, 0x9b59b6)

    def top_stocks(self, top_list):
        lines = "\n".join([
            f"#{i+1} **{s['ticker']}** — Điểm: {s['score']}/100 | {s['signal']} | Giá: {s['entry']}"
            for i, s in enumerate(top_list)
        ])
        return self.send_embed(f"🏆 TOP {len(top_list)} MÃ TỐT NHẤT", lines, 0xf39c12)

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
                score += 10

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
        entry_price = round(float(latest['close']), 2)
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
# 📥 LẤY DỮ LIỆU GIÁ THỰC — API TCBS CÔNG KHAI
# ============================================================
def get_stock_data(ticker, days=LOOKBACK_DAYS):
    """Lấy dữ liệu giá THẬT từ API TCBS — không cần vnstock"""
    try:
        end_ts = int(time.time())
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
        url = (f"https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"
               f"?ticker={ticker}&type=stock&resolution=D&from={start_ts}&to={end_ts}")
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data or len(data) < 50:
            return None
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "tradingDate": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df
    except Exception:
        return None


def get_filtered_hose_stocks():
    """Danh sách lọc sẵn — Top 60 mã lớn, thanh khoản cao HOSE"""
    print("\n" + "="*60)
    print("🔍 DANH SÁCH LỌC SẴN — TOP 60 MÃ LỚN NHẤT HOSE")
    print("📋 Tiêu chí: Vốn hóa lớn | Thanh khoản cao | An toàn cuối năm")
    print("="*60)
    symbols = [
        "VNM","ACB","VCB","BID","CTG","MBB","TPB","HDB","STB","SSB",
        "VIC","VHM","VRE","MWG","FRT","PNJ","DGW","KDH","NLG","NVL",
        "FPT","MSN","GAS","PLX","POW","GVR","HPG","NKG","HSG","VND",
        "SAB","BMP","DHG","DMC","BVH","VSC","VIX","TCB","VPB","BCM",
        "SCS","REE","HAX","CII","GMD","VHC","VGT","KBC","SBT","HT1",
        "VGC","VC3","MIG","PPC","PVI","QNS","TCH","SAB","VNM","FPT"
    ]
    # Loại trùng
    symbols = list(dict.fromkeys(symbols))
    print(f"✅ Số mã trong danh sách lọc: {len(symbols)}")
    return symbols

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
            print(f"  [{i}/{len(candidates)}] {ticker}...", end=" ", flush=True)
            df = get_stock_data(ticker)
            if df is not None:
                analysis = self.analyzer.generate_signal(df, ticker)
                results.append(analysis)
                print(f"Điểm: {analysis['score']} | {analysis['signal']}")
            else:
                print("Bỏ qua")
            time.sleep(0.5)

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        # Sắp xếp theo điểm giảm dần, chọn TOP
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        print("\n" + "="*60)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ TÍN HIỆU TỐT NHẤT")
        print("="*60)
        for rank, stock in enumerate(top_stocks, 1):
            print(f"#{rank} {stock['ticker']:6s} | Điểm: {stock['score']:3d}/100 | {stock['signal']:8s} | Giá: {stock['entry']}")
        print("="*60 + "\n")

        # Gửi thông báo Discord
        self.notifier.watchlist_update(len(results), self.watchlist)
        self.notifier.top_stocks(top_stocks)

        return top_stocks

    def monitor_watchlist(self):
        """Theo dõi danh sách đã chọn — gửi tín hiệu MUA/BÁN"""
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
    print(f"⚙️  RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}")
    print(f"⚙️  Rủi ro: {RISK_PERCENT}% | Tỷ lệ L/R: 1:{REWARD_RATIO}")
    print(f"⚙️  Số mã chọn ra: TOP {TOP_STOCKS_LIMIT}")
    print("="*60 + "\n")

    bot = SmartStockBot()

    # Gửi xác nhận online
    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — LỌC TỰ ĐỘNG HOSE",
        f"📊 Nguồn dữ liệu: API TCBS (giá thật)\n🎯 Sẽ chọn TOP {TOP_STOCKS_LIMIT} mã tốt nhất",
        0x2ecc71
    )

    # Bước 1: Lọc & chọn danh sách tốt nhất
    top_stocks = bot.build_watchlist()

    # Bước 2: Theo dõi liên tục — quét lại mỗi giờ
    print("\n⏳ Bot đang theo dõi... Quét lại mỗi 60 phút")
    last_monitor = datetime.now()

    while True:
        time.sleep(60)
        elapsed = (datetime.now() - last_monitor).total_seconds()
        if elapsed >= 3600:
            print("\n" + "-"*60)
            print(f"🕐 CẬP NHẬT TÍN HIỆU — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*60)
            bot.monitor_watchlist()
            last_monitor = datetime.now()
