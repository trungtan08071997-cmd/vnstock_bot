# ============================================================
# 🇻🇳 BOT LỌC TỰ ĐỘNG & PHÂN TÍCH — TOP MÃ TỐT NHẤT HOSE
# Chỉ báo: RSI(9) | MACD(10,21,7) | SMA(10,40) | ADX(14) | Bollinger
# TP/SL tự động | Thông báo Discord đầy đủ chỉ số
# Nguồn giá: API VNDIRECT (không bị chặn trên Render)
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

# === Tham số chỉ báo — Tối ưu thị trường VN ===
RSI_PERIOD = 9
MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 7
SMA_SHORT = 10
SMA_LONG = 40
ADX_PERIOD = 14
BB_PERIOD = 18
BB_STD = 1.8

# === Số mã chọn ra ===
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

    def new_signal(self, ticker, direction, entry, tp, sl, score, ind):
        emoji = "🟢" if direction == "MUA" else "🔴"
        color = 0x2ecc71 if direction == "MUA" else 0xe67e22
        title = f"{emoji} TÍN HIỆU MỚI — {direction} {ticker}"
        desc = f"""
📌 **Mã:** {ticker}
📊 **Điểm tín hiệu:** {score}/100
💰 **Giá vào:** {entry}
🎯 **Mục tiêu TP:** {tp}
🛑 **Dừng lỗ SL:** {sl}
📐 **Tỷ lệ L/R:** 1 : {REWARD_RATIO}
📈 **RSI:** {ind['rsi']} | **ADX:** {ind['adx']} ({ind['trend_strength']})
📉 **MACD:** {ind['macd']}/{ind['macd_signal']} | **Xu hướng:** {ind['trend']}
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
            (f"**#{i+1} {s['ticker']}** — {s['score']}/100 | {s['signal']}\n"
             f"   💰 Giá: {s['entry']} | 🎯 TP: {s['TP']} | 🛑 SL: {s['SL']}\n"
             f"   📊 RSI: {s['rsi']} | ADX: {s['adx']} ({s['trend_strength']}) | Xu hướng: {s['trend']}")
            for i, s in enumerate(top_list)
        ])
        return self.send_embed(f"🏆 TOP {len(top_list)} MÃ TỐT NHẤT — KÈM TP/SL & CHỈ SỐ", lines, 0xf39c12)

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

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=BB_PERIOD).mean()
        bb_std = df['close'].rolling(window=BB_PERIOD).std()
        df['bb_upper'] = df['bb_mid'] + BB_STD * bb_std
        df['bb_lower'] = df['bb_mid'] - BB_STD * bb_std

        # ADX (14) — Đo lực xu hướng
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

        return df

    def calculate_score(self, df):
        """Tính điểm tổng hợp 0-100 — càng cao càng tốt"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0

        # 1. Xu hướng giá trên SMA (20 điểm)
        if latest['close'] > latest[f'sma{SMA_SHORT}']: score += 10
        if latest[f'sma{SMA_SHORT}'] > latest[f'sma{SMA_LONG}']: score += 10

        # 2. RSI hợp lý (20 điểm)
        if 45 < latest['rsi'] < 70: score += 20
        elif 35 < latest['rsi'] < 75: score += 12

        # 3. MACD (20 điểm)
        if latest['macd'] > latest['macd_signal']:
            score += 12
            if prev['macd'] <= prev['macd_signal']: score += 8

        # 4. ADX — lực xu hướng (15 điểm)
        if latest['adx'] > 25: score += 15
        elif latest['adx'] > 20: score += 8

        # 5. Vị trí Bollinger (15 điểm)
        if latest['close'] > latest['bb_mid']: score += 15

        # 6. Khối lượng tăng (10 điểm)
        avg_vol = df['volume'].tail(10).mean()
        if latest['volume'] > avg_vol * 1.1: score += 10

        return score

    def generate_signal(self, df, ticker):
        df = self.calculate_all_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        entry_price = round(float(latest['close']), 2)
        score = self.calculate_score(df)
        adx_val = round(float(latest['adx']), 1)

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

        trend = "TĂNG" if entry_price > latest[f'sma{SMA_SHORT}'] else "GIẢM"
        trend_strength = "MẠNH" if adx_val > 25 else ("TRUNG BÌNH" if adx_val > 20 else "YẾU/ĐI NGANG")

        ind = {
            "rsi": round(float(latest['rsi']), 1),
            "macd": round(float(latest['macd']), 2),
            "macd_signal": round(float(latest['macd_signal']), 2),
            "adx": adx_val,
            "trend": trend,
            "trend_strength": trend_strength
        }

        return {
            "ticker": ticker,
            "signal": direction,
            "score": score,
            "entry": entry_price,
            "TP": take_profit,
            "SL": stop_loss,
            **ind
        }

# ============================================================
# 📥 LẤY DỮ LIỆU GIÁ THỰC — API VNDIRECT
# ============================================================
def get_stock_data(ticker, days=LOOKBACK_DAYS):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:{ticker}~date:gte:{start_date}~date:lte:{end_date}&size=500")
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "application/json"
        })
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data or len(data) < 50:
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
        return df
    except Exception:
        return None


def get_filtered_hose_stocks():
    """Danh sách lọc sẵn — Top mã lớn, thanh khoản cao HOSE"""
    print("\n" + "="*60)
    print("🔍 DANH SÁCH LỌC SẴN — TOP MÃ LỚN NHẤT HOSE")
    print("📋 Tiêu chí: Vốn hóa lớn | Thanh khoản cao | An toàn cuối năm")
    print("="*60)
    symbols = [
        "VNM","ACB","VCB","BID","CTG","MBB","TPB","HDB","STB","SSB",
        "VIC","VHM","VRE","MWG","FRT","PNJ","DGW","KDH","NLG","NVL",
        "FPT","MSN","GAS","PLX","POW","GVR","HPG","NKG","HSG","VND",
        "SAB","BMP","DHG","DMC","BVH","VSC","VIX","TCB","VPB","BCM",
        "SCS","REE","HAX","CII","GMD","VHC","VGT","KBC","SBT","HT1",
        "VGC","VC3","MIG","PPC","PVI","QNS","TCH","SBT","VNM","FPT"
    ]
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

    def build_watchlist(self):
        candidates = get_filtered_hose_stocks()
        results = []
        print(f"\n⏳ Đang phân tích {len(candidates)} mã... (có thể mất vài phút)")

        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i}/{len(candidates)}] {ticker}...", end=" ", flush=True)
            df = get_stock_data(ticker)
            if df is not None:
                analysis = self.analyzer.generate_signal(df, ticker)
                results.append(analysis)
                print(f"Điểm: {analysis['score']} | {analysis['signal']} | RSI: {analysis['rsi']} | ADX: {analysis['adx']}")
            else:
                print("Bỏ qua")
            time.sleep(0.4)

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        print("\n" + "="*60)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ TÍN HIỆU TỐT NHẤT")
        print("="*60)
        for rank, s in enumerate(top_stocks, 1):
            print(f"#{rank} {s['ticker']:6s} | {s['score']:3d}/100 | {s['signal']:8s} | Giá: {s['entry']} | TP: {s['TP']} | SL: {s['SL']} | RSI: {s['rsi']} | ADX: {s['adx']}")
        print("="*60 + "\n")

        self.notifier.watchlist_update(len(results), self.watchlist)
        self.notifier.top_stocks(top_stocks)
        return top_stocks

    def monitor_watchlist(self):
        print("\n" + "="*60)
        print("📊 QUÉT TÍN HIỆU DANH SÁCH THEO DÕI")
        print("="*60 + "\n")
        for ticker in self.watchlist:
            df = get_stock_data(ticker)
            if df is not None:
                result = self.analyzer.generate_signal(df, ticker)
                if result['signal'] in ['MUA', 'BÁN']:
                    print(f"✅ {ticker}: {result['signal']} | Vào: {result['entry']} | TP: {result['TP']} | SL: {result['SL']}")
                    ind = {k: result[k] for k in ['rsi','macd','macd_signal','adx','trend','trend_strength']}
                    self.notifier.new_signal(
                        ticker, result['signal'], result['entry'],
                        result['TP'], result['SL'], result['score'], ind
                    )
                else:
                    print(f"⏳ {ticker}: Điểm {result['score']}/100 — Chờ tín hiệu")
            time.sleep(0.4)

# ============================================================
# 🏃 CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BOT LỌC TỰ ĐỘNG HOSE — FULL CHỈ SỐ + TP/SL")
    print("="*60)
    print(f"⚙️  RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL} | ADX: {ADX_PERIOD}")
    print(f"⚙️  Rủi ro: {RISK_PERCENT}% | Tỷ lệ L/R: 1:{REWARD_RATIO} | TOP: {TOP_STOCKS_LIMIT} mã")
    print("="*60 + "\n")

    bot = SmartStockBot()

    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — FULL CHỈ SỐ + TP/SL",
        f"📊 Nguồn: API VNDIRECT (giá thật)\n📈 Chỉ báo: RSI | MACD | SMA | ADX | Bollinger\n🎯 Sẽ chọn TOP {TOP_STOCKS_LIMIT} mã tốt nhất",
        0x2ecc71
    )

    bot.build_watchlist()

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
