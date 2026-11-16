# ============================================================
# 🇻🇳 BOT LỌC TỰ ĐỘNG HOSE — PHIÊN BẢN CÓ VNINDEX
# Timeframe: D1 (Ngày) | Quét: 8:00 sáng T2-T6
# Tiêu chí: EMA Ribbon | Khối lượng | ADX+DI | MACD | RSI | Breakout
# Thêm: So sánh VNINDEX | Relative Strength | Lọc theo xu hướng thị trường
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

# === Tham số chỉ báo — Timeframe: D1 (Ngày) ===
EMA_SHORT = 20    # 20 ngày
EMA_MID = 50      # 50 ngày
EMA_LONG = 200    # 200 ngày
RSI_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
VOL_LOOKBACK = 20
BREAKOUT_PERIOD = 20
RS_PERIOD = 20    # So sánh mạnh/yếu hơn VNINDEX (20 phiên)

# === Ngưỡng lọc ===
MIN_VOLUME_RATIO = 1.5
MIN_ADX = 25
RSI_LOW = 55
RSI_HIGH = 75
BREAKOUT_MAX_DISTANCE = 0.03  # < 3% dưới đỉnh 20 phiên

# === VNINDEX lọc thị trường ===
MARKET_RSI_UP = 45   # Chỉ ưu tiên khi VNINDEX RSI trên mức này
MARKET_DOWN_PCT_STOP = -0.02  # Bỏ qua nếu VNINDEX giảm > 2% trong 3 phiên

# === Số mã chọn ra ===
TOP_STOCKS_LIMIT = 8

# === Quản trị rủi ro ===
RISK_PERCENT = 2.5
REWARD_RATIO = 1.5

# === Số ngày dữ liệu lịch sử ===
LOOKBACK_DAYS = 260  # Đủ để tính EMA200 + VNINDEX so sánh

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
            "footer": {"text": "📢 Hệ thống lọc tự động — Thị trường Việt Nam | Timeframe: D1 (Ngày)"}
        }
        try:
            resp = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            return resp.status_code == 204
        except Exception as e:
            print(f"⚠️ Lỗi Discord: {e}")
            return False

    def market_summary(self, vnindex_info):
        """Báo cáo tổng quan thị trường VNINDEX"""
        status_emoji = "🟢" if vnindex_info['trend'] == "TĂNG" else ("🟡" if vnindex_info['trend'] == "NGANG" else "🔴")
        desc = f"""
{status_emoji} **VNINDEX hiện tại:** {vnindex_info['close']:.2f}
📊 **Xu hướng:** {vnindex_info['trend']} (EMA20: {'✅' if vnindex_info['above_ema20'] else '❌'})
📈 **RSI thị trường:** {vnindex_info['rsi']:.1f}
📉 **Thay đổi 3 phiên:** {vnindex_info['pct_3d']:+.2f}%
💡 **Nhận định:** {vnindex_info['note']}
        """
        return self.send_embed("📊 TỔNG QUAN THỊ TRƯỜNG — VNINDEX", desc, 0x3498db)

    def new_signal(self, ticker, direction, entry, tp, sl, score, ind):
        emoji = "🟢" if direction == "MUA ✅" else "⏳"
        color = 0x2ecc71 if direction == "MUA ✅" else 0x9b59b6
        title = f"{emoji} TÍN HIỆU — {direction.split()[0]} {ticker}"
        vs_market = "✅ Mạnh hơn VNINDEX" if ind['rs_vs_market'] > 0 else "⚠️ Yếu hơn VNINDEX"
        desc = f"""
📌 **Mã:** {ticker}
📊 **Điểm tín hiệu:** {score}/100
💰 **Giá vào:** {entry}
🎯 **Mục tiêu TP:** {tp}
🛑 **Dừng lỗ SL:** {sl}
📐 **Tỷ lệ L/R:** 1 : {REWARD_RATIO}
📈 **RSI:** {ind['rsi']} | **ADX:** {ind['adx']} ({ind['trend_strength']})
📉 **MACD:** {ind['macd']}/{ind['macd_signal']} | **Vol Ratio:** {ind['vol_ratio']}
📊 **EMA Ribbon:** {ind['ema_status']} | **Cách đỉnh 20D:** {ind['breakout_pct']}%
🆚 **So sánh VNINDEX:** {vs_market} (RS{RS_PERIOD}: {ind['rs_vs_market']:+.2%})
⏱️ **Timeframe:** D1 (Ngày)
        """
        return self.send_embed(title, desc, color)

    def watchlist_update(self, count, stocks_list):
        desc = f"""
✅ Đã lọc xong sàn HOSE
📋 Số mã phân tích: {count}
📈 Danh sách theo dõi mới: {', '.join(stocks_list)}
⏱️ Timeframe phân tích: D1 (Ngày)
        """
        return self.send_embed("📋 CẬP NHẬT DANH SÁCH THEO DÕI", desc, 0x9b59b6)

    def top_stocks(self, top_list, vnindex_info):
        lines = f"📊 VNINDEX: {vnindex_info['close']:.2f} | Xu hướng: {vnindex_info['trend']}\n"
        lines += "─" * 50 + "\n"
        lines += "\n".join([
            (f"**#{i+1} {s['ticker']}** — {s['score']}/100 | {s['signal']}\n"
             f"   💰 Giá: {s['entry']} | 🎯 TP: {s['TP']} | 🛑 SL: {s['SL']}\n"
             f"   📊 RSI: {s['rsi']} | ADX: {s['adx']} | Vol: {s['vol_ratio']}x\n"
             f"   🆚 RS{RS_PERIOD}: {s['rs_vs_market']:+.2%} | Đỉnh 20D: {s['breakout_pct']}%")
            for i, s in enumerate(top_list)
        ])
        return self.send_embed(f"🏆 TOP {len(top_list)} MÃ TỐT NHẤT — Timeframe: D1", lines, 0xf39c12)

# ============================================================
# 📊 LỚP PHÂN TÍCH KỸ THUẬT
# ============================================================
class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_all_indicators(self, df, vnindex_df=None):
        df = df.copy()

        # === EMA Ribbon ===
        df['ema20'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=EMA_MID, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # === MACD ===
        df['ema12'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()

        # === ADX, +DI, -DI ===
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

        # === Khối lượng ===
        df['avg_vol_20'] = df['volume'].rolling(window=VOL_LOOKBACK).mean()
        df['vol_ratio'] = df['volume'] / df['avg_vol_20']

        # === Đỉnh 20 phiên ===
        df['high_20d'] = df['high'].rolling(window=BREAKOUT_PERIOD).max()
        df['pct_from_high_20d'] = (df['high_20d'] - df['close']) / df['high_20d']

        # === So sánh với VNINDEX — Relative Strength ===
        if vnindex_df is not None:
            df = df.merge(vnindex_df[['date','vnindex_close']], on='date', how='left')
            df['ret_20'] = df['close'].pct_change(RS_PERIOD)
            df['vnindex_ret_20'] = df['vnindex_close'].pct_change(RS_PERIOD)
            df['rs_vs_market'] = df['ret_20'] - df['vnindex_ret_20']  # >0 = mạnh hơn thị trường

        return df

    def calculate_score(self, df, market_ok=True):
        """
        Thang điểm mới — Tổng 100đ
        EMA Ribbon:       30đ
        Volume Ratio:     15đ
        ADX + DI+:       15đ
        MACD > Signal:   15đ
        RSI 55~75:       10đ
        Breakout 20D:    10đ
        Mạnh hơn VNINDEX: +5đ
        """
        latest = df.iloc[-1]
        score = 0
        checks = {}

        # 1️⃣ EMA Ribbon — 30 điểm
        close_above_ema20 = latest['close'] > latest['ema20']
        ema20_above_ema50 = latest['ema20'] > latest['ema50']
        ema50_above_ema200 = latest['ema50'] > latest['ema200']
        ema_ribbon_ok = close_above_ema20 and ema20_above_ema50 and ema50_above_ema200

        if ema_ribbon_ok:
            score += 30
            checks['ema_status'] = "✅ Hoàn hảo"
        elif close_above_ema20 and ema20_above_ema50:
            score += 20
            checks['ema_status'] = "✅ Tốt"
        elif close_above_ema20:
            score += 10
            checks['ema_status'] = "⚠️ Giá trên EMA20"
        else:
            checks['ema_status'] = "❌ Không thỏa"

        # 2️⃣ Volume Ratio — 15 điểm
        vol_ratio = latest['vol_ratio']
        if vol_ratio >= 2.0:
            score += 15
        elif vol_ratio >= MIN_VOLUME_RATIO:
            score += 10
        checks['vol_ratio'] = round(float(vol_ratio), 2)

        # 3️⃣ ADX + DI+ — 15 điểm
        adx_ok = latest['adx'] > MIN_ADX
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
        if RSI_LOW < rsi < RSI_HIGH:
            score += 10
        elif 45 < rsi <= RSI_LOW or RSI_HIGH <= rsi < 80:
            score += 5
        checks['rsi_ok'] = RSI_LOW < rsi < RSI_HIGH

        # 6️⃣ Breakout 20D — 10 điểm
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

        # ⚠️ Nếu thị trường xấu — giảm điểm tối đa
        if not market_ok:
            score = int(score * 0.7)  # Giảm 30% điểm khi thị trường không thuận lợi

        return score, checks

    def check_all_conditions(self, df, market_ok=True):
        """
        7 Điều kiện lọc chính + Điều kiện thị trường
        """
        latest = df.iloc[-1]

        cond1 = latest['close'] > latest['ema20'] > latest['ema50'] > latest['ema200']
        cond2 = latest['vol_ratio'] > MIN_VOLUME_RATIO
        cond3 = latest['adx'] > MIN_ADX
        cond4 = latest['plus_di'] > latest['minus_di']
        cond5 = latest['macd'] > latest['macd_signal']
        cond6 = RSI_LOW < latest['rsi'] < RSI_HIGH
        cond7 = latest['pct_from_high_20d'] <= BREAKOUT_MAX_DISTANCE

        all_pass = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and market_ok

        return all_pass, {
            "EMA Ribbon": cond1,
            f"Vol > {MIN_VOLUME_RATIO}x": cond2,
            f"ADX > {MIN_ADX}": cond3,
            "+DI > -DI": cond4,
            "MACD > Signal": cond5,
            f"RSI {RSI_LOW}-{RSI_HIGH}": cond6,
            "Cách đỉnh 20D < 3%": cond7,
            "Thị trường OK": market_ok
        }

    def generate_signal(self, df, ticker, vnindex_df=None, market_ok=True):
        df = self.calculate_all_indicators(df, vnindex_df)
        latest = df.iloc[-1]
        entry_price = round(float(latest['close']), 2)

        all_conditions_passed, _ = self.check_all_conditions(df, market_ok)
        score, checks = self.calculate_score(df, market_ok)

        direction = "MUA ✅" if all_conditions_passed else "THEO DÕI ⏳"

        # TP/SL
        risk = RISK_PERCENT / 100
        stop_loss = round(entry_price * (1 - risk), 2)
        risk_amount = entry_price - stop_loss
        take_profit = round(entry_price + risk_amount * REWARD_RATIO, 2)

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
            "rs_vs_market": checks['rs_vs_market']
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
# 📥 LẤY DỮ LIỆU — API VNDIRECT
# ============================================================
def get_vnindex_data(days=LOOKBACK_DAYS):
    """Lấy dữ liệu VNINDEX — So sánh thị trường"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (f"https://api-finfo.vndirect.com.vn/v4/stock_prices?sort=date"
               f"&q=code:VNINDEX~date:gte:{start_date}~date:lte:{end_date}&size=500")
        resp = requests.get(url, timeout=15, headers={            "User-Agent": "Mozilla/5.0", "Accept": "application/json"
        })
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data or len(data) < 60:
            return None
        df = pd.DataFrame(data)
        df = df.rename(columns={"close": "vnindex_close", "high": "vnindex_high", "low": "vnindex_low", "volume": "vnindex_volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Tính chỉ số VNINDEX
        df['ema20'] = df['vnindex_close'].ewm(span=20, adjust=False).mean()
        delta = df['vnindex_close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=9).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=9).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        df['pct_change_3d'] = df['vnindex_close'].pct_change(3)
        
        latest = df.iloc[-1]
        
        # Đánh giá xu hướng thị trường
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
        
        # Kiểm tra điều kiện thị trường OK
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
        if not data or len(data) < 100:
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
    """Danh sách mã lớn, thanh khoản cao HOSE"""
    print("\n" + "="*80)
    print("🔍 DANH SÁCH LỌC — TOP MÃ LỚN HOSE")
    print("📋 Timeframe: D1 (Ngày) | Có tích hợp VNINDEX")
    print("="*80)
    symbols = [
        "VNM","ACB","VCB","BID","CTG","MBB","TPB","HDB","STB","SSB",
        "VIC","VHM","VRE","MWG","FRT","PNJ","DGW","KDH","NLG","NVL",
        "FPT","MSN","GAS","PLX","POW","GVR","HPG","NKG","HSG","VND",
        "SAB","BMP","DHG","DMC","BVH","VSC","VIX","TCB","VPB","BCM",
        "SCS","REE","HAX","CII","GMD","VHC","VGT","KBC","SBT","HT1",
        "VGC","VC3","MIG","PPC","PVI","QNS","TCH"
    ]
    symbols = list(dict.fromkeys(symbols))
    print(f"✅ Số mã: {len(symbols)}")
    return symbols

# ============================================================
# 🚀 HỆ THỐNG CHÍNH
# ============================================================
class SmartStockBot:
    def __init__(self):
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.watchlist = []
        self.vnindex_data = None

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
        print(f"📋 Ngưỡng: Vol>{MIN_VOLUME_RATIO}x | ADX>{MIN_ADX} | RSI{RSI_LOW}-{RSI_HIGH} | Cách đỉnh 20D<3%")
        print("-"*80)

        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None

        for i, ticker in enumerate(candidates, 1):
            print(f"  [{i:2d}/{len(candidates)}] {ticker:<6s}...", end=" ", flush=True)
            df = get_stock_data(ticker)
            if df is not None:
                analysis = self.analyzer.generate_signal(df, ticker, vnindex_df, market_ok)
                results.append(analysis)
                print(f"Điểm: {analysis['score']:3d}/100 | {analysis['signal']} | RS: {analysis['rs_vs_market']:+.2%}")
            else:
                print("Bỏ qua (không đủ dữ liệu)")
            time.sleep(0.35)

        if not results:
            print("❌ Không có dữ liệu phân tích được!")
            return []

        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_stocks = results_sorted[:TOP_STOCKS_LIMIT]
        self.watchlist = [s['ticker'] for s in top_stocks]

        print("\n" + "="*80)
        print(f"🏆 TOP {len(top_stocks)} MÃ CÓ ĐIỂM CAO NHẤT — Timeframe: D1 (Ngày)")
        print("="*80)
        for rank, s in enumerate(top_stocks, 1):
            print(f"#{rank} {s['ticker']:<6s} | {s['score']:3d}/100 | {s['signal']} | Giá: {s['entry']} | TP: {s['TP']} | SL: {s['SL']}")
            print(f"       RSI: {s['rsi']} | ADX: {s['adx']} | Vol Ratio: {s['vol_ratio']} | RS{RS_PERIOD}: {s['rs_vs_market']:+.2%}")
        print("="*80 + "\n")

        self.notifier.watchlist_update(len(results), self.watchlist)
        self.notifier.top_stocks(top_stocks, self.vnindex_data)
        return top_stocks

    def monitor_watchlist(self):
        print("\n" + "="*60)
        print("📊 QUÉT TÍN HIỆU DANH SÁCH THEO DÕI")
        print("="*60 + "\n")
        
        # Cập nhật VNINDEX mỗi lần quét
        self.vnindex_data = get_vnindex_data()
        market_ok = self.vnindex_data['market_ok'] if self.vnindex_data else True
        vnindex_df = self.vnindex_data['df'] if self.vnindex_data else None
        
        for ticker in self.watchlist:
            df = get_stock_data(ticker)
            if df is not None:
                result = self.analyzer.generate_signal(df, ticker, vnindex_df, market_ok)
                print(f"✅ {ticker:<6s} | {result['signal']} | Điểm: {result['score']}/100 | Giá: {result['entry']}")
                if result['signal'] == "MUA ✅":
                    self.notifier.new_signal(
                        ticker, result['signal'], result['entry'],
                        result['TP'], result['SL'], result['score'],
                        {k: result[k] for k in ['rsi','macd','macd_signal','adx','trend_strength','vol_ratio','ema_status','breakout_pct','rs_vs_market']}
                    )
            time.sleep(0.35)

# ============================================================
# 🏃 CHƯƠNG TRÌNH CHÍNH
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 BOT LỌC TỰ ĐỘNG HOSE — PHIÊN BẢN CÓ VNINDEX")
    print("⏱️  Timeframe phân tích: D1 (Ngày) — Dữ liệu giá cuối phiên mỗi ngày giao dịch")
    print("="*80)
    print(f"⚙️  EMA: {EMA_SHORT}/{EMA_MID}/{EMA_LONG} ngày | RSI: {RSI_PERIOD} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL} | ADX: {ADX_PERIOD}")
    print(f"⚙️  Khối lượng: Trung bình {VOL_LOOKBACK} ngày | Breakout: {BREAKOUT_PERIOD} phiên")
    print(f"⚙️  VNINDEX: Lọc thị trường | RS{RS_PERIOD}: So sánh mạnh/yếu hơn thị trường")
    print(f"⚙️  Rủi ro: {RISK_PERCENT}% | Tỷ lệ L/R: 1:{REWARD_RATIO} | TOP: {TOP_STOCKS_LIMIT} mã")
    print(f"⏰ Chạy: 8:00 sáng Thứ 2 → Thứ 6")
    print("="*80 + "\n")

    bot = SmartStockBot()

    bot.notifier.send_embed(
        "✅ BOT ĐÃ ONLINE — PHIÊN BẢN VNINDEX",
        f"📊 Nguồn: API VNDIRECT\n⏱️ Timeframe: D1 (Ngày)\n📈 Chỉ báo: EMA Ribbon | Khối lượng | ADX+DI | MACD | RSI | Breakout\n🆚 So sánh: VNINDEX + RS{RS_PERIOD}\n⏰ Quét mỗi sáng 8:00 T2-T6",
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

        is_weekday = now.weekday() < 5  # Thứ 2 đến Thứ 6
        is_8am = now.hour == 8 and now.minute <= 5
        today_str = now.strftime("%Y-%m-%d")

        if is_weekday and is_8am and last_scan_date != today_str:
            print("\n" + "="*80)
            print(f"🕐 {now.strftime('%A %d/%m/%Y %H:%M')} — BẮT ĐẦU QUÉT DANH SÁCH MỚI")
            print("="*80)
            bot.build_watchlist()
            last_scan_date = today_str
            print("✅ Hoàn thành. Chờ đến sáng mai...\n")

