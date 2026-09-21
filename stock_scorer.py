"""
Chấm điểm tổng hợp và sàng lọc cổ phiếu
- Chạy song song 5 luồng
- Áp dụng config tối ưu từ Bayesian Optimization
- Format chi tiết 6 chỉ báo + Risk Management
"""
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import get_stock_history, get_financial_ratios
from technical_analyzer import TechnicalAnalyzer
from fundamental_analyzer import fundamental_score, passes_basic_filter
from sector_analyzer import (get_sector, sector_score, get_top_sectors,
                              rank_sectors, SECTOR_DATA)


# ============================================================
# 1. CHẤM ĐIỂM 1 CỔ PHIẾU
# ============================================================
def composite_score(symbol):
    """Chấm điểm tổng hợp cho 1 cổ phiếu."""
    result = {
        "symbol": symbol,
        "sector": get_sector(symbol),
        "technical": 0,
        "fundamental": 0,
        "sentiment": 0,
        "sector_score": 0,
        "total": 0,
        "adjusted_total": 0,
        "signal": "N/A",
        "has_fundamental": False,
        "details": {},
    }

    # ===== 1. DỮ LIỆU GIÁ =====
    df = get_stock_history(symbol)
    if df is None or len(df) < 50:
        result["details"]["error"] = "Không đủ dữ liệu giá"
        return result

    ta = TechnicalAnalyzer(df)
    tech_score, tech_details = ta.technical_score()
    result["technical"] = tech_score
    result["details"]["technical"] = tech_details

    # ===== 2. CHỈ SỐ CƠ BẢN =====
    ratios = get_financial_ratios(symbol)
    fund_score, fund_details = fundamental_score(ratios)
    result["fundamental"] = fund_score
    result["details"]["fundamental"] = fund_details
    result["details"]["ratios"] = ratios

    has_fund = (
        bool(ratios) and
        any(ratios.get(k) is not None for k in ["pe", "pb", "roe"])
    )
    result["has_fundamental"] = has_fund

    # ===== 3. SENTIMENT (mặc định 50, cập nhật từ main) =====
    result["sentiment"] = 50

    # ===== 4. ĐIỂM TỔNG HỢP =====
    if has_fund:
        w = config.SCORING_WEIGHTS
        total = (tech_score * w["technical"]
                 + fund_score * w["fundamental"]
                 + result["sentiment"] * w["sentiment"])
    else:
        w = config.SCORING_WEIGHTS_NO_FUNDAMENTAL
        total = (tech_score * w["technical"]
                 + result["sentiment"] * w["sentiment"])

    result["total"] = round(total, 1)

    # ===== 5. ĐIỂM NGÀNH =====
    sec = sector_score(result["sector"])
    result["sector_score"] = sec.get("total", 0)
    result["details"]["sector"] = sec

    # ===== 6. ĐIỂM ĐIỀU CHỈNH =====
    sw = config.SECTOR_WEIGHT
    result["adjusted_total"] = round(
        total * (1 - sw) + result["sector_score"] * sw, 1)
    result["signal"] = _get_signal(result["adjusted_total"])

    return result


def _get_signal(score):
    """Xác định tín hiệu — dùng signal_thresholds từ config tối ưu."""
    tc = getattr(config, "TRADING_CONFIG", {})
    t = tc.get("signal_thresholds", config.SIGNAL_THRESHOLDS)

    if score >= t["strong_buy"]:
        return "MUA MẠNH"
    if score >= t["buy"]:
        return "MUA"
    if score >= t["hold"]:
        return "GIỮ"
    if score >= t["sell"]:
        return "BÁN"
    return "BÁN MẠNH"


# ============================================================
# 2. SÀNG LỌC
# ============================================================
def screen_stocks(watchlist, sentiment_map=None):
    """Sàng lọc cổ phiếu — chạy song song 5 luồng."""
    all_sectors_ranked = rank_sectors()
    top_sectors = [s["sector"] for s in all_sectors_ranked[:config.TOP_SECTORS]]
    weak_sectors = [s["sector"] for s in all_sectors_ranked[-3:]]

    tc = getattr(config, "TRADING_CONFIG", {})
    min_score = tc.get("exit_threshold", 30) + 10  # = 40

    print(f"🏭 Top ngành ưu tiên: {', '.join(top_sectors)}")
    print(f"⚠️  Ngành yếu: {', '.join(weak_sectors)}")
    print(f"📊 Sẽ phân tích {len(watchlist)} mã (song song 5 luồng)")
    print(f"⚙️  Config: Entry >= {tc.get('entry_threshold', 55)} | "
          f"SL {tc.get('sl_mult', 3.0)} ATR | "
          f"TP {tc.get('tp_mult', 5.0)} ATR | "
          f"Max hold {tc.get('max_hold_days', 25)}d\n")

    def analyze_one(sym):
        try:
            sector = get_sector(sym)
            is_priority = sector in top_sectors
            is_weak = sector in weak_sectors

            score = composite_score(sym)

            if score.get("details", {}).get("error"):
                return None, {"reason": "no_data", "sym": sym}

            # Áp sentiment từ tin tức
            if sentiment_map and sym in sentiment_map:
                score["sentiment"] = sentiment_map[sym]

            # Tính lại total
            has_fund = score.get("has_fundamental", False)
            tech = score.get("technical", 0)
            fund = score.get("fundamental", 0)
            sent = score.get("sentiment", 50)

            if has_fund:
                w = config.SCORING_WEIGHTS
                total = (tech * w["technical"]
                         + fund * w["fundamental"]
                         + sent * w["sentiment"])
            else:
                w = config.SCORING_WEIGHTS_NO_FUNDAMENTAL
                total = tech * w["technical"] + sent * w["sentiment"]

            score["total"] = round(total, 1)

            # Điểm điều chỉnh theo ngành
            sw = config.SECTOR_WEIGHT
            sec_score = score.get("sector_score", 0)
            base_adjusted = total * (1 - sw) + sec_score * sw

            if is_priority:
                adjusted = base_adjusted + 5
                score["sector_bonus"] = "+5 (ưu tiên)"
            elif is_weak:
                adjusted = base_adjusted - 8
                score["sector_bonus"] = "-8 (yếu)"
            else:
                adjusted = base_adjusted
                score["sector_bonus"] = "0"

            score["adjusted_total"] = round(max(0, min(100, adjusted)), 1)
            score["signal"] = _get_signal(score["adjusted_total"])
            score["is_priority_sector"] = is_priority

            # Bộ lọc cơ bản
            ratios = score.get("details", {}).get("ratios")
            if not passes_basic_filter(ratios):
                return None, {"reason": "filter", "sym": sym}

            if score["adjusted_total"] < min_score:
                return None, {"reason": "low_score", "sym": sym}

            return score, None

        except Exception as e:
            return None, {"reason": f"error: {str(e)[:50]}", "sym": sym}

    results = []
    skipped = {"no_data": 0, "filter": 0, "low_score": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_one, sym): sym for sym in watchlist}

        completed = 0
        for future in as_completed(futures):
            sym = futures[future]
            completed += 1
            sector = get_sector(sym)
            tag = "⭐" if sector in top_sectors else (
                "⚠️" if sector in weak_sectors else "  ")

            try:
                score, skip = future.result()
                if score:
                    results.append(score)
                    print(f"  [{completed:2d}/{len(watchlist)}] {tag} {sym} "
                          f"[{sector}] → ✅ {score['adjusted_total']}/100 "
                          f"[{score['signal']}] {score['sector_bonus']}")
                else:
                    reason = skip.get("reason", "unknown")
                    if reason in skipped:
                        skipped[reason] += 1
                    else:
                        skipped["error"] += 1
                    print(f"  [{completed:2d}/{len(watchlist)}] {tag} {sym} "
                          f"[{sector}] → ⚪ {reason}")
            except Exception as e:
                print(f"  [{completed:2d}/{len(watchlist)}] {tag} {sym} → ❌ {str(e)[:50]}")
                skipped["error"] += 1

    results.sort(key=lambda x: x["adjusted_total"], reverse=True)

    print(f"\n📊 KẾT QUẢ:")
    print(f"   ✅ Đạt: {len(results)} mã")
    print(f"   ❌ Không dữ liệu: {skipped['no_data']}")
    print(f"   ⚪ Bị lọc cơ bản: {skipped['filter']}")
    print(f"   ⚪ Điểm thấp: {skipped['low_score']}")
    if skipped["error"] > 0:
        print(f"   ⚠️  Lỗi khác: {skipped['error']}")

    return results[:config.TOP_STOCKS]


# ============================================================
# 3. FORMAT BÁO CÁO
# ============================================================
def format_stock_line(stock):
    """Format cổ phiếu chi tiết cho báo cáo."""
    r = stock
    emoji = {"MUA MẠNH": "🟢", "MUA": "🟢", "MUA THĂM DÒ": "🟢",
             "CHỜ": "🟡", "KHÔNG VÀO": "⚪", "BÁN": "🔴",
             "TRUNG TÍNH": "⚪"}.get(r.get("signal", ""), "⚪")

    sector_tag = "⭐" if r.get("is_priority_sector") else ""

    tech = r.get("details", {}).get("technical", {})
    confidence = tech.get("confidence", "")
    conf_icon = {"HIGH": "🔥", "MEDIUM": "✅", "LOW": "⚠️"}.get(confidence, "")

    # Cơ bản
    ratios = r.get("details", {}).get("ratios") or {}
    has_fund = r.get("has_fundamental", False)
    if has_fund:
        pe = ratios.get("pe")
        pb = ratios.get("pb")
        roe = ratios.get("roe")
        pe_str = f"{pe:.1f}" if pe is not None else "N/A"
        pb_str = f"{pb:.2f}" if pb is not None else "N/A"
        roe_str = f"{roe:.1f}" if roe is not None else "N/A"
        fund_line = f"   <b>Cơ bản:</b> P/E {pe_str} | P/B {pb_str} | ROE {roe_str}%"
    else:
        fund_line = "   <b>Cơ bản:</b> <i>(không có dữ liệu)</i>"

    # Chỉ báo kỹ thuật
    sb = tech.get("score_breakdown", {})
    detail_lines = _format_indicator_details(sb)

    # Risk Management
    entry = tech.get("entry_action", "N/A")
    reason = tech.get("entry_reason", "")
    rm = tech.get("risk_management", {})

    rm_text = ""
    if rm.get("stop_loss"):
        close = rm.get("close", 0)
        sl = rm.get("stop_loss", 0)
        tp1 = rm.get("take_profit_1", 0)
        tp2 = rm.get("take_profit_2", 0)
        sl_pct = round((close - sl) / close * 100, 1) if close else 0
        tp1_pct = round((tp1 - close) / close * 100, 1) if close else 0
        tp2_pct = round((tp2 - close) / close * 100, 1) if close else 0
        rm_text = (
            f"\n   💰 <b>Quản trị rủi ro</b> "
            f"(SL {rm.get('sl_mult')}×ATR / TP {rm.get('tp_mult')}×ATR):\n"
            f"   • Entry: {close} | SL: {sl} (-{sl_pct}%)\n"
            f"   • TP1: {tp1} (+{tp1_pct}%) | TP2: {tp2} (+{tp2_pct}%)\n"
            f"   • R:R = {rm.get('risk_reward_ratio', 'N/A')}"
        )

    return (
        f"{emoji} <b>{r.get('symbol', '?')}</b> {sector_tag}"
        f"[{r.get('sector', '?')}] — "
        f"<b>{r.get('adjusted_total', 0)}/100</b> — "
        f"{r.get('signal', '?')} {conf_icon}\n"
        f"   <b>Điểm:</b> KT {r.get('technical', 0):.0f} | "
        f"CB {r.get('fundamental', 0):.0f} | "
        f"Tin {r.get('sentiment', 0):.0f} | "
        f"Ngành {r.get('sector_score', 0):.0f} "
        f"({r.get('sector_bonus', '0')})\n"
        f"{fund_line}\n"
        f"   <b>📊 Chỉ báo kỹ thuật:</b>\n"
        f"{detail_lines}"
        f"{rm_text}\n"
        f"   🎯 <b>Hành động: {entry}</b>\n"
        f"   💬 {reason}"
    )


def _format_indicator_details(sb):
    """Format chi tiết 6 chỉ báo."""
    lines = []

    # EMA
    ema = sb.get("ema_ribbon", {})
    if ema and ema.get("status") != "no_data":
        status_map = {
            "FULL_BULL": "✅ FULL BULL", "BULL": "✅ BULL",
            "WEAK_BULL": "🟡 CHUYỂN TIẾP", "WEAK_BEAR": "🟠 YẾU",
            "FULL_BEAR": "❌ BEAR",
        }
        st = status_map.get(ema.get("status", ""), "")
        pairs = ema.get("bull_pairs", 0)
        intra = ema.get("intraday", {}).get("signal", "")
        pos = ema.get("position", {}).get("signal", "")
        lines.append(
            f"   • <b>EMA Ribbon:</b> {ema.get('score', 0)}/25 → "
            f"Swing {pairs}/4 {st} | Intra: {intra} | Position: {pos}"
        )
    else:
        lines.append(f"   • <b>EMA Ribbon:</b> {ema.get('score', 0)}/25 → không đủ DL")

    # MACD
    macd = sb.get("macd", {})
    if macd and macd.get("status") != "no_data":
        cross = macd.get("fresh_cross", "")
        cross_tag = ""
        if cross == "bullish": cross_tag = " 🚀 CẮT LÊN"
        elif cross == "bearish": cross_tag = " 🔻 CẮT XUỐNG"
        above = "✓ trên Signal" if macd.get("above_signal") else "✗ dưới Signal"
        hist = "↗ tăng" if macd.get("hist_rising") else "↘ giảm"
        hist_pos = "dương" if macd.get("hist_positive") else "âm"
        vals = macd.get("values", {})
        vals_str = ""
        if vals:
            vals_str = (f" [MACD={vals.get('macd')} "
                        f"Sig={vals.get('signal')}]")
        lines.append(
            f"   • <b>MACD:</b> {macd.get('score', 0)}/20 → "
            f"{above} | Hist {hist_pos}, {hist}{vals_str}{cross_tag}"
        )
    else:
        lines.append(f"   • <b>MACD:</b> {macd.get('score', 0)}/20 → không đủ DL")

    # RSI
    rsi = sb.get("rsi", {})
    if rsi and rsi.get("status") != "no_data":
        value = rsi.get("value", "N/A")
        rising = "↗" if rsi.get("rising") else "↘"
        note = rsi.get("note", "")
        lines.append(
            f"   • <b>RSI(14):</b> {rsi.get('score', 0)}/15 → "
            f"{value} {rising} ({note})"
        )
    else:
        lines.append(f"   • <b>RSI:</b> {rsi.get('score', 0)}/15 → không đủ DL")

    # Volume
    vol = sb.get("volume", {})
    if vol and vol.get("status") != "no_data":
        ratio = vol.get("ratio", 0)
        status_vn = {
            "surge": "đột biến 🚀", "high": "cao",
            "normal": "bình thường", "low": "thấp",
            "very_low": "rất thấp ⚠️",
        }.get(vol.get("status", ""), "")
        lines.append(
            f"   • <b>Volume:</b> {vol.get('score', 0)}/15 → "
            f"Vol/SMA20 = {ratio}x ({status_vn})"
        )
    else:
        lines.append(f"   • <b>Volume:</b> {vol.get('score', 0)}/15 → không đủ DL")

    # Bollinger
    bb = sb.get("bollinger", {})
    if bb and bb.get("status") != "no_data":
        pos = round(bb.get("position", 0) * 100, 1)
        status_vn = {
            "near_lower": "gần dải dưới", "lower_half": "nửa dưới",
            "middle": "giữa dải", "upper_half": "nửa trên",
            "near_upper": "gần dải trên",
        }.get(bb.get("status", ""), "")
        squeeze = " 🔔 NÉN" if bb.get("squeeze") else ""
        lines.append(
            f"   • <b>Bollinger:</b> {bb.get('score', 0)}/15 → "
            f"Vị trí {pos}% ({status_vn}) | Width {bb.get('width', 0)}%{squeeze}"
        )
    else:
        lines.append(f"   • <b>Bollinger:</b> {bb.get('score', 0)}/15 → không đủ DL")

    # ATR
    atr = sb.get("atr", {})
    if atr and atr.get("status") != "no_data":
        atr_pct = atr.get("atr_pct", 0)
        status_vn = {
            "optimal": "lý tưởng", "high": "cao",
            "low": "thấp", "extreme": "cực cao ⚠️",
        }.get(atr.get("status", ""), "")
        lines.append(
            f"   • <b>ATR(14):</b> {atr.get('score', 0)}/10 → "
            f"{atr_pct}% giá ({status_vn})"
        )
    else:
        lines.append(f"   • <b>ATR:</b> {atr.get('score', 0)}/10 → không đủ DL")

    return "\n".join(lines) + "\n"


# ============================================================
# 4. TEST
# ============================================================
if __name__ == "__main__":
    # Test nhanh 5 mã
    test_list = ["VCB", "HPG", "FPT", "MWG", "VHM"]
    print("🧪 TEST STOCK_SCORER")
    print("=" * 60)
    results = screen_stocks(test_list)
    print(f"\n🏆 TOP {len(results)} CỔ PHIẾU:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {format_stock_line(r)}")
