"""
Daily Scanner - Scan cổ phiếu dài hạn hàng ngày
Kết hợp: Fundamental + News + Technical
Gửi Discord: Top 20 + cảnh báo tin tức mới
"""
import os
import re
import json
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from data_fetcher import get_stock_history, get_financial_ratios
from fundamental_screener import (rank_by_longterm_score,
                                    check_hard_criteria,
                                    score_fundamental_longterm)
from technical_analyzer import TechnicalAnalyzer
from news_crawler import crawl_all_news, filter_recent
from news_sentiment import analyze_text


# ============================================================
# CONFIG
# ============================================================
UNIVERSE = "vn100"          # "vn30" | "vn100" | "all"
TOP_N = 20                  # Top 20 cổ phiếu
CACHE_RATIOS_HOURS = 24 * 7 # Cache ratios 7 ngày
MIN_FUNDAMENTAL_SCORE = 40  # Điểm fundamental tối thiểu
SAVE_HISTORY_FILE = "output/scanner_history.json"


# ============================================================
# VN100 UNIVERSE (Midcap + Bluechip)
# ============================================================
VN100_UNIVERSE = [
    # VN30
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    # Midcap chất lượng
    "AAA", "ANV", "BAF", "BCG", "BFC", "BMP", "BSR", "CII", "CMG", "CSV",
    "CTD", "DBC", "DCM", "DGC", "DGW", "DIG", "DPM", "DXG", "ELC", "FRT",
    "GEX", "GIL", "HAG", "HAH", "HCM", "HDC", "HDG", "HT1", "IMP", "KBC",
    "KDC", "KDH", "LCG", "NKG", "NLG", "NT2", "NVL", "PC1", "PET", "PDR",
    "PNJ", "PPC", "PTB", "PVT", "REE", "SBT", "SCR", "SZC", "TCH", "TLG",
    "VCI", "VGC", "VHC", "VND", "VPI", "VSC", "VTP", "DHC", "TNG", "MSB",
    "OCB", "LPB", "NAB", "EIB", "KLB", "BAB", "SGB", "VBB", "PGB", "BVB",
]


# ============================================================
# LOAD RATIOS (có cache)
# ============================================================
def load_all_ratios(universe):
    """Tải ratios cho toàn universe với cache."""
    cache_file = "output/ratios_cache.json"
    
    # Đọc cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cache_time = cache.get("_cached_at", "")
            cache_dt = datetime.fromisoformat(cache_time) if cache_time else None
            if cache_dt and (datetime.now() - cache_dt).total_seconds() < CACHE_RATIOS_HOURS * 3600:
                print(f"  ✓ Dùng cache ratios ({len(cache)-1} mã)")
                return {k: v for k, v in cache.items() if k != "_cached_at"}
        except Exception:
            cache = {}
    
    print(f"  📊 Tải ratios cho {len(universe)} mã...")
    ratios_dict = {}
    
    def fetch(sym):
        try:
            r = get_financial_ratios(sym)
            return sym, r
        except Exception:
            return sym, None
    
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch, s): s for s in universe}
        for f in as_completed(futures):
            sym, r = f.result()
            if r:
                ratios_dict[sym] = r
    
    # Lưu cache
    os.makedirs("output", exist_ok=True)
    cache_data = dict(ratios_dict)
    cache_data["_cached_at"] = datetime.now().isoformat()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, default=str)
    
    return ratios_dict


# ============================================================
# LOAD NEWS + TÌM MÃ HOT
# ============================================================
def load_news_for_universe(universe, hours_back=24):
    """Crawl news, tìm mã nào có tin mới."""
    print(f"  📰 Crawl tin tức...")
    articles = crawl_all_news()
    articles = filter_recent(articles, hours=hours_back)
    
    print(f"  📰 Có {len(articles)} bài trong {hours_back}h")
    
    # Tìm mã trong universe có tin
    symbol_news = {}
    pattern_cache = {sym: re.compile(rf"\b{sym}\b", re.IGNORECASE) 
                     for sym in universe}
    
    for art in articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        for sym in universe:
            if pattern_cache[sym].search(text):
                if sym not in symbol_news:
                    symbol_news[sym] = []
                
                sentiment = analyze_text(text)
                symbol_news[sym].append({
                    "title": art.get("title", "")[:120],
                    "source": art.get("source", ""),
                    "time": art.get("published_str", ""),
                    "sentiment": round(sentiment, 2),
                })
    
    return symbol_news, articles


# ============================================================
# TECHNICAL CHECK
# ============================================================
def check_technical(symbol, ratios):
    """Chấm điểm technical ngắn hạn (timing)."""
    try:
        df = get_stock_history(symbol, 250)
        if df is None or len(df) < 100:
            return None
        
        ta = TechnicalAnalyzer(df)
        score, details = ta.technical_score()
        
        return {
            "score": score,
            "regime": details.get("regime", "N/A"),
            "entry_action": details.get("entry_action", "N/A"),
            "confidence": details.get("confidence", "N/A"),
        }
    except Exception as e:
        return None


# ============================================================
# SCAN CHÍNH
# ============================================================
def scan_daily(universe=None, top_n=TOP_N):
    """Scan hàng ngày."""
    if universe is None:
        universe = VN100_UNIVERSE
    
    print(f"\n{'='*60}")
    print(f"📅 DAILY SCANNER — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'='*60}")
    print(f"  Universe: {len(universe)} mã")
    print(f"  Top N: {top_n}\n")
    
    # ===== BƯỚC 1: RATIOS =====
    print("📊 BƯỚC 1: Tải chỉ số cơ bản")
    ratios_dict = load_all_ratios(universe)
    print(f"  ✓ Có ratios cho {len(ratios_dict)} mã\n")
    
    # ===== BƯỚC 2: FUNDAMENTAL RANKING =====
    print("🏆 BƯỚC 2: Xếp hạng theo điểm dài hạn")
    ranking = rank_by_longterm_score(ratios_dict)
    
    passed = [r for r in ranking if r["pass_hard"]]
    failed = [r for r in ranking if not r["pass_hard"]]
    
    print(f"  ✓ Pass hard criteria: {len(passed)} mã")
    print(f"  ✗ Fail: {len(failed)} mã\n")
    
    # Chỉ giữ mã có điểm >= MIN
    candidates = [r for r in passed if r["score"] >= MIN_FUNDAMENTAL_SCORE]
    print(f"  📋 Ứng viên (score >= {MIN_FUNDAMENTAL_SCORE}): {len(candidates)} mã\n")
    
    if not candidates:
        print("❌ Không có ứng viên nào")
        return []
    
    # ===== BƯỚC 3: TIN TỨC =====
    print("📰 BƯỚC 3: Tìm tin tức")
    candidate_symbols = [c["symbol"] for c in candidates]
    symbol_news, all_articles = load_news_for_universe(candidate_symbols, hours_back=24)
    print(f"  ✓ Có tin cho {len(symbol_news)} mã\n")
    
    # ===== BƯỚC 4: TECHNICAL CHECK TOP 30 =====
    print(f"📈 BƯỚC 4: Check technical top 30")
    top_candidates = candidates[:30]
    tech_results = {}
    
    def tech_fetch(c):
        sym = c["symbol"]
        ratios = ratios_dict.get(sym)
        tech = check_technical(sym, ratios)
        return sym, tech
    
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(tech_fetch, c): c for c in top_candidates}
        for f in as_completed(futures):
            sym, tech = f.result()
            if tech:
                tech_results[sym] = tech
    
    print(f"  ✓ Có technical cho {len(tech_results)} mã\n")
    
    # ===== BƯỚC 5: SCORE TỔNG HỢP =====
    print("🎯 BƯỚC 5: Tính điểm tổng hợp")
    
    final_results = []
    for c in top_candidates:
        sym = c["symbol"]
        fund_score = c["score"]
        tech = tech_results.get(sym, {})
        tech_score = tech.get("score", 50)
        
        # News score
        news = symbol_news.get(sym, [])
        if news:
            avg_sent = sum(n["sentiment"] for n in news) / len(news)
            news_score = (avg_sent + 1) / 2 * 100
        else:
            news_score = 50
        
        # ===== ĐIỂM TỔNG HỢP =====
        # 60% fundamental + 25% technical + 15% news
        total = (
            fund_score * 0.60
            + tech_score * 0.25
            + news_score * 0.15
        )
        
        final_results.append({
            "symbol": sym,
            "fundamental_score": fund_score,
            "technical_score": tech_score,
            "news_score": round(news_score, 1),
            "total_score": round(total, 1),
            "breakdown": c["breakdown"],
            "failed": c["failed"],
            "news_count": len(news),
            "news": news[:3],  # 3 tin mới nhất
            "tech": tech,
        })
    
    final_results.sort(key=lambda x: x["total_score"], reverse=True)
    top = final_results[:top_n]
    
    print(f"  ✓ Top {len(top)} cổ phiếu\n")
    
    return top


# ============================================================
# SO SÁNH VỚI NGÀY HÔM QUA
# ============================================================
def compare_with_yesterday(top_list):
    """So sánh với danh sách hôm qua."""
    if not os.path.exists(SAVE_HISTORY_FILE):
        return {"new": [], "out": [], "keep": []}
    
    try:
        with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        yesterday = set(h["symbol"] for h in history.get("yesterday_top", []))
        today = set(t["symbol"] for t in top_list)
        
        return {
            "new": list(today - yesterday),    # Mã mới vào top
            "out": list(yesterday - today),    # Mã rời top
            "keep": list(today & yesterday),   # Mã giữ nguyên
        }
    except Exception:
        return {"new": [], "out": [], "keep": []}


def save_today_top(top_list):
    """Lưu top hôm nay."""
    os.makedirs("output", exist_ok=True)
    
    history = {}
    if os.path.exists(SAVE_HISTORY_FILE):
        try:
            with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}
    
    history["yesterday_top"] = [{"symbol": t["symbol"]} for t in top_list]
    history["yesterday_date"] = datetime.now().isoformat()
    
    with open(SAVE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# FORMAT BÁO CÁO
# ============================================================
def format_daily_report(top_list, changes):
    """Format báo cáo HTML cho Discord."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    lines = []
    lines.append(f"╔══════════════════════════════════════════╗")
    lines.append(f"║  📅 DAILY SCAN - {now}          ║")
    lines.append(f"╚══════════════════════════════════════════╝")
    lines.append("")
    
    # Cảnh báo tin mới
    if changes["new"]:
        lines.append(f"🚨 **CỔ PHIẾU MỚI VÀO TOP:** {', '.join(changes['new'])}")
        lines.append("")
    
    if changes["out"]:
        lines.append(f"⚠️ **RỜI TOP:** {', '.join(changes['out'])}")
        lines.append("")
    
    lines.append(f"🏆 **TOP {len(top_list)} CỔ PHIẾU DÀI HẠN:**")
    lines.append("")
    
    for i, t in enumerate(top_list, 1):
        sym = t["symbol"]
        
        # Emoji
        if sym in changes["new"]:
            tag = "🆕"
        elif t["news_count"] >= 3:
            tag = "🚨"
        else:
            tag = "  "
        
        # Dòng chính
        lines.append(
            f"{i:2d}. {tag} **{sym}** — "
            f"**{t['total_score']}/100** "
            f"(CB {t['fundamental_score']}, KT {t['technical_score']}, "
            f"Tin {t['news_score']})"
        )
        
        # Chỉ số cơ bản
        bd = t["breakdown"]
        pe = bd.get("pe", {}).get("value")
        pb = bd.get("pb", {}).get("value")
        roe = bd.get("roe", {}).get("value")
        
        pe_s = f"{pe:.1f}" if pe else "N/A"
        pb_s = f"{pb:.2f}" if pb else "N/A"
        roe_s = f"{roe:.1f}" if roe else "N/A"
        
        lines.append(f"    • P/E {pe_s} | P/B {pb_s} | ROE {roe_s}%")
        
        # Tin tức
        if t["news"]:
            for n in t["news"][:2]:
                sent_icon = "🟢" if n["sentiment"] > 0.1 else (
                    "🔴" if n["sentiment"] < -0.1 else "⚪")
                lines.append(f"    {sent_icon} [{n['time']}] {n['title'][:80]}")
        
        # Technical
        tech = t.get("tech", {})
        if tech:
            lines.append(f"    📊 {tech.get('entry_action', 'N/A')} "
                         f"({tech.get('regime', 'N/A')}, "
                         f"{tech.get('confidence', 'N/A')})")
        
        lines.append("")
    
    # Footer
    lines.append("─" * 45)
    lines.append("⚠️ *Báo cáo chỉ mang tính tham khảo*")
    lines.append("💡 *Bạn tự quyết định mua mã nào, bao nhiêu vốn*")
    
    return "\n".join(lines)


# ============================================================
# GỬI DISCORD
# ============================================================
def send_to_discord(message):
    """Gửi báo cáo lên Discord."""
    webhook_url = config.DISCORD_WEBHOOK_URL
    if not webhook_url or "PASTE_URL" in webhook_url:
        print("⚠️ Chưa cấu hình Discord webhook")
        return False
    
    # Discord giới hạn 2000 ký tự
    max_len = 1900
    chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
    
    ok = True
    for idx, chunk in enumerate(chunks, 1):
        try:
            r = requests.post(webhook_url, json={
                "content": chunk,
                "username": "Daily Scanner",
            }, timeout=10)
            if r.status_code not in (200, 204):
                print(f"  ⚠️ Discord lỗi {r.status_code}")
                ok = False
        except Exception as e:
            print(f"  ⚠️ {e}")
            ok = False
    
    return ok


# ============================================================
# MAIN
# ============================================================
def main():
    print("╔══════════════════════════════════════════╗")
    print("║   DAILY LONG-TERM SCANNER                ║")
    print("╚══════════════════════════════════════════╝")
    
    # Scan
    top = scan_daily()
    
    if not top:
        print("❌ Không có kết quả")
        return
    
    # So sánh
    changes = compare_with_yesterday(top)
    if changes["new"]:
        print(f"🚨 Mã mới vào top: {', '.join(changes['new'])}")
    
    # Format
    report = format_daily_report(top, changes)
    
    # In console
    import re
    plain = re.sub(r"\*\*|\*", "", report)
    print("\n" + plain)
    
    # Lưu
    os.makedirs("output", exist_ok=True)
    with open("output/daily_scan_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "top": top,
            "changes": changes,
        }, f, ensure_ascii=False, indent=2, default=str)
    
    # Gửi Discord
    print("\n📨 Gửi Discord...")
    if send_to_discord(report):
        print("  ✓ Đã gửi")
    
    # Lưu lịch sử
    save_today_top(top)
    print("  ✓ Đã lưu lịch sử")


if __name__ == "__main__":
    main()
