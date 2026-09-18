"""
Trích xuất mã cổ phiếu từ tin tức và xếp hạng "hot"
"""
import re
from collections import defaultdict
#from vnstock import Listing
from news_sentiment import analyze_text
from utils_silent import silence_vnstock

_VALID_SYMBOLS = None
BLACKLIST = {
    "GDP", "CPI", "FDI", "IPO", "ETF", "USD", "EUR", "JPY", "VND",
    "CEO", "CFO", "M&A", "ROE", "ROA", "EPS", "PE", "PB",
    "FED", "ECB", "BOJ", "IMF", "WTO", "FTA", "CPTPP", "EVFTA",
    "HNX", "HOSE", "UPCOM", "SSC", "UBCK", "VSD", "SBV", "NHNN",
    "BCT", "BTC", "AI", "IT", "CNTT", "BHXH", "BHYT",
    "TTCK", "BDS", "BĐS", "NHTM", "DNNN", "KCN", "BOT",
}


def get_valid_symbols():
    """Lấy danh sách mã hợp lệ (cache)."""
    global _VALID_SYMBOLS
    if _VALID_SYMBOLS is not None:
        return _VALID_SYMBOLS

    print("📋 Đang tải danh sách mã hợp lệ...")
    symbols = set()

    # Lazy import để tránh banner khi không cần
    try:
        from vnstock import Listing
        listing = Listing(source="KBS")
        df = listing.all_symbols()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                sym = str(row.get("symbol", "")).strip().upper()
                if sym:
                    symbols.add(sym)
    except Exception as e:
        print(f"  ⚠️ Lỗi: {e}")

    if not symbols:
        # Fallback: dùng VN30
        symbols = set(config.VN30_LIST)

    print(f"  ✓ {len(symbols)} mã hợp lệ")
    _VALID_SYMBOLS = symbols
    return symbols

def extract_symbols_from_text(text):
    """Trích xuất mã CK từ đoạn text."""
    if not text:
        return []
    valid = get_valid_symbols()
    candidates = re.findall(r"\b([A-Z]{3,4})\b", text)
    found, seen = [], set()
    for c in candidates:
        cu = c.upper()
        if cu in seen or cu in BLACKLIST or cu not in valid:
            continue
        seen.add(cu)
        found.append(cu)
    return found


def analyze_news_for_symbols(articles):
    """Xếp hạng mã theo độ "hot" từ tin tức (có lưu thời gian)."""
    data = defaultdict(lambda: {
        "mentions": 0, "sources": set(),
        "sentiments": [], "titles": [],
    })
    
    for art in articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        symbols = extract_symbols_from_text(text)
        if not symbols:
            continue
        
        sent = analyze_text(text)
        pub_str = art.get("published_str", "")
        pub_dt = art.get("published_dt")
        
        for sym in symbols:
            data[sym]["mentions"] += 1
            data[sym]["sources"].add(art.get("source", "unknown"))
            data[sym]["sentiments"].append(sent)
            
            # Lưu title + thời gian + source (tối đa 5 bài)
            if len(data[sym]["titles"]) < 5:
                data[sym]["titles"].append({
                    "title": art.get("title", "")[:120],
                    "time": pub_str,
                    "source": art.get("source", ""),
                    "url": art.get("url", ""),
                })

    results = []
    for sym, d in data.items():
        mentions = d["mentions"]
        n_sources = len(d["sources"])
        avg_sent = sum(d["sentiments"]) / len(d["sentiments"]) if d["sentiments"] else 0
        freq_score = min(60, mentions / 20 * 60)
        source_score = min(20, n_sources / 5 * 20)
        sent_score = abs(avg_sent) * 20
        
        # Sắp xếp titles theo thời gian (nếu có)
        titles = d["titles"]
        
        results.append({
            "symbol": sym,
            "mentions": mentions,
            "sources": n_sources,
            "avg_sentiment": round(avg_sent, 3),
            "hot_score": round(freq_score + source_score + sent_score, 1),
            "sample_titles": titles,  # bây giờ là list of dict
        })
    
    results.sort(key=lambda x: x["hot_score"], reverse=True)
    return results

def get_hot_symbols(articles, top_n=30, min_mentions=2):
    """Lấy top N mã hot."""
    ranked = analyze_news_for_symbols(articles)
    filtered = [r for r in ranked if r["mentions"] >= min_mentions]
    if len(filtered) < top_n:
        filtered = ranked
    top = filtered[:top_n]
    
    print(f"\n🔥 TOP {len(top)} MÃ HOT:")
    for i, r in enumerate(top[:15], 1):
        icon = "🟢" if r["avg_sentiment"] > 0.1 else (
            "🔴" if r["avg_sentiment"] < -0.1 else "⚪")
        print(f"  {i:2d}. {r['symbol']:5s} — {r['mentions']:2d} mentions, "
              f"{r['sources']} nguồn, sent {icon}{r['avg_sentiment']:+.2f}, "
              f"hot={r['hot_score']:.0f}")
        # In thêm 1 tin mới nhất để kiểm chứng
        if r.get("sample_titles"):
            t = r["sample_titles"][0]
            print(f"       └ [{t.get('time', '?')}] {t.get('title', '')[:80]}")
    
    return [r["symbol"] for r in top]