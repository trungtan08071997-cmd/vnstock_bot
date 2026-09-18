"""
Bot phân tích chứng khoán Việt Nam
Mode:
- "vn30": Chỉ phân tích VN30 (mặc định)
- "news": Trích mã từ tin tức
- "both": Giao của VN30 và mã hot trong tin tức
"""
import sys
import time
import re
from datetime import datetime

import config
from news_crawler import crawl_all_news, filter_recent
from symbol_extractor import get_hot_symbols, analyze_news_for_symbols
from stock_scorer import screen_stocks
from news_sentiment import analyze_text
from reporter import build_report, save_report_json
from notifier import notify_all


def build_sentiment_map(articles, symbols):
    """
    Tính sentiment cho từng mã với 5 sub-indicators.
    
    Returns:
        dict: {symbol: sentiment_score (0-100)}
    """
    from news_sentiment import sentiment_score_full

    if not articles:
        return {}

    sentiment_map = {}
    for sym in symbols:
        score, details = sentiment_score_full(sym, articles)
        sentiment_map[sym] = score

    return sentiment_map


def build_sentiment_details(articles, symbols):
    """
    Lấy chi tiết sentiment cho từng mã (để hiển thị báo cáo).
    """
    from news_sentiment import sentiment_score_full

    if not articles:
        return {}

    details_map = {}
    for sym in symbols:
        score, details = sentiment_score_full(sym, articles)
        details_map[sym] = details

    return details_map


def _get_symbols_by_mode(articles, mode, top_n):
    """
    Chọn danh sách mã theo mode.
    
    Returns:
        (symbols, hot_ranking, mode_desc)
    """
    vn30 = config.VN30_LIST

    # ===== MODE VN30 =====
    if mode == "vn30":
        return sorted(vn30), [], f"VN30 ({len(vn30)} mã)"

    # ===== MODE NEWS =====
    elif mode == "news":
        if not articles:
            return config.WATCHLIST[:top_n], [], "Watchlist (không có tin)"
        hot_ranking = analyze_news_for_symbols(articles)
        symbols = get_hot_symbols(articles, top_n=top_n,
                                  min_mentions=config.MIN_MENTIONS)
        return symbols, hot_ranking, f"Tin tức ({len(symbols)} mã)"

    # ===== MODE BOTH =====
    elif mode == "both":
        if not articles:
            return sorted(vn30), [], "VN30 (không có tin)"
        hot_ranking = analyze_news_for_symbols(articles)
        hot_symbols = set(get_hot_symbols(articles, top_n=top_n * 3,
                                          min_mentions=config.MIN_MENTIONS))
        # Ưu tiên mã VN30 xuất hiện trong tin tức
        priority = [s for s in vn30 if s in hot_symbols]
        rest = [s for s in vn30 if s not in hot_symbols]
        symbols = priority + rest
        return symbols, hot_ranking, (
            f"VN30 + Tin tức ({len(priority)} mã có tin, "
            f"{len(rest)} mã không có tin)"
        )

    # ===== FALLBACK =====
    return sorted(vn30), [], "VN30 fallback"


def run_once(top_n=30, mode=None):
    """
    Chạy một lần phân tích đầy đủ.
    
    Args:
        top_n: Số mã hot tối đa (dùng cho mode news/both)
        mode: "vn30" | "news" | "both" (mặc định từ config)
    """
    mode = mode or config.ANALYSIS_MODE

    print("=" * 60)
    print("🚀 BOT PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"📋 Mode: {mode.upper()}")
    print("=" * 60)

    # ===== BƯỚC 1: CRAWL TIN TỨC =====
    print("\n📰 BƯỚC 1: CRAWL TIN TỨC")
    print("-" * 60)
    articles = crawl_all_news()
    
    if not articles:
        print("❌ Không crawl được tin. Sentiment sẽ = 50 (trung tính).")
        articles = []
    else:
        articles = filter_recent(articles, hours=config.NEWS_HOURS_BACK)
        print(f"📰 Còn {len(articles)} bài trong {config.NEWS_HOURS_BACK}h")

    # ===== BƯỚC 2: XÁC ĐỊNH DANH SÁCH MÃ =====
    print("\n🎯 BƯỚC 2: XÁC ĐỊNH DANH SÁCH CỔ PHIẾU")
    print("-" * 60)
    symbols, hot_ranking, mode_desc = _get_symbols_by_mode(articles, mode, top_n)
    print(f"📋 Danh sách: {mode_desc}")
    print(f"📋 Mã: {', '.join(symbols)}")

    if not symbols:
        print("❌ Không có mã nào để phân tích.")
        return []

    # ===== BƯỚC 3: PHÂN TÍCH =====
    print("\n📊 BƯỚC 3: PHÂN TÍCH KỸ THUẬT + CƠ BẢN + NGÀNH")
    print("-" * 60)
    
    sentiment_map = build_sentiment_map(articles, symbols) if articles else {}
    sentiment_details = build_sentiment_details(articles, symbols)

    
    # Hiển thị sentiment của các mã có tin
    if sentiment_map:
        print(f"\n📰 CHI TIẾT SENTIMENT:")
        for sym in symbols:
            if sym in sentiment_map:
                d = sentiment_details[sym]
                print(f"   {sym}: {d['total']}/100 {d['status']} "
                      f"({d['mentions']} bài, {d['sources']} nguồn)")

    stocks = screen_stocks(symbols, sentiment_map=sentiment_map)
    print(f"\n✅ Tìm được {len(stocks)} cổ phiếu tiềm năng")

    # ===== BƯỚC 4: TẠO BÁO CÁO =====
    print("\n📝 BƯỚC 4: TẠO BÁO CÁO")
    print("-" * 60)
    report = build_report(stocks, hot_ranking=hot_ranking, mode=mode)
    save_report_json(stocks, hot_ranking=hot_ranking, mode=mode)

    # ===== BƯỚC 5: GỬI THÔNG BÁO =====
    print("\n📨 BƯỚC 5: GỬI THÔNG BÁO")
    print("-" * 60)
    notify_all(report)

    return stocks


def run_schedule(interval_hours=12, mode=None):
    """Chạy theo lịch."""
    import schedule
    print(f"⏰ Bot chạy mỗi {interval_hours} giờ. Ctrl+C để dừng.\n")
    
    # Chạy ngay lần đầu
    run_once(mode=mode)
    
    # Lên lịch
    schedule.every(interval_hours).hours.do(run_once, mode=mode)
    
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    """Entry point."""
    args = sys.argv[1:]

    # Parse mode
    mode = None
    for i, a in enumerate(args):
        if a == "--mode" and i + 1 < len(args):
            mode = args[i + 1].lower()
            if mode not in ("vn30", "news", "both"):
                print(f"⚠️ Mode '{mode}' không hợp lệ. Dùng: vn30 | news | both")
                mode = None

    # Chạy theo lịch
    if "--schedule" in args:
        interval = 12
        for i, a in enumerate(args):
            if a == "--interval" and i + 1 < len(args):
                try:
                    interval = int(args[i + 1])
                except ValueError:
                    pass
        run_schedule(interval, mode=mode)
    else:
        # Chạy một lần
        top_n = 30
        for i, a in enumerate(args):
            if a == "--top" and i + 1 < len(args):
                try:
                    top_n = int(args[i + 1])
                except ValueError:
                    pass
        run_once(top_n=top_n, mode=mode)


if __name__ == "__main__":
    main()