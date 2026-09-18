"""
Crawl tin tức từ nhiều nguồn RSS - có chuẩn hóa thời gian
"""
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

RSS_SOURCES = {
    "cafef": "https://cafef.vn/home.rss",
    "cafef_stock": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "vnexpress_business": "https://vnexpress.net/rss/kinh-doanh.rss",
    "vietstock": "https://vietstock.vn/144/chung-khoan.rss",
    "thanhnien_business": "https://thanhnien.vn/rss/kinh-te.rss",
    "tuoitre_business": "https://tuoitre.vn/rss/kinh-doanh.rss",
}

# Múi giờ Việt Nam (UTC+7)
VN_TZ = timezone(timedelta(hours=7))


def _parse_pubdate(pub_str):
    """
    Chuẩn hóa chuỗi ngày từ RSS về datetime (giờ VN).
    Trả về datetime hoặc None nếu không parse được.
    """
    if not pub_str:
        return None
    
    # Các format phổ biến trong RSS Việt Nam
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",   # Tue, 17 Sep 2026 15:30:00 +0700
        "%a, %d %b %Y %H:%M:%S %Z",   # Tue, 17 Sep 2026 15:30:00 GMT
        "%a, %d %b %Y %H:%M:%S",      # Không có timezone
        "%Y-%m-%dT%H:%M:%S%z",        # 2026-09-17T15:30:00+07:00
        "%Y-%m-%dT%H:%M:%SZ",         # 2026-09-17T15:30:00Z
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(pub_str.strip(), fmt)
            # Nếu không có timezone → coi như giờ VN
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=VN_TZ)
            # Chuyển về giờ VN
            return dt.astimezone(VN_TZ)
        except ValueError:
            continue
    
    return None


def _format_time_vn(dt):
    """Format thời gian dạng: '17/09 15:30' hoặc 'Hôm nay 15:30'."""
    if dt is None:
        return ""
    
    now = datetime.now(VN_TZ)
    delta = now - dt
    
    # Trong vòng 1 giờ
    if delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() // 60)
        if minutes < 1:
            return "vừa xong"
        return f"{minutes} phút trước"
    
    # Trong ngày hôm nay
    if dt.date() == now.date():
        return f"Hôm nay {dt.strftime('%H:%M')}"
    
    # Hôm qua
    if (now.date() - dt.date()).days == 1:
        return f"Hôm qua {dt.strftime('%H:%M')}"
    
    # Cũ hơn
    return dt.strftime("%d/%m %H:%M")


def fetch_rss(url, source_name, timeout=15):
    """Lấy tin từ một RSS feed."""
    articles = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.encoding = "utf-8"
        root = ET.fromstring(resp.content)
        
        for item in root.findall(".//item"):
            title = _get_text(item, "title")
            link = _get_text(item, "link")
            desc = _get_text(item, "description")
            pub_raw = _get_text(item, "pubDate")
            
            if not title:
                continue
            
            # Parse thời gian
            pub_dt = _parse_pubdate(pub_raw)
            
            articles.append({
                "title": title,
                "summary": _clean_html(desc),
                "url": link,
                "source": source_name,
                "published_raw": pub_raw,
                "published_dt": pub_dt,           # datetime object
                "published_str": _format_time_vn(pub_dt),  # chuỗi hiển thị
            })
    except Exception as e:
        print(f"  ⚠️ Lỗi RSS {source_name}: {str(e)[:80]}")
    return articles


def _get_text(parent, tag):
    el = parent.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def _clean_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def crawl_all_news(max_workers=6):
    """Crawl tất cả nguồn song song."""
    all_articles = []
    print("📰 Đang crawl tin tức từ các nguồn...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_rss, url, name): name
                   for name, url in RSS_SOURCES.items()}
        for future in as_completed(futures):
            source = futures[future]
            try:
                articles = future.result()
                all_articles.extend(articles)
                print(f"  ✓ {source}: {len(articles)} bài")
            except Exception as e:
                print(f"  ✗ {source}: {str(e)[:80]}")

    # Loại trùng theo URL
    seen, unique = set(), []
    for art in all_articles:
        key = art.get("url") or art.get("title", "")[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(art)
    
    # Sắp xếp mới nhất lên đầu
    unique.sort(
        key=lambda a: a.get("published_dt") or datetime.min.replace(tzinfo=VN_TZ),
        reverse=True,
    )
    
    print(f"📰 Tổng: {len(unique)} bài từ {len(set(a['source'] for a in unique))} nguồn")
    return unique


def filter_recent(articles, hours=72):
    """Lọc bài trong N giờ gần đây."""
    if not articles:
        return articles
    
    cutoff = datetime.now(VN_TZ) - timedelta(hours=hours)
    recent = []
    
    for art in articles:
        dt = art.get("published_dt")
        if dt is None:
            # Không parse được → giữ lại
            recent.append(art)
        elif dt >= cutoff:
            recent.append(art)
    
    return recent